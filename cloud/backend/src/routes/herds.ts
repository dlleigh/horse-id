import { Router } from "express";
import { eq, sql, and, ne } from "drizzle-orm";
import { db } from "../db/client.js";
import { herds, horses, photos } from "../db/schema.js";
import { getConfig } from "../services/config.js";
import { createFolder, renameFolder, trashFolder } from "../services/drive.js";

const router = Router();

// GET /api/herds — list all herds with horse and photo counts
router.get("/", async (_req, res) => {
  const results = await db.execute(sql`
    SELECT h.id, h.name,
      (SELECT count(*)::int FROM horses WHERE horses.herd_id = h.id) as "horseCount",
      (SELECT count(*)::int FROM photos p JOIN horses ho ON ho.id = p.horse_id WHERE ho.herd_id = h.id) as "photoCount"
    FROM herds h
    ORDER BY h.name
  `);

  res.json(results.rows);
});

// GET /api/herds/:id/horses — horses in a herd with photo counts
router.get("/:id/horses", async (req, res) => {
  const herdId = Number(req.params.id);

  const results = await db.execute(sql`
    SELECT h.id, h.name, h.status,
      (SELECT count(*)::int FROM photos WHERE photos.horse_id = h.id) as "photoCount",
      (SELECT count(*)::int FROM photos WHERE photos.horse_id = h.id AND processing_status = 'ready') as "readyCount",
      (SELECT p.id FROM photos p WHERE p.horse_id = h.id AND p.excluded = false
       ORDER BY p.processing_status = 'ready' DESC, p.id LIMIT 1) as "thumbnailPhotoId"
    FROM horses h
    WHERE h.herd_id = ${herdId}
    ORDER BY h.name
  `);

  res.json(results.rows);
});

// POST /api/herds — create a new herd (and Drive folder)
router.post("/", async (req, res) => {
  const { name } = req.body;
  if (!name || typeof name !== "string" || !name.trim()) {
    return res.status(400).json({ error: "Name is required" });
  }

  const trimmed = name.trim();

  // Check for duplicate name
  const existing = await db
    .select({ id: herds.id })
    .from(herds)
    .where(eq(herds.name, trimmed))
    .limit(1);
  if (existing.length > 0) {
    return res.status(400).json({ error: `A herd named "${trimmed}" already exists` });
  }

  try {
    const config = getConfig();
    const driveFolderId = await createFolder(trimmed, config.googleDriveDirectoryId);

    const [newHerd] = await db
      .insert(herds)
      .values({ name: trimmed, driveFolderId })
      .returning({ id: herds.id, name: herds.name });

    res.json({ id: newHerd.id, name: newHerd.name, horseCount: 0, photoCount: 0 });
  } catch (err: any) {
    console.error("[herds] Failed to create herd:", err);
    res.status(500).json({ error: "Failed to create herd" });
  }
});

// PATCH /api/herds/:id — rename a herd (and its Drive folder)
router.patch("/:id", async (req, res) => {
  const herdId = Number(req.params.id);
  const { name } = req.body;
  if (!name || typeof name !== "string" || !name.trim()) {
    return res.status(400).json({ error: "Name is required" });
  }

  const trimmed = name.trim();

  const [herd] = await db
    .select({ id: herds.id, name: herds.name, driveFolderId: herds.driveFolderId })
    .from(herds)
    .where(eq(herds.id, herdId))
    .limit(1);
  if (!herd) {
    return res.status(404).json({ error: "Herd not found" });
  }

  // Check for duplicate name (excluding this herd)
  const duplicate = await db
    .select({ id: herds.id })
    .from(herds)
    .where(and(eq(herds.name, trimmed), ne(herds.id, herdId)))
    .limit(1);
  if (duplicate.length > 0) {
    return res.status(400).json({ error: `A herd named "${trimmed}" already exists` });
  }

  try {
    await renameFolder(herd.driveFolderId, trimmed);
    await db.update(herds).set({ name: trimmed }).where(eq(herds.id, herdId));
    res.json({ id: herd.id, name: trimmed });
  } catch (err: any) {
    console.error("[herds] Failed to rename herd:", err);
    res.status(500).json({ error: "Failed to rename herd" });
  }
});

// DELETE /api/herds/:id — remove an empty herd (and trash its Drive folder)
router.delete("/:id", async (req, res) => {
  const herdId = Number(req.params.id);

  const [herd] = await db
    .select({ id: herds.id, driveFolderId: herds.driveFolderId })
    .from(herds)
    .where(eq(herds.id, herdId))
    .limit(1);
  if (!herd) {
    return res.status(404).json({ error: "Herd not found" });
  }

  // Check for horses
  const [{ count }] = await db
    .select({ count: sql<number>`count(*)::int` })
    .from(horses)
    .where(eq(horses.herdId, herdId));
  if (count > 0) {
    return res
      .status(400)
      .json({ error: `Cannot delete herd: it still has ${count} horse${count === 1 ? "" : "s"}` });
  }

  try {
    await trashFolder(herd.driveFolderId);
    await db.delete(herds).where(eq(herds.id, herdId));
    res.json({ success: true });
  } catch (err: any) {
    console.error("[herds] Failed to delete herd:", err);
    res.status(500).json({ error: "Failed to delete herd" });
  }
});

export default router;

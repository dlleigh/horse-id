import { Router } from "express";
import { eq, sql, and } from "drizzle-orm";
import { db } from "../db/client.js";
import { horses, herds, photos } from "../db/schema.js";
import { moveFolder } from "../services/drive.js";

const router = Router();

// GET /api/horses — search horses by name
router.get("/", async (req, res) => {
  const q = String(req.query.q || "").trim();
  if (!q) {
    res.json([]);
    return;
  }

  const results = await db.execute(sql`
    SELECT h.id, h.name, hd.name AS "herdName"
    FROM horses h
    JOIN herds hd ON hd.id = h.herd_id
    WHERE h.name ILIKE ${'%' + q + '%'}
    ORDER BY h.name
    LIMIT 10
  `);

  res.json(results.rows);
});

// GET /api/horses/:id — horse detail with photos
router.get("/:id", async (req, res) => {
  const horseId = Number(req.params.id);

  const [horse] = await db
    .select({
      id: horses.id,
      name: horses.name,
      status: horses.status,
      herdId: horses.herdId,
      herdName: herds.name,
    })
    .from(horses)
    .innerJoin(herds, eq(herds.id, horses.herdId))
    .where(eq(horses.id, horseId));

  if (!horse) {
    res.status(404).json({ error: "Horse not found" });
    return;
  }

  const horsePhotos = await db
    .select({
      id: photos.id,
      filename: photos.filename,
      processingStatus: photos.processingStatus,
      detectionResult: photos.detectionResult,
      excluded: photos.excluded,
    })
    .from(photos)
    .where(eq(photos.horseId, horseId))
    .orderBy(photos.filename);

  res.json({ ...horse, photos: horsePhotos });
});

// POST /api/horses/:id/move — move a horse to a different herd
router.post("/:id/move", async (req, res) => {
  try {
    const horseId = Number(req.params.id);
    const { herdId: destHerdId } = req.body;

    if (!destHerdId || typeof destHerdId !== "number") {
      res.status(400).json({ error: "herdId is required" });
      return;
    }

    // Fetch the horse with its current herd info
    const [horse] = await db
      .select({
        id: horses.id,
        name: horses.name,
        herdId: horses.herdId,
        driveFolderId: horses.driveFolderId,
        currentHerdDriveFolderId: herds.driveFolderId,
      })
      .from(horses)
      .innerJoin(herds, eq(herds.id, horses.herdId))
      .where(eq(horses.id, horseId));

    if (!horse) {
      res.status(404).json({ error: "Horse not found" });
      return;
    }

    if (horse.herdId === destHerdId) {
      res.status(400).json({ error: "Horse is already in that herd" });
      return;
    }

    // Fetch the destination herd
    const [destHerd] = await db
      .select()
      .from(herds)
      .where(eq(herds.id, destHerdId));

    if (!destHerd) {
      res.status(404).json({ error: "Destination herd not found" });
      return;
    }

    // Check for name collision in destination herd
    const [existing] = await db
      .select({ id: horses.id })
      .from(horses)
      .where(and(eq(horses.herdId, destHerdId), eq(horses.name, horse.name)));

    if (existing) {
      res.status(409).json({
        error: `A horse named "${horse.name}" already exists in ${destHerd.name}`,
      });
      return;
    }

    // Move the Drive folder
    await moveFolder(
      horse.driveFolderId,
      horse.currentHerdDriveFolderId,
      destHerd.driveFolderId
    );

    // Update the DB
    await db
      .update(horses)
      .set({ herdId: destHerdId })
      .where(eq(horses.id, horseId));

    res.json({ success: true, herdId: destHerdId, herdName: destHerd.name });
  } catch (err) {
    console.error("Failed to move horse:", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    res.status(500).json({ error: `Move failed: ${message}` });
  }
});

export default router;

import { Router } from "express";
import { eq, sql } from "drizzle-orm";
import { db } from "../db/client.js";
import { herds, horses, photos } from "../db/schema.js";

const router = Router();

// GET /api/herds — list all herds with horse and photo counts
router.get("/", async (_req, res) => {
  const results = await db.execute(sql`
    SELECT h.id, h.name,
      (SELECT count(*) FROM horses WHERE horses.herd_id = h.id) as "horseCount",
      (SELECT count(*) FROM photos p JOIN horses ho ON ho.id = p.horse_id WHERE ho.herd_id = h.id) as "photoCount"
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
      (SELECT count(*) FROM photos WHERE photos.horse_id = h.id) as "photoCount",
      (SELECT count(*) FROM photos WHERE photos.horse_id = h.id AND processing_status = 'ready') as "readyCount"
    FROM horses h
    WHERE h.herd_id = ${herdId}
    ORDER BY h.name
  `);

  res.json(results.rows);
});

export default router;

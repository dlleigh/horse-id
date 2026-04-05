import { Router } from "express";
import { eq } from "drizzle-orm";
import { db } from "../db/client.js";
import { horses, herds, photos } from "../db/schema.js";

const router = Router();

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

export default router;

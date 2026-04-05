import { Router } from "express";
import { eq } from "drizzle-orm";
import { db } from "../db/client.js";
import { photos } from "../db/schema.js";
import { getDriveClient } from "../services/drive.js";

const router = Router();

// PATCH /api/photos/:id — toggle exclude/include
router.patch("/:id", async (req, res) => {
  const photoId = Number(req.params.id);
  const { excluded } = req.body;

  if (typeof excluded !== "boolean") {
    res.status(400).json({ error: "excluded must be a boolean" });
    return;
  }

  const [updated] = await db
    .update(photos)
    .set({ excluded })
    .where(eq(photos.id, photoId))
    .returning({ id: photos.id, excluded: photos.excluded });

  if (!updated) {
    res.status(404).json({ error: "Photo not found" });
    return;
  }

  res.json(updated);
});

// GET /api/photos/:id/image — proxy image from Drive
router.get("/:id/image", async (req, res) => {
  const photoId = Number(req.params.id);

  const [photo] = await db
    .select({ driveFileId: photos.driveFileId, filename: photos.filename })
    .from(photos)
    .where(eq(photos.id, photoId));

  if (!photo) {
    res.status(404).json({ error: "Photo not found" });
    return;
  }

  try {
    const drive = getDriveClient();
    const response = await drive.files.get(
      { fileId: photo.driveFileId, alt: "media", supportsAllDrives: true },
      { responseType: "stream" }
    );

    res.setHeader("Content-Type", response.headers["content-type"] ?? "image/jpeg");
    res.setHeader("Cache-Control", "public, max-age=86400");
    (response.data as NodeJS.ReadableStream).pipe(res);
  } catch (err) {
    console.error(`Failed to proxy photo ${photoId}:`, err);
    res.status(502).json({ error: "Failed to fetch image from Drive" });
  }
});

export default router;

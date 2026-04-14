import { Router } from "express";
import { eq, sql } from "drizzle-orm";
import { db } from "../db/client.js";
import { photos } from "../db/schema.js";
import { getDriveClient } from "../services/drive.js";

const router = Router();

// GET /api/photos/errors — list photos with error status
router.get("/errors", async (_req, res) => {
  const rows = await db.execute(sql`
    SELECT p.id, p.filename, p.drive_file_id, p.processing_status, p.detection_result,
           h.name AS horse_name, hd.name AS herd_name
    FROM photos p
    JOIN horses h ON h.id = p.horse_id
    JOIN herds hd ON hd.id = h.herd_id
    WHERE p.processing_status = 'error'
    ORDER BY hd.name, h.name, p.filename
  `);
  res.json(rows.rows);
});

// POST /api/photos/:id/retry — reset error photo to pending
router.post("/:id/retry", async (req, res) => {
  const photoId = Number(req.params.id);

  const [updated] = await db
    .update(photos)
    .set({ processingStatus: "pending", detectionResult: null })
    .where(eq(photos.id, photoId))
    .returning({ id: photos.id, processingStatus: photos.processingStatus });

  if (!updated) {
    res.status(404).json({ error: "Photo not found" });
    return;
  }

  res.json(updated);
});

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
// ?size=thumb returns a Drive-generated thumbnail (much faster)
router.get("/:id/image", async (req, res) => {
  const photoId = Number(req.params.id);
  const wantThumb = req.query.size === "thumb";

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

    if (wantThumb) {
      // Fetch thumbnailLink from Drive metadata and redirect
      const meta = await drive.files.get({
        fileId: photo.driveFileId,
        fields: "thumbnailLink",
        supportsAllDrives: true,
      });

      const thumbLink = meta.data.thumbnailLink;
      if (thumbLink) {
        // Replace default size with 400px — plenty for grid cards
        const sized = thumbLink.replace(/=s\d+$/, "=s400");
        res.setHeader("Cache-Control", "public, max-age=86400");
        res.redirect(302, sized);
        return;
      }
      // Fall through to full image if no thumbnail available
    }

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

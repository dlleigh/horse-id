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

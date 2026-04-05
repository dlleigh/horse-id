import { Router } from "express";
import { sql } from "drizzle-orm";
import { db } from "../db/client.js";
import multer from "multer";
import { writeFileSync, unlinkSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const router = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

// POST /api/identify — upload image, get ranked matches
router.post("/", upload.single("image"), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "No image uploaded" });
    return;
  }

  const herdId = req.body.herd_id ? Number(req.body.herd_id) : null;
  const topK = req.body.top_k ? Number(req.body.top_k) : 5;

  // Save to temp file for the Python extractor
  const tmpPath = join(tmpdir(), `identify-${Date.now()}.jpg`);

  try {
    writeFileSync(tmpPath, req.file.buffer);

    // For now, shell out to a Python script for extraction + query.
    // In production this would invoke the ML worker Lambda.
    const { execSync } = await import("child_process");
    const scriptPath = join(import.meta.dirname, "../../../../workers/identify_cli.py");
    const args = [tmpPath, "--top-k", String(topK)];
    if (herdId) args.push("--herd-id", String(herdId));

    const result = execSync(
      `python ${scriptPath} ${args.join(" ")}`,
      { encoding: "utf-8", timeout: 60000, env: process.env }
    );

    const predictions = JSON.parse(result);
    res.json({ predictions });
  } catch (err) {
    console.error("Identification failed:", err);
    res.status(500).json({ error: "Identification failed" });
  } finally {
    try { unlinkSync(tmpPath); } catch {}
  }
});

export default router;

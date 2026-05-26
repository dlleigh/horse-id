import { Router } from "express";
import { resetStuckPhotos, resetAllFeatures, fanOutProcessing } from "../services/process.js";

const router = Router();

let running = false;

// POST /api/process — reset stuck photos and re-fan-out processing
router.post("/", async (_req, res) => {
  if (running) {
    res.json({ status: "already_running" });
    return;
  }

  running = true;
  (async () => {
    try {
      await resetStuckPhotos();
      await fanOutProcessing("detect");
      await fanOutProcessing("extract");
    } catch (err) {
      console.error("[process] Reprocess failed:", err);
    } finally {
      running = false;
    }
  })();

  res.json({ status: "started" });
});

// POST /api/process/re-extract — delete all embeddings and re-extract with current model
router.post("/re-extract", async (_req, res) => {
  if (running) {
    res.json({ status: "already_running" });
    return;
  }

  running = true;
  const { deleted, reset } = await resetAllFeatures();

  (async () => {
    try {
      await fanOutProcessing("extract");
    } catch (err) {
      console.error("[process] Re-extract failed:", err);
    } finally {
      running = false;
    }
  })();

  res.json({ status: "started", deleted, reset });
});

export default router;

import { Router } from "express";
import { resetStuckPhotos, fanOutProcessing } from "../services/process.js";

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

export default router;

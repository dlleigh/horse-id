import { Router } from "express";
import { eq } from "drizzle-orm";
import { db } from "../db/client.js";
import { syncRuns } from "../db/schema.js";
import { runSync } from "../services/sync.js";

const router = Router();

// POST /api/sync — trigger a sync
router.post("/", async (_req, res) => {
  try {
    const { syncRunId } = await runSync();
    res.json({ syncRunId });
  } catch (err) {
    console.error("Sync failed:", err);
    res.status(500).json({ error: "Sync failed" });
  }
});

// GET /api/sync/:id — get sync run status
router.get("/:id", async (req, res) => {
  const syncId = Number(req.params.id);

  const [run] = await db
    .select()
    .from(syncRuns)
    .where(eq(syncRuns.id, syncId));

  if (!run) {
    res.status(404).json({ error: "Sync run not found" });
    return;
  }

  res.json(run);
});

export default router;

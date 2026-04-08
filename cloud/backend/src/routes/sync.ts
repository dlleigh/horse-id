import { Router } from "express";
import { eq } from "drizzle-orm";
import { db } from "../db/client.js";
import { syncRuns } from "../db/schema.js";
import { runIncrementalSync } from "../services/sync.js";

const router = Router();

// POST /api/sync — create sync run row, kick off work in background
router.post("/", async (_req, res) => {
  try {
    const [syncRun] = await db
      .insert(syncRuns)
      .values({ status: "running" })
      .returning({ id: syncRuns.id });

    // Fire and forget — frontend polls GET /api/sync/:id
    runIncrementalSync(syncRun.id).catch(err => {
      console.error("Sync failed:", err);
    });

    res.json({ syncRunId: syncRun.id });
  } catch (err) {
    console.error("Sync failed to start:", err);
    res.status(500).json({ error: "Sync failed to start" });
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

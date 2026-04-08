import { Router } from "express";
import { sql } from "drizzle-orm";
import { db } from "../db/client.js";

const router = Router();

// GET /api/stats — photo processing counts + sync status
router.get("/", async (_req, res) => {
  const [photoCounts, latestSync] = await Promise.all([
    db.execute(sql`
      SELECT
        count(*)::int AS total,
        count(*) FILTER (WHERE processing_status = 'pending')::int AS pending,
        count(*) FILTER (WHERE processing_status = 'detecting')::int AS detecting,
        count(*) FILTER (WHERE processing_status = 'detected')::int AS detected,
        count(*) FILTER (WHERE processing_status = 'extracting')::int AS extracting,
        count(*) FILTER (WHERE processing_status = 'ready')::int AS ready,
        count(*) FILTER (WHERE processing_status = 'error')::int AS error
      FROM photos
    `),
    db.execute(sql`
      SELECT id, status, herds_total, herds_scanned, last_heartbeat,
             files_scanned, files_added, files_removed, files_moved, completed_at
      FROM sync_runs ORDER BY id DESC LIMIT 1
    `),
  ]);

  const counts = photoCounts.rows[0] as Record<string, number>;
  const sync = latestSync.rows[0] as {
    id: number;
    status: string;
    herds_total: number;
    herds_scanned: number;
    last_heartbeat: string;
    files_scanned: number;
    files_added: number;
    files_removed: number;
    files_moved: number;
    completed_at: string | null;
  } | undefined;

  // Consider a sync alive only if heartbeat is within the last 5 minutes
  const isRunning =
    sync?.status === "running" &&
    sync.last_heartbeat &&
    Date.now() - new Date(sync.last_heartbeat).getTime() < 300_000;

  const lastSync = !isRunning && sync?.status === "completed" ? {
    filesScanned: sync.files_scanned,
    filesAdded: sync.files_added,
    filesRemoved: sync.files_removed,
    filesMoved: sync.files_moved,
  } : null;

  res.json({
    ...counts,
    syncStatus: isRunning ? "running" : null,
    syncProgressTotal: isRunning ? sync.herds_total : null,
    syncProgressDone: isRunning ? sync.herds_scanned : null,
    lastSync,
  });
});

export default router;

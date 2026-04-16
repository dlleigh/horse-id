import "../env.js";
import { runIncrementalSync } from "../services/sync.js";
import { db } from "../db/client.js";
import { syncRuns } from "../db/schema.js";

async function main() {
  console.log("Starting sync...");
  const [syncRun] = await db.insert(syncRuns).values({ status: "running" }).returning({ id: syncRuns.id });
  await runIncrementalSync(syncRun.id);
  const syncRunId = syncRun.id;
  console.log(`Sync run ${syncRunId} finished`);
}

main().catch((err) => {
  console.error("Sync failed:", err);
  process.exit(1);
});

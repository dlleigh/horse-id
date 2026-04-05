import "../env.js";
import { runSync } from "../services/sync.js";

async function main() {
  console.log("Starting sync...");
  const { syncRunId } = await runSync();
  console.log(`Sync run ${syncRunId} finished`);
}

main().catch((err) => {
  console.error("Sync failed:", err);
  process.exit(1);
});

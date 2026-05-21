import { eq, sql } from "drizzle-orm";
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { db } from "../db/client.js";
import { syncRuns } from "../db/schema.js";
import { getConfig } from "./config.js";
import {
  getStartPageToken,
  getChanges,
  listFolders,
  listImageFiles,
  type DriveChange,
} from "./drive.js";

const region = process.env.AWS_REGION || "us-east-2";
const lambda = new LambdaClient({ region });
const FUNCTION_NAME = process.env.ML_WORKER_LAMBDA || "horse-id-ml-worker";

const SYNC_BATCH_SIZE = 50;

// ── Changes API sync (incremental) ──────────────────────────────────

/**
 * Incremental sync using Drive Changes API.
 * Fetches only what changed since last sync, dispatches to Lambda in batches.
 */
export async function runIncrementalSync(syncRunId: number): Promise<void> {
  try {
    // Get stored token
    const { rows } = await db.execute(
      sql`SELECT changes_token FROM drive_sync_state WHERE id = 1`
    );
    const storedToken = (rows[0] as { changes_token: string } | undefined)
      ?.changes_token;

    if (!storedToken) {
      console.log("[sync] No changes token found, running full scan...");
      await runFullScanAsChanges(syncRunId);
      return;
    }

    console.log("[sync] Fetching changes since last sync...");
    const { changes, newToken } = await getChanges(storedToken);

    if (changes.length === 0) {
      console.log("[sync] No changes detected.");
      await db
        .update(syncRuns)
        .set({ status: "completed", completedAt: new Date() })
        .where(eq(syncRuns.id, syncRunId));
      await db.execute(
        sql`INSERT INTO drive_sync_state (id, changes_token, updated_at)
            VALUES (1, ${newToken}, now())
            ON CONFLICT (id) DO UPDATE SET changes_token = ${newToken}, updated_at = now()`
      );
      return;
    }

    console.log(`[sync] Found ${changes.length} changes`);

    // Separate folder changes from file changes
    const folderChanges: DriveChange[] = [];
    const fileChanges: DriveChange[] = [];
    for (const change of changes) {
      if (change.mimeType === "application/vnd.google-apps.folder") {
        folderChanges.push(change);
      } else if (change.mimeType?.startsWith("image/")) {
        fileChanges.push(change);
      }
      // Removals without mimeType — could be either
      if (change.removed && !change.mimeType) {
        folderChanges.push(change);
        fileChanges.push(change);
      }
    }

    await db
      .update(syncRuns)
      .set({ herdsTotal: fileChanges.length, lastHeartbeat: new Date() })
      .where(eq(syncRuns.id, syncRunId));

    // Dispatch to Lambda in batches
    const batches: { changes: object[]; folder_changes: object[] }[] = [];
    for (let i = 0; i < fileChanges.length; i += SYNC_BATCH_SIZE) {
      batches.push({
        changes: fileChanges.slice(i, i + SYNC_BATCH_SIZE).map(toPayload),
        folder_changes: i === 0 ? folderChanges.map(toPayload) : [],
      });
    }

    if (batches.length === 0 && folderChanges.length > 0) {
      batches.push({
        changes: [],
        folder_changes: folderChanges.map(toPayload),
      });
    }

    let dispatched = 0;
    for (const batch of batches) {
      await lambda.send(
        new InvokeCommand({
          FunctionName: FUNCTION_NAME,
          InvocationType: "Event",
          Payload: Buffer.from(
            JSON.stringify({
              task: "sync_batch",
              sync_run_id: syncRunId,
              ...batch,
            })
          ),
        })
      );
      dispatched++;
    }

    console.log(
      `[sync] Dispatched ${dispatched} sync batches (${fileChanges.length} files, ${folderChanges.length} folders)`
    );

    await db.execute(
      sql`INSERT INTO drive_sync_state (id, changes_token, updated_at)
          VALUES (1, ${newToken}, now())
          ON CONFLICT (id) DO UPDATE SET changes_token = ${newToken}, updated_at = now()`
    );

    await db
      .update(syncRuns)
      .set({
        status: "completed",
        completedAt: new Date(),
        filesScanned: fileChanges.length,
      })
      .where(eq(syncRuns.id, syncRunId));
  } catch (err) {
    await db
      .update(syncRuns)
      .set({ status: "failed", completedAt: new Date() })
      .where(eq(syncRuns.id, syncRunId));
    throw err;
  }
}

function toPayload(change: DriveChange): object {
  return {
    file_id: change.fileId,
    folder_id: change.fileId,
    name: change.name,
    parent_id: change.parentId,
    md5: change.md5Checksum,
    mime_type: change.mimeType,
    removed: change.removed,
  };
}

// ── Full scan as Lambda batches (first time or fallback) ────────────

/**
 * Full scan — lists all folders and files from Drive, then dispatches
 * as sync_batch tasks to Lambda (same path as incremental sync).
 * Used when no Changes API token exists (first sync).
 */
export async function runFullScanAsChanges(syncRunId: number): Promise<void> {
  const config = getConfig();
  const rootFolderId = config.googleDriveDirectoryId;

  try {
    const startToken = await getStartPageToken();

    console.log("[sync] Full scan: listing Drive folders...");
    const herdFolders = await listFolders(rootFolderId);

    await db
      .update(syncRuns)
      .set({ herdsTotal: herdFolders.length, lastHeartbeat: new Date() })
      .where(eq(syncRuns.id, syncRunId));

    // Build folder changes: herds first, then horses (order matters for _upsert_folder)
    const folderChanges: object[] = [];
    const allFileChanges: object[] = [];
    let totalFiles = 0;
    let herdsScanned = 0;

    // Process herds in parallel (bounded concurrency to avoid Drive rate limits)
    const HERD_CONCURRENCY = 5;
    for (let h = 0; h < herdFolders.length; h += HERD_CONCURRENCY) {
      const herdBatch = herdFolders.slice(h, h + HERD_CONCURRENCY);

      await Promise.all(
        herdBatch.map(async (herd) => {
          folderChanges.push({
            folder_id: herd.id,
            name: herd.name,
            parent_id: rootFolderId,
            removed: false,
          });

          const horseFolders = await listFolders(herd.id);

          // List all horse image files in parallel within this herd
          await Promise.all(
            horseFolders.map(async (horse) => {
              folderChanges.push({
                folder_id: horse.id,
                name: horse.name,
                parent_id: herd.id,
                removed: false,
              });

              const imageFiles = await listImageFiles(horse.id);
              for (const file of imageFiles) {
                allFileChanges.push({
                  file_id: file.id,
                  name: file.name,
                  parent_id: horse.id,
                  md5: file.md5Checksum,
                  mime_type: "image/jpeg",
                  removed: false,
                });
              }
              totalFiles += imageFiles.length;
            })
          );

          herdsScanned += 1;
          await db
            .update(syncRuns)
            .set({
              herdsScanned,
              filesScanned: totalFiles,
              lastHeartbeat: new Date(),
            })
            .where(eq(syncRuns.id, syncRunId));
        })
      );
    }

    console.log(
      `[sync] Full scan: ${folderChanges.length} folders, ${totalFiles} files`
    );

    // --- Reconcile: find DB records that no longer exist in Drive ---
    const driveFolderIds = new Set(
      folderChanges.map((fc: any) => fc.folder_id as string)
    );
    const driveFileIds = new Set(
      allFileChanges.map((fc: any) => fc.file_id as string)
    );

    const { rows: dbHorses } = await db.execute(
      sql`SELECT drive_folder_id FROM horses`
    );
    for (const row of dbHorses as { drive_folder_id: string }[]) {
      if (!driveFolderIds.has(row.drive_folder_id)) {
        folderChanges.push({
          folder_id: row.drive_folder_id,
          name: null,
          parent_id: null,
          removed: true,
        });
      }
    }

    const { rows: dbHerds } = await db.execute(
      sql`SELECT drive_folder_id FROM herds`
    );
    for (const row of dbHerds as { drive_folder_id: string }[]) {
      if (!driveFolderIds.has(row.drive_folder_id)) {
        folderChanges.push({
          folder_id: row.drive_folder_id,
          name: null,
          parent_id: null,
          removed: true,
        });
      }
    }

    const { rows: dbPhotos } = await db.execute(
      sql`SELECT drive_file_id FROM photos`
    );
    for (const row of dbPhotos as { drive_file_id: string }[]) {
      if (!driveFileIds.has(row.drive_file_id)) {
        allFileChanges.push({
          file_id: row.drive_file_id,
          name: null,
          parent_id: null,
          md5: null,
          mime_type: null,
          removed: true,
        });
      }
    }

    const removedFolders = folderChanges.filter(
      (f: any) => f.removed
    ).length;
    const removedFiles = allFileChanges.filter((f: any) => f.removed).length;
    if (removedFolders > 0 || removedFiles > 0) {
      console.log(
        `[sync] Reconciliation: ${removedFolders} folders, ${removedFiles} files to remove`
      );
    }

    // Reset filesScanned before dispatching — Lambda sync_batch will increment from 0
    await db
      .update(syncRuns)
      .set({ filesScanned: 0, lastHeartbeat: new Date() })
      .where(eq(syncRuns.id, syncRunId));

    // First batch: all folder changes (no files) so DB hierarchy is created
    const batches: { changes: object[]; folder_changes: object[] }[] = [
      { changes: [], folder_changes: folderChanges },
    ];

    // Remaining batches: file changes
    for (let i = 0; i < allFileChanges.length; i += SYNC_BATCH_SIZE) {
      batches.push({
        changes: allFileChanges.slice(i, i + SYNC_BATCH_SIZE),
        folder_changes: [],
      });
    }

    let dispatched = 0;
    for (const batch of batches) {
      await lambda.send(
        new InvokeCommand({
          FunctionName: FUNCTION_NAME,
          InvocationType: "Event",
          Payload: Buffer.from(
            JSON.stringify({
              task: "sync_batch",
              sync_run_id: syncRunId,
              ...batch,
            })
          ),
        })
      );
      dispatched++;
    }

    console.log(`[sync] Full scan: dispatched ${dispatched} sync batches`);

    // Store Changes API token for future incremental syncs
    await db.execute(
      sql`INSERT INTO drive_sync_state (id, changes_token, updated_at)
          VALUES (1, ${startToken}, now())
          ON CONFLICT (id) DO UPDATE SET changes_token = ${startToken}, updated_at = now()`
    );

    await db
      .update(syncRuns)
      .set({
        status: "completed",
        completedAt: new Date(),
        filesScanned: totalFiles,
      })
      .where(eq(syncRuns.id, syncRunId));
  } catch (err) {
    await db
      .update(syncRuns)
      .set({ status: "failed", completedAt: new Date() })
      .where(eq(syncRuns.id, syncRunId));
    throw err;
  }
}

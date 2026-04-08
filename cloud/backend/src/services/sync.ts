import { eq, notInArray, inArray, sql } from "drizzle-orm";
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { db } from "../db/client.js";
import { herds, horses, photos, syncRuns } from "../db/schema.js";
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
      await runFullSync(syncRunId);
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

// ── Full scan sync (first time or fallback) ─────────────────────────

/**
 * Full scan — lists all folders and files from Drive.
 * Used when no Changes API token exists (first sync).
 */
export async function runFullSync(syncRunId: number): Promise<void> {
  const config = getConfig();
  const rootFolderId = config.googleDriveDirectoryId;

  let filesScanned = 0;
  let filesAdded = 0;
  let filesRemoved = 0;
  let filesMoved = 0;

  try {
    const startToken = await getStartPageToken();

    console.log("Listing herd folders...");
    const herdFolders = await listFolders(rootFolderId);
    console.log(`  Found ${herdFolders.length} herds`);

    const activeDriveHerdIds: string[] = [];
    const activeDriveHorseIds: string[] = [];
    const activeDriveFileIds: string[] = [];

    const herdData: {
      folder: (typeof herdFolders)[0];
      horseFolders: typeof herdFolders;
    }[] = [];
    let totalHorses = 0;
    for (const herdFolder of herdFolders) {
      const horseFolders = await listFolders(herdFolder.id);
      herdData.push({ folder: herdFolder, horseFolders });
      totalHorses += horseFolders.length;
    }
    console.log(
      `  Found ${totalHorses} horses across ${herdFolders.length} herds`
    );

    let horsesScanned = 0;
    await db
      .update(syncRuns)
      .set({ herdsTotal: totalHorses, lastHeartbeat: new Date() })
      .where(eq(syncRuns.id, syncRunId));

    for (const { folder: herdFolder, horseFolders } of herdData) {
      activeDriveHerdIds.push(herdFolder.id);

      const [existingHerd] = await db
        .select()
        .from(herds)
        .where(eq(herds.driveFolderId, herdFolder.id));

      let herdId: number;
      if (existingHerd) {
        if (existingHerd.name !== herdFolder.name) {
          await db
            .update(herds)
            .set({ name: herdFolder.name })
            .where(eq(herds.id, existingHerd.id));
        }
        herdId = existingHerd.id;
      } else {
        const [newHerd] = await db
          .insert(herds)
          .values({ name: herdFolder.name, driveFolderId: herdFolder.id })
          .returning({ id: herds.id });
        herdId = newHerd.id;
      }

      for (const horseFolder of horseFolders) {
        const horseStart = Date.now();
        activeDriveHorseIds.push(horseFolder.id);

        const [existingHorse] = await db
          .select()
          .from(horses)
          .where(eq(horses.driveFolderId, horseFolder.id));

        let horseId: number;
        if (existingHorse) {
          const updates: Partial<typeof horses.$inferInsert> = {};
          if (existingHorse.name !== horseFolder.name)
            updates.name = horseFolder.name;
          if (existingHorse.herdId !== herdId) {
            updates.herdId = herdId;
            filesMoved++;
          }
          if (Object.keys(updates).length > 0) {
            await db
              .update(horses)
              .set(updates)
              .where(eq(horses.id, existingHorse.id));
          }
          horseId = existingHorse.id;
        } else {
          const [newHorse] = await db
            .insert(horses)
            .values({
              name: horseFolder.name,
              herdId,
              driveFolderId: horseFolder.id,
            })
            .returning({ id: horses.id });
          horseId = newHorse.id;
        }

        const driveStart = Date.now();
        const imageFiles = await listImageFiles(horseFolder.id);
        const driveMs = Date.now() - driveStart;
        filesScanned += imageFiles.length;

        const dbStart = Date.now();
        const driveFileIds = imageFiles.map((f) => f.id);
        const existingPhotos =
          driveFileIds.length > 0
            ? await db
                .select()
                .from(photos)
                .where(inArray(photos.driveFileId, driveFileIds))
            : [];
        const existingByDriveId = new Map(
          existingPhotos.map((p) => [p.driveFileId, p])
        );

        const toInsert: (typeof photos.$inferInsert)[] = [];
        const toUpdate: {
          id: number;
          updates: Partial<typeof photos.$inferInsert>;
        }[] = [];

        for (const file of imageFiles) {
          activeDriveFileIds.push(file.id);
          const existing = existingByDriveId.get(file.id);

          if (existing) {
            const updates: Partial<typeof photos.$inferInsert> = {};
            if (existing.driveMd5 !== file.md5Checksum)
              updates.driveMd5 = file.md5Checksum;
            if (existing.filename !== file.name) updates.filename = file.name;
            if (existing.horseId !== horseId) {
              updates.horseId = horseId;
              updates.processingStatus = "pending";
            }
            if (Object.keys(updates).length > 0) {
              toUpdate.push({ id: existing.id, updates });
            }
          } else {
            toInsert.push({
              horseId,
              filename: file.name,
              driveFileId: file.id,
              driveMd5: file.md5Checksum,
              processingStatus: "pending",
            });
          }
        }

        if (toInsert.length > 0) {
          await db.insert(photos).values(toInsert);
          filesAdded += toInsert.length;
        }

        if (toUpdate.length > 0) {
          await Promise.all(
            toUpdate.map(({ id, updates }) =>
              db.update(photos).set(updates).where(eq(photos.id, id))
            )
          );
        }
        const dbMs = Date.now() - dbStart;

        horsesScanned++;
        await db
          .update(syncRuns)
          .set({
            herdsScanned: horsesScanned,
            filesScanned,
            filesAdded,
            lastHeartbeat: new Date(),
          })
          .where(eq(syncRuns.id, syncRunId));

        const totalMs = Date.now() - horseStart;
        console.log(
          `  ${horseFolder.name}: ${imageFiles.length} files, drive=${driveMs}ms db=${dbMs}ms total=${totalMs}ms`
        );
      }
    }

    if (activeDriveFileIds.length > 0) {
      const deletedPhotos = await db
        .delete(photos)
        .where(notInArray(photos.driveFileId, activeDriveFileIds))
        .returning({ id: photos.id });
      filesRemoved = deletedPhotos.length;
    }

    if (activeDriveHorseIds.length > 0) {
      await db
        .delete(horses)
        .where(notInArray(horses.driveFolderId, activeDriveHorseIds));
    }

    if (activeDriveHerdIds.length > 0) {
      await db
        .delete(herds)
        .where(notInArray(herds.driveFolderId, activeDriveHerdIds));
    }

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
        filesScanned,
        filesAdded,
        filesRemoved,
        filesMoved,
      })
      .where(eq(syncRuns.id, syncRunId));

    console.log(
      `\nFull sync complete: scanned=${filesScanned}, added=${filesAdded}, removed=${filesRemoved}, moved=${filesMoved}`
    );
  } catch (err) {
    await db
      .update(syncRuns)
      .set({ status: "failed", completedAt: new Date() })
      .where(eq(syncRuns.id, syncRunId));
    throw err;
  }
}

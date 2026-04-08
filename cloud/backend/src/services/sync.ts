import { eq, notInArray, inArray } from "drizzle-orm";
import { db } from "../db/client.js";
import { herds, horses, photos, syncRuns } from "../db/schema.js";
import { getConfig } from "./config.js";
import { listFolders, listImageFiles } from "./drive.js";

export async function runSync(syncRunId: number): Promise<void> {
  const config = getConfig();
  const rootFolderId = config.googleDriveDirectoryId;

  let filesScanned = 0;
  let filesAdded = 0;
  let filesRemoved = 0;
  let filesMoved = 0;

  try {
    // 1. List herd folders from Drive
    console.log("Listing herd folders...");
    const herdFolders = await listFolders(rootFolderId);
    console.log(`  Found ${herdFolders.length} herds`);

    const activeDriveHerdIds: string[] = [];
    const activeDriveHorseIds: string[] = [];
    const activeDriveFileIds: string[] = [];

    // First pass: list all herd and horse folders to get total count
    const herdData: { folder: typeof herdFolders[0]; horseFolders: typeof herdFolders }[] = [];
    let totalHorses = 0;
    for (const herdFolder of herdFolders) {
      const horseFolders = await listFolders(herdFolder.id);
      herdData.push({ folder: herdFolder, horseFolders });
      totalHorses += horseFolders.length;
    }
    console.log(`  Found ${totalHorses} horses across ${herdFolders.length} herds`);

    let horsesScanned = 0;
    await db
      .update(syncRuns)
      .set({ herdsTotal: totalHorses, lastHeartbeat: new Date() })
      .where(eq(syncRuns.id, syncRunId));

    for (const { folder: herdFolder, horseFolders } of herdData) {
      activeDriveHerdIds.push(herdFolder.id);

      // Upsert herd
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
          console.log(`  Renamed herd: ${existingHerd.name} -> ${herdFolder.name}`);
        }
        herdId = existingHerd.id;
      } else {
        const [newHerd] = await db
          .insert(herds)
          .values({ name: herdFolder.name, driveFolderId: herdFolder.id })
          .returning({ id: herds.id });
        herdId = newHerd.id;
        console.log(`  New herd: ${herdFolder.name}`);
      }

      for (const horseFolder of horseFolders) {
        const horseStart = Date.now();
        activeDriveHorseIds.push(horseFolder.id);

        // Upsert horse
        const [existingHorse] = await db
          .select()
          .from(horses)
          .where(eq(horses.driveFolderId, horseFolder.id));

        let horseId: number;
        if (existingHorse) {
          const updates: Partial<typeof horses.$inferInsert> = {};
          if (existingHorse.name !== horseFolder.name) updates.name = horseFolder.name;
          if (existingHorse.herdId !== herdId) {
            updates.herdId = herdId;
            filesMoved++;
          }
          if (Object.keys(updates).length > 0) {
            await db.update(horses).set(updates).where(eq(horses.id, existingHorse.id));
            if (updates.name) console.log(`  Renamed horse: ${existingHorse.name} -> ${horseFolder.name}`);
            if (updates.herdId) console.log(`  Moved horse: ${horseFolder.name} to ${herdFolder.name}`);
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

        // 3. List image files within this horse folder
        const driveStart = Date.now();
        const imageFiles = await listImageFiles(horseFolder.id);
        const driveMs = Date.now() - driveStart;
        filesScanned += imageFiles.length;

        // Batch: fetch all existing photos for this horse's drive file IDs in one query
        const dbStart = Date.now();
        const driveFileIds = imageFiles.map(f => f.id);
        const existingPhotos = driveFileIds.length > 0
          ? await db
              .select()
              .from(photos)
              .where(inArray(photos.driveFileId, driveFileIds))
          : [];
        const existingByDriveId = new Map(existingPhotos.map(p => [p.driveFileId, p]));

        // Collect inserts and updates
        const toInsert: (typeof photos.$inferInsert)[] = [];
        const toUpdate: { id: number; updates: Partial<typeof photos.$inferInsert> }[] = [];

        for (const file of imageFiles) {
          activeDriveFileIds.push(file.id);
          const existing = existingByDriveId.get(file.id);

          if (existing) {
            const updates: Partial<typeof photos.$inferInsert> = {};
            if (existing.driveMd5 !== file.md5Checksum) updates.driveMd5 = file.md5Checksum;
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

        // Batch insert new photos
        if (toInsert.length > 0) {
          await db.insert(photos).values(toInsert);
          filesAdded += toInsert.length;
        }

        // Batch updates (still individual but could be parallelized)
        if (toUpdate.length > 0) {
          await Promise.all(
            toUpdate.map(({ id, updates }) =>
              db.update(photos).set(updates).where(eq(photos.id, id))
            )
          );
        }
        const dbMs = Date.now() - dbStart;

        // Update progress after each horse
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
        console.log(`  ${horseFolder.name}: ${imageFiles.length} files, drive=${driveMs}ms db=${dbMs}ms total=${totalMs}ms`);
      }
    }

    // 4. Detect deletions: remove DB records for things no longer in Drive
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

    // Update sync run
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

    console.log(`\nSync complete: scanned=${filesScanned}, added=${filesAdded}, removed=${filesRemoved}, moved=${filesMoved}`);
  } catch (err) {
    await db
      .update(syncRuns)
      .set({ status: "failed", completedAt: new Date() })
      .where(eq(syncRuns.id, syncRunId));
    throw err;
  }
}

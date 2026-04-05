import { eq, and, notInArray } from "drizzle-orm";
import { db } from "../db/client.js";
import { herds, horses, photos, syncRuns } from "../db/schema.js";
import { getConfig } from "./config.js";
import { listFolders, listImageFiles } from "./drive.js";

export async function runSync(): Promise<{ syncRunId: number }> {
  const config = getConfig();
  const rootFolderId = config.googleDriveDirectoryId;

  // Create sync run
  const [syncRun] = await db
    .insert(syncRuns)
    .values({ status: "running" })
    .returning({ id: syncRuns.id });
  const syncRunId = syncRun.id;

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

    for (const herdFolder of herdFolders) {
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

      // 2. List horse folders within this herd
      const horseFolders = await listFolders(herdFolder.id);

      for (const horseFolder of horseFolders) {
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
        const imageFiles = await listImageFiles(horseFolder.id);
        filesScanned += imageFiles.length;

        for (const file of imageFiles) {
          activeDriveFileIds.push(file.id);

          const [existingPhoto] = await db
            .select()
            .from(photos)
            .where(eq(photos.driveFileId, file.id));

          if (existingPhoto) {
            // Update if md5 changed or horse moved
            const updates: Partial<typeof photos.$inferInsert> = {};
            if (existingPhoto.driveMd5 !== file.md5Checksum) updates.driveMd5 = file.md5Checksum;
            if (existingPhoto.filename !== file.name) updates.filename = file.name;
            if (existingPhoto.horseId !== horseId) {
              updates.horseId = horseId;
              updates.processingStatus = "pending"; // re-process if moved
            }
            if (Object.keys(updates).length > 0) {
              await db.update(photos).set(updates).where(eq(photos.id, existingPhoto.id));
            }
          } else {
            await db.insert(photos).values({
              horseId,
              filename: file.name,
              driveFileId: file.id,
              driveMd5: file.md5Checksum,
              processingStatus: "pending",
            });
            filesAdded++;
          }
        }
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
    return { syncRunId };
  } catch (err) {
    await db
      .update(syncRuns)
      .set({ status: "failed", completedAt: new Date() })
      .where(eq(syncRuns.id, syncRunId));
    throw err;
  }
}

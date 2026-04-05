import {
  pgTable,
  serial,
  text,
  integer,
  boolean,
  timestamp,
  unique,
  index,
  customType,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";

const vector = customType<{ data: number[]; driverParam: string }>({
  dataType() {
    return "vector(1536)";
  },
  toDriver(value: number[]): string {
    return `[${value.join(",")}]`;
  },
  fromDriver(value: unknown): number[] {
    const str = value as string;
    return str
      .slice(1, -1)
      .split(",")
      .map(Number);
  },
});

export const herds = pgTable("herds", {
  id: serial("id").primaryKey(),
  name: text("name").notNull().unique(),
  driveFolderId: text("drive_folder_id").notNull().unique(),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
});

export const horses = pgTable(
  "horses",
  {
    id: serial("id").primaryKey(),
    name: text("name").notNull(),
    herdId: integer("herd_id")
      .notNull()
      .references(() => herds.id, { onDelete: "cascade" }),
    driveFolderId: text("drive_folder_id").notNull().unique(),
    status: text("status").default("active"),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
  },
  (table) => [unique().on(table.herdId, table.name)]
);

export const photos = pgTable("photos", {
  id: serial("id").primaryKey(),
  horseId: integer("horse_id")
    .notNull()
    .references(() => horses.id, { onDelete: "cascade" }),
  filename: text("filename").notNull(),
  driveFileId: text("drive_file_id").notNull().unique(),
  driveMd5: text("drive_md5"),
  processingStatus: text("processing_status").default("pending"),
  detectionResult: text("detection_result"),
  excluded: boolean("excluded").default(false),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
});

export const features = pgTable(
  "features",
  {
    id: serial("id").primaryKey(),
    photoId: integer("photo_id")
      .notNull()
      .unique()
      .references(() => photos.id, { onDelete: "cascade" }),
    horseId: integer("horse_id")
      .notNull()
      .references(() => horses.id, { onDelete: "cascade" }),
    embedding: vector("embedding").notNull(),
    extractedAt: timestamp("extracted_at", { withTimezone: true }).defaultNow(),
  },
  (table) => [index("features_embedding_idx").using("ivfflat", table.embedding)]
);

export const syncRuns = pgTable("sync_runs", {
  id: serial("id").primaryKey(),
  startedAt: timestamp("started_at", { withTimezone: true }).defaultNow(),
  completedAt: timestamp("completed_at", { withTimezone: true }),
  status: text("status").default("running"),
  filesScanned: integer("files_scanned").default(0),
  filesAdded: integer("files_added").default(0),
  filesRemoved: integer("files_removed").default(0),
  filesMoved: integer("files_moved").default(0),
});

// Relations
export const herdsRelations = relations(herds, ({ many }) => ({
  horses: many(horses),
}));

export const horsesRelations = relations(horses, ({ one, many }) => ({
  herd: one(herds, { fields: [horses.herdId], references: [herds.id] }),
  photos: many(photos),
  features: many(features),
}));

export const photosRelations = relations(photos, ({ one }) => ({
  horse: one(horses, { fields: [photos.horseId], references: [horses.id] }),
  feature: one(features, { fields: [photos.id], references: [features.photoId] }),
}));

export const featuresRelations = relations(features, ({ one }) => ({
  photo: one(photos, { fields: [features.photoId], references: [photos.id] }),
  horse: one(horses, { fields: [features.horseId], references: [horses.id] }),
}));

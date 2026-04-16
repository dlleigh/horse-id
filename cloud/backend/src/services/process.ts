import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { sql } from "drizzle-orm";
import { db } from "../db/client.js";

const region = process.env.AWS_REGION || "us-east-2";
const lambda = new LambdaClient({ region });
const FUNCTION_NAME = process.env.ML_WORKER_LAMBDA || "horse-id-ml-worker";

const BATCH_SIZE = 20;
const MAX_CONCURRENT = 10;

interface PhotoRow {
  id: number;
  horse_id: number;
  drive_file_id: string;
  filename: string;
}

async function invokeLambda(task: string, photos: PhotoRow[]): Promise<number> {
  const payload = {
    task,
    photos: photos.map((p) => ({
      id: p.id,
      horse_id: p.horse_id,
      drive_file_id: p.drive_file_id,
      filename: p.filename,
    })),
  };

  const response = await lambda.send(
    new InvokeCommand({
      FunctionName: FUNCTION_NAME,
      InvocationType: "Event", // async — returns immediately
      Payload: Buffer.from(JSON.stringify(payload)),
    })
  );

  return response.StatusCode ?? 0;
}

export async function fanOutProcessing(
  task: "detect" | "extract"
): Promise<{ batches: number; photos: number }> {
  // Query eligible photos
  let rows: PhotoRow[];
  if (task === "detect") {
    const result = await db.execute(sql`
      SELECT p.id, p.horse_id, p.drive_file_id, p.filename
      FROM photos p
      WHERE p.processing_status = 'pending' AND p.excluded = false
      ORDER BY p.id
      LIMIT 10000
    `);
    rows = result.rows as unknown as PhotoRow[];
  } else {
    const result = await db.execute(sql`
      SELECT p.id, p.horse_id, p.drive_file_id, p.filename
      FROM photos p
      WHERE p.processing_status = 'detected'
        AND p.detection_result = 'SINGLE'
        AND NOT EXISTS (SELECT 1 FROM features f WHERE f.photo_id = p.id)
      ORDER BY p.id
      LIMIT 10000
    `);
    rows = result.rows as unknown as PhotoRow[];
  }

  if (rows.length === 0) {
    console.log(`[process] No photos pending ${task}.`);
    return { batches: 0, photos: 0 };
  }

  // Split into batches
  const batches: PhotoRow[][] = [];
  for (let i = 0; i < rows.length; i += BATCH_SIZE) {
    batches.push(rows.slice(i, i + BATCH_SIZE));
  }

  console.log(
    `[process] Fanning out ${rows.length} photos for ${task} across ${batches.length} batches`
  );

  // Invoke in chunks of MAX_CONCURRENT
  let invoked = 0;
  let errors = 0;
  for (let i = 0; i < batches.length; i += MAX_CONCURRENT) {
    const chunk = batches.slice(i, i + MAX_CONCURRENT);
    const results = await Promise.allSettled(
      chunk.map((batch) => invokeLambda(task, batch))
    );
    for (const r of results) {
      if (r.status === "fulfilled" && r.value === 202) {
        invoked++;
      } else {
        errors++;
        if (r.status === "rejected") {
          console.error(`[process] Lambda invoke error:`, r.reason);
        }
      }
    }
  }

  console.log(
    `[process] ${task}: ${invoked} batches invoked, ${errors} errors`
  );
  return { batches: invoked, photos: rows.length };
}

/**
 * Reset photos stuck in transient states for > 15 minutes.
 * Call before fanOutProcessing to recover from Lambda failures.
 */
export async function resetStuckPhotos(): Promise<{ detecting: number; extracting: number }> {
  const detectResult = await db.execute(sql`
    UPDATE photos SET processing_status = 'pending'
    WHERE processing_status = 'detecting'
      AND updated_at < now() - interval '15 minutes'
  `);

  const extractResult = await db.execute(sql`
    UPDATE photos SET processing_status = 'detected'
    WHERE processing_status = 'extracting'
      AND updated_at < now() - interval '15 minutes'
  `);

  const detecting = Number(detectResult.rowCount ?? 0);
  const extracting = Number(extractResult.rowCount ?? 0);

  if (detecting > 0 || extracting > 0) {
    console.log(`[process] Reset stuck photos: ${detecting} detecting, ${extracting} extracting`);
  }

  return { detecting, extracting };
}

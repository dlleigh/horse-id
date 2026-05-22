import { Router } from "express";
import { randomUUID } from "crypto";
import { S3Client, PutObjectCommand, DeleteObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";

const router = Router();

const region = process.env.AWS_REGION || "us-east-2";
const s3 = new S3Client({ region });
const lambda = new LambdaClient({ region });
const BUCKET = process.env.IDENTIFY_BUCKET || "horse-id-temp-uploads";
const FUNCTION_NAME = process.env.ML_WORKER_LAMBDA || "horse-id-ml-worker";

// GET /api/identify/upload-url — presigned S3 URL for direct browser upload
router.get("/upload-url", async (_req, res) => {
  try {
    const s3Key = `tmp/identify-${randomUUID()}`;
    const command = new PutObjectCommand({ Bucket: BUCKET, Key: s3Key });
    const uploadUrl = await getSignedUrl(s3, command, { expiresIn: 300 });
    res.json({ uploadUrl, s3Key });
  } catch (err) {
    console.error("Failed to generate upload URL:", err);
    res.status(500).json({ error: "Failed to generate upload URL" });
  }
});

// POST /api/identify — run identification against an image uploaded to S3
router.post("/", async (req, res) => {
  const s3Key = req.body.s3_key;
  if (!s3Key || typeof s3Key !== "string") {
    res.status(400).json({ error: "s3_key is required" });
    return;
  }

  const herdId = req.body.herd_id ? Number(req.body.herd_id) : null;
  const topK = req.body.top_k ? Number(req.body.top_k) : 5;

  try {
    const payload = {
      task: "identify",
      s3_bucket: BUCKET,
      s3_key: s3Key,
      herd_id: herdId,
      top_k: topK,
    };

    const response = await lambda.send(new InvokeCommand({
      FunctionName: FUNCTION_NAME,
      Payload: Buffer.from(JSON.stringify(payload)),
    }));

    const result = JSON.parse(Buffer.from(response.Payload!).toString());

    if (result.status === "error") {
      console.error("Lambda error:", result.message);
      res.status(500).json({ error: result.message });
      return;
    }

    res.json({ predictions: result.predictions });
  } catch (err) {
    console.error("Identification failed:", err);
    res.status(500).json({ error: "Identification failed" });
  } finally {
    try {
      await s3.send(new DeleteObjectCommand({ Bucket: BUCKET, Key: s3Key }));
    } catch {}
  }
});

export default router;

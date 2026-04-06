import { Router } from "express";
import { randomUUID } from "crypto";
import multer from "multer";
import { S3Client, PutObjectCommand, DeleteObjectCommand } from "@aws-sdk/client-s3";
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";

const router = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

const region = process.env.AWS_REGION || "us-east-2";
const s3 = new S3Client({ region });
const lambda = new LambdaClient({ region });
const BUCKET = process.env.IDENTIFY_BUCKET || "horse-id-temp-uploads";
const FUNCTION_NAME = process.env.ML_WORKER_LAMBDA || "horse-id-ml-worker";

// POST /api/identify — upload image, get ranked matches via ML worker Lambda
router.post("/", upload.single("image"), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "No image uploaded" });
    return;
  }

  const herdId = req.body.herd_id ? Number(req.body.herd_id) : null;
  const topK = req.body.top_k ? Number(req.body.top_k) : 5;
  const s3Key = `tmp/identify-${randomUUID()}.jpg`;

  try {
    // Upload image to S3
    await s3.send(new PutObjectCommand({
      Bucket: BUCKET,
      Key: s3Key,
      Body: req.file.buffer,
      ContentType: req.file.mimetype,
    }));

    // Invoke Lambda with S3 reference
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
    // Clean up S3 temp file
    try {
      await s3.send(new DeleteObjectCommand({ Bucket: BUCKET, Key: s3Key }));
    } catch {}
  }
});

export default router;

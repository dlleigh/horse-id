import "./env.js";
import express from "express";
import cors from "cors";
import herdsRouter from "./routes/herds.js";
import horsesRouter from "./routes/horses.js";
import photosRouter from "./routes/photos.js";
import syncRouter from "./routes/sync.js";
import identifyRouter from "./routes/identify.js";
import statsRouter from "./routes/stats.js";
import processRouter from "./routes/process.js";
import benchmarkRouter from "./routes/benchmark.js";
import { requireAuth } from "./middleware/requireAuth.js";

const app = express();
app.use(cors({ origin: true, credentials: true }));
app.use(express.json());

// Allow ?token=<jwt> query param for endpoints that can't send headers (e.g. <img src>)
app.use((req, _res, next) => {
  if (!req.headers.authorization && typeof req.query.token === "string") {
    req.headers.authorization = `Bearer ${req.query.token}`;
  }
  next();
});

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.use("/api/herds", requireAuth, herdsRouter);
app.use("/api/horses", requireAuth, horsesRouter);
app.use("/api/photos", requireAuth, photosRouter);
app.use("/api/sync", requireAuth, syncRouter);
app.use("/api/identify", requireAuth, identifyRouter);
app.use("/api/stats", requireAuth, statsRouter);
app.use("/api/process", requireAuth, processRouter);
app.use("/api/benchmark", requireAuth, benchmarkRouter);

if (!process.env.AWS_LAMBDA_FUNCTION_NAME) {
  const port = process.env.PORT ?? 3000;
  app.listen(port, () => {
    console.log(`Backend listening on port ${port}`);
  });
}

export default app;

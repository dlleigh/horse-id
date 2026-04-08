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

const app = express();
app.use(cors());
app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.use("/api/herds", herdsRouter);
app.use("/api/horses", horsesRouter);
app.use("/api/photos", photosRouter);
app.use("/api/sync", syncRouter);
app.use("/api/identify", identifyRouter);
app.use("/api/stats", statsRouter);
app.use("/api/process", processRouter);

const port = process.env.PORT ?? 3000;
app.listen(port, () => {
  console.log(`Backend listening on port ${port}`);
});

export default app;

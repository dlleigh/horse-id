import "./env.js";
import express from "express";

const app = express();
app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

const port = process.env.PORT ?? 3000;
app.listen(port, () => {
  console.log(`Backend listening on port ${port}`);
});

export default app;

import { createRemoteJWKSet, jwtVerify } from "jose";
import type { Request, Response, NextFunction } from "express";

const NEON_AUTH_URL = process.env.NEON_AUTH_URL;
if (!NEON_AUTH_URL) {
  console.warn("[auth] NEON_AUTH_URL not set — auth middleware will reject all requests");
}

const JWKS = NEON_AUTH_URL
  ? createRemoteJWKSet(new URL(`${NEON_AUTH_URL}/.well-known/jwks.json`))
  : null;

export async function requireAuth(req: Request, res: Response, next: NextFunction) {
  if (!JWKS) {
    res.status(503).json({ error: "Auth not configured" });
    return;
  }

  const token = req.headers.authorization?.split(" ")[1];
  if (!token) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }

  try {
    await jwtVerify(token, JWKS);
    next();
  } catch {
    res.status(401).json({ error: "Invalid token" });
    return;
  }
}

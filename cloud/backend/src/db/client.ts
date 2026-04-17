import { neon, type NeonQueryFunction } from "@neondatabase/serverless";
import { drizzle, type NeonHttpDatabase } from "drizzle-orm/neon-http";
import * as schema from "./schema.js";

// Lazy init — DATABASE_URL may not be available at import time (loaded from SSM)
let _sql: NeonQueryFunction<false, false> | null = null;
let _db: NeonHttpDatabase<typeof schema> | null = null;

function init() {
  if (!_sql) {
    const url = process.env.DATABASE_URL;
    if (!url) {
      throw new Error("DATABASE_URL environment variable is required");
    }
    _sql = neon(url);
    _db = drizzle(_sql, { schema });
  }
}

export const db = new Proxy({} as NeonHttpDatabase<typeof schema>, {
  get(_target, prop, receiver) {
    init();
    return Reflect.get(_db!, prop, receiver);
  },
});

export const sql = new Proxy((() => {}) as unknown as NeonQueryFunction<false, false>, {
  apply(_target, thisArg, args) {
    init();
    return Reflect.apply(_sql!, thisArg, args);
  },
  get(_target, prop, receiver) {
    init();
    return Reflect.get(_sql!, prop, receiver);
  },
});

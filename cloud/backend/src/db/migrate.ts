import "../env.js";
import { neon } from "@neondatabase/serverless";
import { readFileSync, readdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const migrationsDir = join(__dirname, "../../../db/migrations");

async function migrate() {
  const url = process.env.DATABASE_URL;
  if (!url) {
    console.error("DATABASE_URL is required");
    process.exit(1);
  }

  const sql = neon(url);

  const files = readdirSync(migrationsDir)
    .filter((f) => f.endsWith(".sql"))
    .sort();

  console.log(`Found ${files.length} migration(s)`);

  for (const file of files) {
    console.log(`Running ${file}...`);
    const content = readFileSync(join(migrationsDir, file), "utf-8");
    const statements = content
      .split(";")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    for (const stmt of statements) {
      await sql(stmt);
    }
    console.log(`  done (${statements.length} statements)`);
  }

  console.log("All migrations complete");
}

migrate().catch((err) => {
  console.error("Migration failed:", err);
  process.exit(1);
});

import { Router } from "express";
import { sql } from "drizzle-orm";
import { db } from "../db/client.js";

const router = Router();

// Simple seeded PRNG (mulberry32)
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Fisher-Yates shuffle with seeded RNG
function shuffle<T>(arr: T[], rng: () => number): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

interface FeatureRow {
  [key: string]: unknown;
  id: number;
  horse_id: number;
  horse_name: string;
  herd_name: string;
}

interface MatchRow {
  [key: string]: unknown;
  horse_id: number;
  similarity: number;
}

// POST /api/benchmark
router.post("/", async (req, res) => {
  const startTime = Date.now();
  try {
  const herdId = req.body.herdId ? Number(req.body.herdId) : null;
  const testFraction = Math.min(0.5, Math.max(0.05, req.body.testFraction ?? 0.2));
  const minPhotos = Math.max(0, Math.floor(req.body.minPhotos ?? 0));
  const mode: "individual" | "centroid" | "topk" = req.body.mode === "centroid" ? "centroid" : req.body.mode === "topk" ? "topk" : "individual";
  const topK = Math.max(1, Math.floor(req.body.topK ?? 3));
  const seed = req.body.seed ?? Math.floor(Math.random() * 1_000_000);
  const rng = mulberry32(seed);

  // 1. Fetch all features with horse/herd info (exclude excluded photos)
  const herdFilter = herdId ? sql`AND h.herd_id = ${herdId}` : sql``;
  const allFeatures = await db.execute<FeatureRow>(sql`
    SELECT f.id, f.horse_id, h.name AS horse_name, hd.name AS herd_name
    FROM features f
    JOIN photos p ON p.id = f.photo_id
    JOIN horses h ON h.id = f.horse_id
    JOIN herds hd ON hd.id = h.herd_id
    WHERE p.excluded = false ${herdFilter}
    ORDER BY f.horse_id, f.id
  `);

  // 2. Group by horse
  const byHorse = new Map<number, { name: string; herdName: string; featureIds: number[] }>();
  for (const row of allFeatures.rows) {
    let entry = byHorse.get(row.horse_id);
    if (!entry) {
      entry = { name: row.horse_name, herdName: row.herd_name, featureIds: [] };
      byHorse.set(row.horse_id, entry);
    }
    entry.featureIds.push(row.id);
  }

  // 3. Stratified split
  const testFeatures: { id: number; horseId: number }[] = [];
  const trainingIds: number[] = [];
  const horseInfo = new Map<number, { name: string; herdName: string }>();

  for (const [horseId, entry] of byHorse) {
    horseInfo.set(horseId, { name: entry.name, herdName: entry.herdName });
    if (entry.featureIds.length < 2 || (minPhotos > 0 && entry.featureIds.length < minPhotos)) {
      // Not enough photos — training only
      trainingIds.push(...entry.featureIds);
      continue;
    }
    const shuffled = shuffle(entry.featureIds, rng);
    const testCount = Math.max(1, Math.round(shuffled.length * testFraction));
    // Ensure at least 1 remains in training
    const actualTestCount = Math.min(testCount, shuffled.length - 1);
    for (let i = 0; i < actualTestCount; i++) {
      testFeatures.push({ id: shuffled[i], horseId });
    }
    for (let i = actualTestCount; i < shuffled.length; i++) {
      trainingIds.push(shuffled[i]);
    }
  }

  if (testFeatures.length === 0) {
    res.status(400).json({
      error: "Not enough photos for benchmarking (need horses with 2+ photos)",
    });
    return;
  }

  // Exclude test IDs from the training set (smaller set = avoids Postgres ROW limit)
  const testIdSet = sql.raw(testFeatures.map((t) => t.id).join(","));
  const herdJoinFilter = herdId
    ? sql`JOIN horses h ON h.id = f.horse_id AND h.herd_id = ${herdId}`
    : sql``;

  // 4. Run similarity queries in parallel (batched to avoid overwhelming DB)
  const CONCURRENCY = 10;
  const queryResults: { test: typeof testFeatures[0]; matches: MatchRow[] }[] = [];

  for (let i = 0; i < testFeatures.length; i += CONCURRENCY) {
    const batch = testFeatures.slice(i, i + CONCURRENCY);
    const batchResults = await Promise.all(
      batch.map(async (test) => {
        const matches = mode === "centroid"
          ? await db.execute<MatchRow>(sql`
              SELECT f.horse_id,
                1 - (AVG(f.embedding) <=> (SELECT embedding FROM features WHERE id = ${test.id})) AS similarity
              FROM features f
              JOIN photos p ON p.id = f.photo_id
              ${herdJoinFilter}
              WHERE f.id NOT IN (${testIdSet})
                AND p.excluded = false
              GROUP BY f.horse_id
              ORDER BY similarity DESC
              LIMIT 5
            `)
          : mode === "topk"
          ? await db.execute<MatchRow>(sql`
              SELECT horse_id, AVG(similarity) AS similarity
              FROM (
                SELECT f.horse_id,
                  1 - (f.embedding <=> (SELECT embedding FROM features WHERE id = ${test.id})) AS similarity,
                  ROW_NUMBER() OVER (
                    PARTITION BY f.horse_id
                    ORDER BY f.embedding <=> (SELECT embedding FROM features WHERE id = ${test.id})
                  ) AS rn
                FROM features f
                JOIN photos p ON p.id = f.photo_id
                ${herdJoinFilter}
                WHERE f.id NOT IN (${testIdSet})
                  AND p.excluded = false
              ) ranked
              WHERE rn <= ${topK}
              GROUP BY horse_id
              ORDER BY similarity DESC
              LIMIT 5
            `)
          : await db.execute<MatchRow>(sql`
              SELECT * FROM (
                SELECT DISTINCT ON (f.horse_id)
                  f.horse_id,
                  1 - (f.embedding <=> (SELECT embedding FROM features WHERE id = ${test.id})) AS similarity
                FROM features f
                JOIN photos p ON p.id = f.photo_id
                ${herdJoinFilter}
                WHERE f.id NOT IN (${testIdSet})
                  AND p.excluded = false
                ORDER BY f.horse_id, f.embedding <=> (SELECT embedding FROM features WHERE id = ${test.id})
              ) sub
              ORDER BY similarity DESC
              LIMIT 5
            `);
        return { test, matches: matches.rows };
      })
    );
    queryResults.push(...batchResults);
  }

  // Aggregate results
  const perHorseAcc = new Map<number, { testPhotos: number; rank1Correct: number; totalSim: number; confusedWith: Map<number, number> }>();
  let rank1Correct = 0;
  let top5Correct = 0;
  let totalTopSim = 0;
  let totalCorrectSim = 0;
  let correctSimCount = 0;

  for (const { test, matches } of queryResults) {
    const topMatch = matches[0];

    if (topMatch) {
      totalTopSim += topMatch.similarity;

      if (topMatch.horse_id === test.horseId) {
        rank1Correct++;
        totalCorrectSim += topMatch.similarity;
        correctSimCount++;
      }

      const inTop5 = matches.some((r) => r.horse_id === test.horseId);
      if (inTop5) top5Correct++;
    }

    // Per-horse tracking
    let acc = perHorseAcc.get(test.horseId);
    if (!acc) {
      acc = { testPhotos: 0, rank1Correct: 0, totalSim: 0, confusedWith: new Map() };
      perHorseAcc.set(test.horseId, acc);
    }
    acc.testPhotos++;
    if (topMatch?.horse_id === test.horseId) {
      acc.rank1Correct++;
      acc.totalSim += topMatch.similarity;
    } else if (topMatch) {
      acc.confusedWith.set(topMatch.horse_id, (acc.confusedWith.get(topMatch.horse_id) ?? 0) + 1);
    }
  }

  // 5. Compute aggregate metrics
  const testCount = testFeatures.length;
  const horsesEvaluated = perHorseAcc.size;

  const perHorseResults = Array.from(perHorseAcc.entries()).map(
    ([horseId, acc]) => ({
      horseId,
      horseName: horseInfo.get(horseId)!.name,
      herdName: horseInfo.get(horseId)!.herdName,
      trainingPhotos: byHorse.get(horseId)!.featureIds.length - acc.testPhotos,
      testPhotos: acc.testPhotos,
      rank1Correct: acc.rank1Correct,
      accuracy: acc.testPhotos > 0 ? acc.rank1Correct / acc.testPhotos : 0,
      avgSimilarity: acc.rank1Correct > 0 ? acc.totalSim / acc.rank1Correct : 0,
      confusedWith: Array.from(acc.confusedWith.entries())
        .sort((a, b) => b[1] - a[1])
        .map(([cHorseId, count]) => ({
          horseId: cHorseId,
          horseName: horseInfo.get(cHorseId)?.name ?? `Horse #${cHorseId}`,
          herdName: horseInfo.get(cHorseId)?.herdName ?? "",
          count,
        })),
    })
  );

  perHorseResults.sort((a, b) => a.accuracy - b.accuracy || a.avgSimilarity - b.avgSimilarity);

  res.json({
    rank1Accuracy: testCount > 0 ? rank1Correct / testCount : 0,
    top5Accuracy: testCount > 0 ? top5Correct / testCount : 0,
    avgTopMatchSimilarity: testCount > 0 ? totalTopSim / testCount : 0,
    avgCorrectMatchSimilarity: correctSimCount > 0 ? totalCorrectSim / correctSimCount : 0,
    testCount,
    trainingCount: trainingIds.length,
    horsesEvaluated,
    horsesTotal: byHorse.size,
    durationMs: Date.now() - startTime,
    seed,
    mode,
    topK: mode === "topk" ? topK : undefined,
    perHorseResults,
  });
  } catch (err) {
    console.error("Benchmark error:", err);
    res.status(500).json({ error: "Benchmark failed: " + (err instanceof Error ? err.message : String(err)) });
  }
});

export default router;

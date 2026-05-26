import { getJWTToken } from '../lib/auth';

const BASE = '/api';

export interface Herd {
  id: number;
  name: string;
  horseCount: number;
  photoCount: number;
}

export interface HorseInHerd {
  id: number;
  name: string;
  status: string;
  photoCount: number;
  readyCount: number;
  thumbnailPhotoId: number | null;
}

export interface Photo {
  id: number;
  filename: string;
  processingStatus: string;
  detectionResult: string | null;
  excluded: boolean;
}

export interface HorseDetail {
  id: number;
  name: string;
  status: string;
  herdId: number;
  herdName: string;
  photos: Photo[];
}

export interface SyncRun {
  id: number;
  startedAt: string;
  completedAt: string | null;
  status: string;
  filesScanned: number;
  filesAdded: number;
  filesRemoved: number;
  filesMoved: number;
}

export interface Stats {
  total: number;
  pending: number;
  detecting: number;
  detected: number;
  extracting: number;
  ready: number;
  error: number;
  activeWorkers: number;
  syncStatus: 'running' | null;
  syncProgressTotal: number | null;
  syncProgressDone: number | null;
  syncFilesScanned: number | null;
  lastSync: {
    filesScanned: number;
    filesAdded: number;
    filesRemoved: number;
    filesMoved: number;
    warnings: string[];
  } | null;
}

export interface Prediction {
  horse_id: number;
  horse_name: string;
  herd_name: string;
  similarity: number;
  reference_photo_id: number;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const token = await getJWTToken();
  const headers = new Headers(init?.headers);
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  const res = await fetch(url, { ...init, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function getHerds(): Promise<Herd[]> {
  return fetchJson(`${BASE}/herds`);
}

export async function getHerdHorses(herdId: number): Promise<HorseInHerd[]> {
  return fetchJson(`${BASE}/herds/${herdId}/horses`);
}

export async function getHorse(horseId: number): Promise<HorseDetail> {
  return fetchJson(`${BASE}/horses/${horseId}`);
}

export async function patchPhoto(photoId: number, excluded: boolean): Promise<{ id: number; excluded: boolean }> {
  return fetchJson(`${BASE}/photos/${photoId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ excluded }),
  });
}

export function photoImageUrl(photoId: number, size?: 'thumb'): string {
  const url = `${BASE}/photos/${photoId}/image`;
  return size ? `${url}?size=${size}` : url;
}

export async function getStats(): Promise<Stats> {
  return fetchJson(`${BASE}/stats`);
}

export async function triggerSync(mode?: 'full' | 'incremental'): Promise<{ syncRunId: number }> {
  return fetchJson(`${BASE}/sync`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: mode ?? 'incremental' }),
  });
}

export async function triggerProcessing(): Promise<{ status: string }> {
  return fetchJson(`${BASE}/process`, { method: 'POST' });
}

export async function getSyncStatus(syncId: number): Promise<SyncRun> {
  return fetchJson(`${BASE}/sync/${syncId}`);
}

export interface ErrorPhoto {
  id: number;
  filename: string;
  drive_file_id: string;
  processing_status: string;
  detection_result: string | null;
  horse_name: string;
  herd_name: string;
}

export async function getErrorPhotos(): Promise<ErrorPhoto[]> {
  return fetchJson(`${BASE}/photos/errors`);
}

export async function retryPhoto(photoId: number): Promise<void> {
  await fetchJson(`${BASE}/photos/${photoId}/retry`, { method: 'POST' });
}

export async function retryAllPhotos(): Promise<{ count: number }> {
  return fetchJson(`${BASE}/photos/retry-all`, { method: 'POST' });
}

export interface ConfusionEntry {
  horseId: number;
  horseName: string;
  herdName: string;
  count: number;
}

export interface BenchmarkPerHorse {
  horseId: number;
  horseName: string;
  herdName: string;
  trainingPhotos: number;
  testPhotos: number;
  rank1Correct: number;
  avgSimilarity: number;
  confusedWith: ConfusionEntry[];
}

export interface BenchmarkResult {
  rank1Accuracy: number;
  top5Accuracy: number;
  avgTopMatchSimilarity: number;
  avgCorrectMatchSimilarity: number;
  testCount: number;
  trainingCount: number;
  horsesEvaluated: number;
  horsesTotal: number;
  durationMs: number;
  seed: number;
  mode: 'individual' | 'centroid';
  perHorseResults: BenchmarkPerHorse[];
}

export async function runBenchmark(herdId?: number, testFraction?: number, minPhotos?: number, mode?: 'individual' | 'centroid', seed?: number): Promise<BenchmarkResult> {
  return fetchJson(`${BASE}/benchmark`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ herdId, testFraction, minPhotos, mode, seed }),
  });
}

export interface HorseSearchResult {
  id: number;
  name: string;
  herdName: string;
}

export async function searchHorses(query: string): Promise<HorseSearchResult[]> {
  return fetchJson(`${BASE}/horses?q=${encodeURIComponent(query)}`);
}

export async function triggerReExtract(): Promise<{ status: string; deleted: number; reset: number }> {
  return fetchJson(`${BASE}/process/re-extract`, { method: 'POST' });
}

export async function identify(image: File, herdId?: number, topK?: number): Promise<{ predictions: Prediction[] }> {
  // 1. Get a presigned S3 upload URL from the backend
  const { uploadUrl, s3Key } = await fetchJson<{ uploadUrl: string; s3Key: string }>(
    `${BASE}/identify/upload-url`
  );

  // 2. Upload the image directly to S3 (bypasses Lambda Function URL 6MB limit)
  const uploadRes = await fetch(uploadUrl, {
    method: 'PUT',
    body: image,
    headers: { 'Content-Type': image.type || 'application/octet-stream' },
  });
  if (!uploadRes.ok) {
    throw new Error(`Image upload failed: HTTP ${uploadRes.status}`);
  }

  // 3. Tell the backend to run identification against the uploaded image
  return fetchJson(`${BASE}/identify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      s3_key: s3Key,
      herd_id: herdId,
      top_k: topK,
    }),
  });
}

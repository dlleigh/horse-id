import { SSMClient, GetParametersCommand } from "@aws-sdk/client-ssm";

export interface Config {
  databaseUrl: string;
  googleDriveDirectoryId: string;
  googleDriveServiceAccountKey: object;
}

let _config: Config | null = null;

/**
 * Load secrets from SSM Parameter Store into process.env.
 * Called once at startup in Lambda; locally env vars come from .env.
 */
const SSM_PARAM_MAP: Record<string, string> = {
  "/horse-id/database-url": "DATABASE_URL",
  "/horse-id/drive-service-account-key": "GOOGLE_DRIVE_SERVICE_ACCOUNT_KEY",
};

async function loadFromSSM(): Promise<void> {
  // Only fetch params that aren't already set as env vars
  const needed = Object.entries(SSM_PARAM_MAP).filter(
    ([, envVar]) => !process.env[envVar]
  );
  if (needed.length === 0) return;

  const ssm = new SSMClient();
  const res = await ssm.send(
    new GetParametersCommand({
      Names: needed.map(([name]) => name),
      WithDecryption: true,
    })
  );

  for (const param of res.Parameters ?? []) {
    const envVar = SSM_PARAM_MAP[param.Name!];
    if (envVar && param.Value) {
      process.env[envVar] = param.Value;
    }
  }
}

let _ssmLoaded = false;

export async function ensureConfig(): Promise<void> {
  if (!_ssmLoaded) {
    await loadFromSSM();
    _ssmLoaded = true;
  }
}

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

export function getConfig(): Config {
  if (_config) return _config;

  const keyJson = requireEnv("GOOGLE_DRIVE_SERVICE_ACCOUNT_KEY");
  let parsedKey: object;
  try {
    parsedKey = JSON.parse(keyJson);
  } catch {
    throw new Error(
      "GOOGLE_DRIVE_SERVICE_ACCOUNT_KEY must be valid JSON"
    );
  }

  _config = {
    databaseUrl: requireEnv("DATABASE_URL"),
    googleDriveDirectoryId: requireEnv("GOOGLE_DRIVE_DIRECTORY_ID"),
    googleDriveServiceAccountKey: parsedKey,
  };

  return _config;
}

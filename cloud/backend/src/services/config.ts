export interface Config {
  databaseUrl: string;
  googleDriveDirectoryId: string;
  googleDriveServiceAccountKey: object;
}

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

let _config: Config | null = null;

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

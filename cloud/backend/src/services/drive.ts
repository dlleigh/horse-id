import { google, type drive_v3 } from "googleapis";
import { getConfig } from "./config.js";

let _drive: drive_v3.Drive | null = null;

export function getDriveClient(): drive_v3.Drive {
  if (_drive) return _drive;

  const config = getConfig();
  const auth = new google.auth.GoogleAuth({
    credentials: config.googleDriveServiceAccountKey as object,
    scopes: ["https://www.googleapis.com/auth/drive.readonly"],
  });

  _drive = google.drive({ version: "v3", auth });
  return _drive;
}

export interface DriveFolder {
  id: string;
  name: string;
}

export interface DriveFile {
  id: string;
  name: string;
  md5Checksum: string | null;
}

export async function listFolders(parentId: string): Promise<DriveFolder[]> {
  const drive = getDriveClient();
  const folders: DriveFolder[] = [];
  let pageToken: string | undefined;

  do {
    const res = await drive.files.list({
      q: `'${parentId}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false`,
      fields: "nextPageToken, files(id, name)",
      supportsAllDrives: true,
      includeItemsFromAllDrives: true,
      pageSize: 1000,
      pageToken,
    });

    for (const f of res.data.files ?? []) {
      if (f.id && f.name) {
        folders.push({ id: f.id, name: f.name });
      }
    }

    pageToken = res.data.nextPageToken ?? undefined;
  } while (pageToken);

  return folders;
}

export async function listImageFiles(folderId: string): Promise<DriveFile[]> {
  const drive = getDriveClient();
  const files: DriveFile[] = [];
  let pageToken: string | undefined;

  do {
    const res = await drive.files.list({
      q: `'${folderId}' in parents and mimeType contains 'image/' and trashed = false`,
      fields: "nextPageToken, files(id, name, md5Checksum)",
      supportsAllDrives: true,
      includeItemsFromAllDrives: true,
      pageSize: 1000,
      pageToken,
    });

    for (const f of res.data.files ?? []) {
      if (f.id && f.name) {
        files.push({
          id: f.id,
          name: f.name,
          md5Checksum: f.md5Checksum ?? null,
        });
      }
    }

    pageToken = res.data.nextPageToken ?? undefined;
  } while (pageToken);

  return files;
}

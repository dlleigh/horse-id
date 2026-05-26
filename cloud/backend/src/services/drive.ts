import { drive_v3, drive } from "@googleapis/drive";
import { GoogleAuth } from "google-auth-library";
import { getConfig } from "./config.js";

let _drive: drive_v3.Drive | null = null;

export function getDriveClient(): drive_v3.Drive {
  if (_drive) return _drive;

  const config = getConfig();
  const auth = new GoogleAuth({
    credentials: config.googleDriveServiceAccountKey as object,
    scopes: ["https://www.googleapis.com/auth/drive"],
  });

  _drive = drive({ version: "v3", auth });
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

export interface DriveChange {
  fileId: string;
  name: string | null;
  parentId: string | null;
  md5Checksum: string | null;
  mimeType: string | null;
  removed: boolean;
}

export interface ChangesResult {
  changes: DriveChange[];
  newToken: string;
}

export async function getStartPageToken(): Promise<string> {
  const drive = getDriveClient();
  const res = await drive.changes.getStartPageToken({
    supportsAllDrives: true,
  });
  return res.data.startPageToken!;
}

export async function getChanges(pageToken: string): Promise<ChangesResult> {
  const drive = getDriveClient();
  const changes: DriveChange[] = [];
  let currentToken = pageToken;

  do {
    const res = await drive.changes.list({
      pageToken: currentToken,
      fields: "nextPageToken, newStartPageToken, changes(fileId, removed, file(id, name, parents, md5Checksum, mimeType, trashed))",
      supportsAllDrives: true,
      includeItemsFromAllDrives: true,
      pageSize: 1000,
    });

    for (const change of res.data.changes ?? []) {
      const file = change.file;
      changes.push({
        fileId: change.fileId!,
        name: file?.name ?? null,
        parentId: file?.parents?.[0] ?? null,
        md5Checksum: file?.md5Checksum ?? null,
        mimeType: file?.mimeType ?? null,
        removed: change.removed === true || file?.trashed === true,
      });
    }

    if (res.data.newStartPageToken) {
      return { changes, newToken: res.data.newStartPageToken };
    }
    currentToken = res.data.nextPageToken!;
  } while (currentToken);

  return { changes, newToken: currentToken };
}

export async function moveFolder(
  folderId: string,
  oldParentId: string,
  newParentId: string
): Promise<void> {
  const drive = getDriveClient();
  await drive.files.update({
    fileId: folderId,
    addParents: newParentId,
    removeParents: oldParentId,
    supportsAllDrives: true,
  });
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

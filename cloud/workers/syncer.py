"""Sync batch handler — processes Drive changes and chains to detection."""

from db import get_connection
from lambda_utils import invoke_detection

DETECTION_BATCH_SIZE = 20


def sync_batch(changes: list[dict], folder_changes: list[dict], sync_run_id: int = None) -> dict:
    """Process a batch of Drive changes: upsert herds/horses/photos, chain to detection.

    Args:
        changes: list of file changes with keys: file_id, name, parent_id, md5, mime_type, removed
        folder_changes: list of folder changes with keys: folder_id, name, parent_id, removed
        sync_run_id: optional sync run ID for progress tracking

    Returns:
        dict with counts of actions taken
    """
    conn = get_connection()
    counts = {"folders_added": 0, "folders_removed": 0, "files_added": 0,
              "files_updated": 0, "files_removed": 0}
    new_photo_rows = []

    # 1. Process folder changes (herds and horses)
    for fc in folder_changes:
        if fc.get("removed"):
            _remove_folder(conn, fc["folder_id"])
            counts["folders_removed"] += 1
        else:
            _upsert_folder(conn, fc["folder_id"], fc["name"], fc.get("parent_id"))
            counts["folders_added"] += 1

    # 2. Process file changes (photos)
    for change in changes:
        if change.get("removed"):
            _remove_photo(conn, change["file_id"])
            counts["files_removed"] += 1
        else:
            result = _upsert_photo(conn, change)
            if result == "added":
                counts["files_added"] += 1
            elif result == "updated":
                counts["files_updated"] += 1

            # Collect new/updated photos for detection dispatch
            if result in ("added", "updated"):
                row = _get_photo_for_dispatch(conn, change["file_id"])
                if row:
                    new_photo_rows.append(row)

    # 3. Update sync run progress
    if sync_run_id:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE sync_runs SET
                    files_scanned = files_scanned + %s,
                    files_added = files_added + %s,
                    files_removed = files_removed + %s,
                    last_heartbeat = now()
                WHERE id = %s""",
                (len(changes), counts["files_added"], counts["files_removed"], sync_run_id),
            )

    # 4. Chain to detection
    if new_photo_rows:
        for i in range(0, len(new_photo_rows), DETECTION_BATCH_SIZE):
            batch = new_photo_rows[i:i + DETECTION_BATCH_SIZE]
            invoke_detection(batch)
        print(f"[syncer] Dispatched {len(new_photo_rows)} photos for detection")

    print(f"[syncer] Batch done: {counts}")
    return counts


def _upsert_folder(conn, folder_id: str, name: str, parent_id: str | None):
    """Upsert a folder as either a herd or a horse based on its parent."""
    with conn.cursor() as cur:
        # Check if parent is the root (a herd folder) or a herd (a horse folder)
        # First check if parent is a known herd
        cur.execute("SELECT id FROM herds WHERE drive_folder_id = %s", (parent_id,))
        parent_herd = cur.fetchone()

        if parent_herd:
            # This is a horse folder (parent is a herd)
            herd_id = parent_herd[0]
            cur.execute("SELECT id, name, herd_id FROM horses WHERE drive_folder_id = %s", (folder_id,))
            existing = cur.fetchone()
            if existing:
                updates = []
                params = []
                if existing[1] != name:
                    updates.append("name = %s")
                    params.append(name)
                if existing[2] != herd_id:
                    updates.append("herd_id = %s")
                    params.append(herd_id)
                if updates:
                    params.append(existing[0])
                    cur.execute(f"UPDATE horses SET {', '.join(updates)} WHERE id = %s", params)
                    if existing[1] != name and existing[2] != herd_id:
                        print(f"  Moved & renamed horse: {existing[1]} -> {name}")
                    elif existing[2] != herd_id:
                        print(f"  Moved horse: {name}")
                    else:
                        print(f"  Renamed horse: {existing[1]} -> {name}")
            else:
                cur.execute(
                    "INSERT INTO horses (name, herd_id, drive_folder_id) VALUES (%s, %s, %s)",
                    (name, herd_id, folder_id),
                )
                print(f"  New horse: {name}")
        else:
            # This might be a herd folder
            cur.execute("SELECT id, name FROM herds WHERE drive_folder_id = %s", (folder_id,))
            existing = cur.fetchone()
            if existing:
                if existing[1] != name:
                    cur.execute("UPDATE herds SET name = %s WHERE id = %s", (name, existing[0]))
                    print(f"  Renamed herd: {existing[1]} -> {name}")
            else:
                cur.execute(
                    "INSERT INTO herds (name, drive_folder_id) VALUES (%s, %s)",
                    (name, folder_id),
                )
                print(f"  New herd: {name}")


def _remove_folder(conn, folder_id: str):
    """Remove a folder (cascade deletes horses/photos if herd, or photos if horse)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM horses WHERE drive_folder_id = %s", (folder_id,))
        cur.execute("DELETE FROM herds WHERE drive_folder_id = %s", (folder_id,))


def _upsert_photo(conn, change: dict) -> str:
    """Upsert a photo. Returns 'added', 'updated', or 'unchanged'."""
    file_id = change["file_id"]
    name = change["name"]
    parent_id = change.get("parent_id")
    md5 = change.get("md5")

    with conn.cursor() as cur:
        # Find the horse this photo belongs to
        cur.execute("SELECT id FROM horses WHERE drive_folder_id = %s", (parent_id,))
        horse_row = cur.fetchone()
        if not horse_row:
            # Parent folder not a known horse — skip
            return "unchanged"
        horse_id = horse_row[0]

        cur.execute(
            "SELECT id, drive_md5, horse_id, filename FROM photos WHERE drive_file_id = %s",
            (file_id,),
        )
        existing = cur.fetchone()

        if existing:
            photo_id, existing_md5, existing_horse_id, existing_filename = existing
            updates = []
            params = []
            if existing_md5 != md5:
                updates.append("drive_md5 = %s")
                params.append(md5)
            if name and name != existing_filename:
                updates.append("filename = %s")
                params.append(name)
            if existing_horse_id != horse_id:
                updates.append("horse_id = %s")
                params.append(horse_id)
                updates.append("processing_status = 'pending'")
            if updates:
                params.append(photo_id)
                cur.execute(
                    f"UPDATE photos SET {', '.join(updates)} WHERE id = %s",
                    params,
                )
                return "updated"
            return "unchanged"
        else:
            cur.execute(
                """INSERT INTO photos (horse_id, filename, drive_file_id, drive_md5, processing_status)
                   VALUES (%s, %s, %s, %s, 'pending')""",
                (horse_id, name, file_id, md5),
            )
            return "added"


def _remove_photo(conn, file_id: str):
    """Remove a photo by drive file ID."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM photos WHERE drive_file_id = %s", (file_id,))


def _get_photo_for_dispatch(conn, file_id: str) -> dict | None:
    """Get photo data needed for detection Lambda dispatch."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT p.id, p.horse_id, p.drive_file_id, p.filename
               FROM photos p
               WHERE p.drive_file_id = %s AND p.processing_status = 'pending'""",
            (file_id,),
        )
        row = cur.fetchone()
        if row:
            return {"id": row[0], "horse_id": row[1], "drive_file_id": row[2], "filename": row[3]}
        return None

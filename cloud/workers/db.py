"""Database client for ML workers. Uses psycopg2 + pgvector."""

import os

import psycopg2
from pgvector.psycopg2 import register_vector


_conn = None


def _is_alive(conn):
    """Check if a connection is actually usable (not just client-side open)."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def get_connection():
    global _conn
    if _conn is not None and not _conn.closed and _is_alive(_conn):
        return _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
    _conn = psycopg2.connect(os.environ["DATABASE_URL"])
    _conn.autocommit = True
    register_vector(_conn)
    return _conn


def update_photo_status(photo_id: int, status: str, detection_result: str = None):
    conn = get_connection()
    with conn.cursor() as cur:
        if detection_result is not None:
            cur.execute(
                "UPDATE photos SET processing_status = %s, detection_result = %s, updated_at = now() WHERE id = %s",
                (status, detection_result, photo_id),
            )
        else:
            cur.execute(
                "UPDATE photos SET processing_status = %s, updated_at = now() WHERE id = %s",
                (status, photo_id),
            )


def insert_feature(photo_id: int, horse_id: int, embedding: list[float]):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO features (photo_id, horse_id, embedding)
               VALUES (%s, %s, %s::vector)
               ON CONFLICT (photo_id) DO UPDATE SET embedding = EXCLUDED.embedding, extracted_at = now()""",
            (photo_id, horse_id, str(embedding)),
        )


def get_pending_photos(limit: int = 100) -> list[dict]:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """SELECT p.id, p.horse_id, p.drive_file_id, p.filename, h.name as horse_name
               FROM photos p
               JOIN horses h ON h.id = p.horse_id
               WHERE p.processing_status = 'pending' AND p.excluded = false
               ORDER BY p.id
               LIMIT %s""",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_detected_photos(limit: int = 100) -> list[dict]:
    """Get photos that have been detected as SINGLE and need feature extraction."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """SELECT p.id, p.horse_id, p.drive_file_id, p.filename, h.name as horse_name
               FROM photos p
               JOIN horses h ON h.id = p.horse_id
               WHERE p.processing_status = 'detected'
                 AND p.detection_result = 'SINGLE'
                 AND p.excluded = false
                 AND NOT EXISTS (SELECT 1 FROM features f WHERE f.photo_id = p.id)
               ORDER BY p.id
               LIMIT %s""",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_herd_id_by_name(name: str) -> int | None:
    """Case-insensitive herd lookup. Returns herd_id or None."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM herds WHERE lower(name) = lower(%s)", (name,))
        row = cur.fetchone()
        return row[0] if row else None


def get_all_herd_names() -> list[str]:
    """Return all herd names for error messages."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM herds ORDER BY name")
        return [row[0] for row in cur.fetchall()]


def query_similar(embedding: list[float], limit: int = 5, herd_id: int = None) -> list[dict]:
    """Return the top `limit` distinct horses ranked by best-photo similarity.

    Uses DISTINCT ON to get the single best photo per horse, then sorts
    and limits to `limit` horses.
    """
    conn = get_connection()
    herd_filter = "AND h.herd_id = %s" if herd_id is not None else ""
    params = [str(embedding), str(embedding)]
    if herd_id is not None:
        params.insert(1, herd_id)
    params.append(limit)

    sql = f"""
        SELECT * FROM (
            SELECT DISTINCT ON (f.horse_id)
                   f.horse_id, h.name as horse_name, hd.name as herd_name,
                   1 - (f.embedding <=> %s::vector) as similarity,
                   p.id as photo_id, p.filename
            FROM features f
            JOIN horses h ON h.id = f.horse_id
            JOIN herds hd ON hd.id = h.herd_id
            JOIN photos p ON p.id = f.photo_id
            WHERE TRUE {herd_filter}
            ORDER BY f.horse_id, f.embedding <=> %s::vector
        ) sub
        ORDER BY similarity DESC
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

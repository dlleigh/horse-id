-- Drive Changes API sync state
CREATE TABLE IF NOT EXISTS drive_sync_state (
  id int PRIMARY KEY DEFAULT 1,
  changes_token text,
  updated_at timestamptz DEFAULT now()
);

-- Track when photos change status (for stuck photo detection)
ALTER TABLE photos ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();

-- Auto-update updated_at on any photo row change
CREATE OR REPLACE FUNCTION update_photos_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER photos_updated_at_trigger
  BEFORE UPDATE ON photos
  FOR EACH ROW
  EXECUTE FUNCTION update_photos_updated_at();

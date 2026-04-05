#!/usr/bin/env python3
"""
Migrate horse photos from flat directory to herd/horse folder structure on Google Drive.

Reads the merged manifest and horse_herds.csv to determine which horses belong to which herds,
then reorganizes photos from the flat horse_photos/ directory into:

    <drive_root>/herds/
        <Herd A>/
            <Horse 1>/
                photo1.jpg
                photo2.jpg
            <Horse 2>/
                photo1.jpg
        <Herd B>/
            ...

Handles numbered variants (e.g., "Bam Bam 1" in herds CSV, "Bam Bam" in manifest):
  - If only one numbered variant exists, all photos go to that variant's herd under the basename.
  - If multiple numbered variants exist in different herds, photos are placed in an _unsorted/
    folder and empty placeholder folders are created in each herd for manual sorting.

Usage:
    python migrate_to_drive.py --drive-root <path_to_drive_data_dir> [--dry-run]
"""

import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict

import pandas as pd


def load_horse_herds(herds_csv_path):
    """Load herds CSV. Returns list of dicts with horse_name, herd, basename."""
    rows = []
    with open(herds_csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def build_herd_mapping(herds_rows):
    """Build horse_name -> herd mapping and basename -> [(horse_name, herd)] for variants.

    Returns:
        direct_mapping: dict of horse_name -> herd (for exact matches, excluding multi-herd ambiguity)
        basename_variants: dict of basename -> [(horse_name, herd)] for numbered variant groups
    """
    # Direct mapping: horse_name -> herd (skip duplicates)
    direct_mapping = {}
    seen_dupes = set()
    for row in herds_rows:
        name = row["horse_name"]
        herd = row["herd"]
        if name in direct_mapping:
            seen_dupes.add(name)
            direct_mapping[name] = None
        else:
            direct_mapping[name] = herd
    direct_mapping = {k: v for k, v in direct_mapping.items() if v is not None}
    if seen_dupes:
        for name in sorted(seen_dupes):
            print(f"  WARNING: '{name}' appears in multiple herds (direct mapping skipped)")

    # Basename -> variants (for numbered names like "Cowboy 1", "Cowboy 2")
    basename_variants = defaultdict(list)
    for row in herds_rows:
        basename_variants[row["basename"]].append((row["horse_name"], row["herd"]))

    return direct_mapping, basename_variants


def sanitize_folder_name(name):
    """Replace characters that are problematic in folder names."""
    return name.strip().replace("/", "-")


def resolve_numbered_variants(manifest_name, direct_mapping, basename_variants):
    """Try to resolve an unmapped manifest name via numbered variant logic.

    Returns:
        (folder_name, herd_name) if resolvable, or None if ambiguous/unresolvable.
        Also returns a reason string for reporting.
    """
    # Check if this name is a basename of any numbered variant group
    if manifest_name not in basename_variants:
        return None, "no herd mapping"

    variants = basename_variants[manifest_name]
    # Filter to variants that are actually numbered (name != basename)
    numbered = [(name, herd) for name, herd in variants if name != manifest_name]

    if not numbered:
        return None, "no herd mapping"

    # Collect unique herds across all numbered variants
    unique_herds = list(set(herd for _, herd in numbered))

    if len(unique_herds) == 1:
        # All numbered variants are in the same herd — easy, use basename as folder name
        return (manifest_name, unique_herds[0]), "single-herd numbered variant"

    if len(numbered) == 1:
        # Only one numbered variant exists — all photos must be this horse
        return (manifest_name, numbered[0][1]), "single numbered variant"

    # Multiple numbered variants in different herds — ambiguous
    return None, f"ambiguous numbered variants: {numbered}"


def main():
    parser = argparse.ArgumentParser(description="Migrate horse photos to herd/horse folder structure")
    parser.add_argument(
        "--drive-root",
        required=True,
        help="Path to the Google Drive data directory (e.g., ~/Library/CloudStorage/.../horse-id/data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually moving files",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root for the herd/horse structure (default: <drive-root>/herds/)",
    )
    args = parser.parse_args()

    drive_root = os.path.expanduser(args.drive_root)
    if not os.path.isdir(drive_root):
        print(f"Error: Drive root not found: {drive_root}")
        sys.exit(1)

    photos_dir = os.path.join(drive_root, "horse_photos")
    herds_csv = os.path.join(drive_root, "horse_herds.csv")
    manifest_csv = os.path.join(drive_root, "horse_photos_manifest_merged.csv")
    output_root = args.output_root or os.path.join(drive_root, "herds")

    for path, label in [(photos_dir, "Photos dir"), (herds_csv, "Herds CSV"), (manifest_csv, "Manifest")]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}")
            sys.exit(1)

    # Load data
    print("Loading horse herds...")
    herds_rows = load_horse_herds(herds_csv)
    direct_mapping, basename_variants = build_herd_mapping(herds_rows)
    print(f"  {len(direct_mapping)} horses with direct herd mappings")

    print("Loading manifest...")
    manifest = pd.read_csv(manifest_csv)
    print(f"  {len(manifest)} total rows")

    # Filter to non-excluded, single-horse photos with a normalized name
    active = manifest[manifest["status"] != "EXCLUDE"].copy()
    active = active[active["normalized_horse_name"].notna()]
    multiple = active[active["num_horses_detected"] == "MULTIPLE"]
    active = active[active["num_horses_detected"] != "MULTIPLE"]
    print(f"  {len(active)} active photos ({len(multiple)} MULTIPLE-horse photos skipped)")

    # Group photos by normalized_horse_name
    photos_by_horse = defaultdict(list)
    for _, row in active.iterrows():
        photos_by_horse[row["normalized_horse_name"]].append(row["filename"])

    # Determine what to migrate
    migrate_plan = []  # (folder_name, herd_name, [filenames])
    skipped_horses = {}  # horse_name -> (photo_count, reason)
    ambiguous_horses = {}  # horse_name -> (photo_count, variants)
    resolved_variants = []  # For reporting

    for horse_name, filenames in sorted(photos_by_horse.items()):
        # First try direct mapping
        herd = direct_mapping.get(horse_name)
        if herd:
            migrate_plan.append((horse_name, herd, filenames))
            continue

        # Try numbered variant resolution
        result, reason = resolve_numbered_variants(horse_name, direct_mapping, basename_variants)
        if result:
            folder_name, herd = result
            migrate_plan.append((folder_name, herd, filenames))
            resolved_variants.append((horse_name, folder_name, herd, len(filenames), reason))
            continue

        # Check if this is an ambiguous numbered variant
        if "ambiguous" in reason:
            variants = basename_variants.get(horse_name, [])
            ambiguous_horses[horse_name] = (len(filenames), variants)
        else:
            skipped_horses[horse_name] = (len(filenames), reason)

    # Summary
    total_photos_to_migrate = sum(len(f) for _, _, f in migrate_plan)
    total_photos_skipped = sum(v[0] for v in skipped_horses.values())
    total_photos_ambiguous = sum(v[0] for v in ambiguous_horses.values())
    herds_needed = sorted(set(herd for _, herd, _ in migrate_plan))

    print(f"\n=== Migration Plan ===")
    print(f"Herds to create: {len(herds_needed)}")
    print(f"Horses to migrate: {len(migrate_plan)}")
    print(f"Photos to copy: {total_photos_to_migrate}")
    if resolved_variants:
        print(f"  (includes {len(resolved_variants)} resolved numbered variants)")
    print(f"Ambiguous horses (need manual sort): {len(ambiguous_horses)} ({total_photos_ambiguous} photos)")
    print(f"Skipped horses (no mapping at all): {len(skipped_horses)} ({total_photos_skipped} photos)")
    print(f"Output directory: {output_root}")

    print(f"\nHerds:")
    for herd in herds_needed:
        horses_in_herd = [(h, f) for h, hr, f in migrate_plan if hr == herd]
        photo_count = sum(len(f) for _, f in horses_in_herd)
        print(f"  {herd}: {len(horses_in_herd)} horses, {photo_count} photos")

    if resolved_variants:
        print(f"\nResolved numbered variants:")
        for manifest_name, folder_name, herd, count, reason in resolved_variants:
            print(f"  {manifest_name} -> {folder_name} in {herd} ({count} photos, {reason})")

    if ambiguous_horses:
        print(f"\nAmbiguous numbered variants (photos -> _unsorted/{{}}, empty folders created in herds):")
        for horse, (count, variants) in sorted(ambiguous_horses.items()):
            variant_str = ", ".join(f"{n} ({h})" for n, h in variants)
            print(f"  {horse}: {count} photos — variants: {variant_str}")

    if skipped_horses:
        print(f"\nSkipped horses (no herd mapping):")
        for horse, (count, reason) in sorted(skipped_horses.items()):
            print(f"  {horse}: {count} photos")

    if args.dry_run:
        print("\n[DRY RUN] No files moved.")
        return

    # Confirm
    total_to_handle = total_photos_to_migrate + total_photos_ambiguous
    print(f"\nThis will create folders and copy {total_to_handle} files.")
    if ambiguous_horses:
        print(f"  ({total_photos_ambiguous} ambiguous photos go to _unsorted/ for manual sorting)")
    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # Execute migration
    print("\nMigrating...")
    copied = 0
    missing = 0
    errors = 0

    for folder_name, herd_name, filenames in migrate_plan:
        herd_folder = sanitize_folder_name(herd_name)
        horse_folder = sanitize_folder_name(folder_name)
        dest_dir = os.path.join(output_root, herd_folder, horse_folder)

        os.makedirs(dest_dir, exist_ok=True)

        for filename in filenames:
            src = os.path.join(photos_dir, filename)
            dst = os.path.join(dest_dir, filename)

            if not os.path.exists(src):
                missing += 1
                continue

            if os.path.exists(dst):
                # Already there (re-run safety)
                continue

            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                print(f"  ERROR copying {filename}: {e}")
                errors += 1

        if copied % 100 == 0 and copied > 0:
            print(f"  ...copied {copied} files")

    # Handle ambiguous horses: move photos to _unsorted/<basename>/ and create empty herd folders
    unsorted_dir = os.path.join(output_root, "_unsorted")
    unsorted_copied = 0

    for horse_name, (count, variants) in ambiguous_horses.items():
        filenames = photos_by_horse[horse_name]
        basename_folder = sanitize_folder_name(horse_name)

        # Move photos to _unsorted/<basename>/
        unsorted_dest = os.path.join(unsorted_dir, basename_folder)
        os.makedirs(unsorted_dest, exist_ok=True)

        for filename in filenames:
            src = os.path.join(photos_dir, filename)
            dst = os.path.join(unsorted_dest, filename)

            if not os.path.exists(src):
                missing += 1
                continue
            if os.path.exists(dst):
                continue

            try:
                shutil.copy2(src, dst)
                unsorted_copied += 1
            except Exception as e:
                print(f"  ERROR copying {filename}: {e}")
                errors += 1

        # Create empty placeholder folders in each variant's herd
        for variant_name, herd in variants:
            herd_folder = sanitize_folder_name(herd)
            # Use basename (not numbered name) as folder name since that's what we want long-term
            horse_folder = sanitize_folder_name(horse_name)
            placeholder_dir = os.path.join(output_root, herd_folder, horse_folder)
            os.makedirs(placeholder_dir, exist_ok=True)

    print(f"\n=== Migration Complete ===")
    print(f"Files copied to herd folders: {copied}")
    print(f"Files copied to _unsorted/: {unsorted_copied}")
    print(f"Files missing (not in photos dir): {missing}")
    print(f"Errors: {errors}")

    # Verify
    print(f"\nVerifying folder structure...")
    for herd in herds_needed:
        herd_folder = sanitize_folder_name(herd)
        herd_path = os.path.join(output_root, herd_folder)
        if os.path.isdir(herd_path):
            horse_dirs = [d for d in os.listdir(herd_path) if os.path.isdir(os.path.join(herd_path, d))]
            total = sum(
                len([f for f in os.listdir(os.path.join(herd_path, d)) if not f.startswith(".")])
                for d in horse_dirs
            )
            empty = [d for d in horse_dirs
                     if len([f for f in os.listdir(os.path.join(herd_path, d)) if not f.startswith(".")]) == 0]
            empty_note = f" ({len(empty)} empty, awaiting manual sort)" if empty else ""
            print(f"  {herd}: {len(horse_dirs)} horses, {total} photos{empty_note}")

    if os.path.isdir(unsorted_dir):
        unsorted_horses = [d for d in os.listdir(unsorted_dir) if os.path.isdir(os.path.join(unsorted_dir, d))]
        unsorted_total = sum(
            len([f for f in os.listdir(os.path.join(unsorted_dir, d)) if not f.startswith(".")])
            for d in unsorted_horses
        )
        print(f"\n  _unsorted/: {len(unsorted_horses)} horses, {unsorted_total} photos")
        print(f"  These need manual sorting into the correct herd folders.")


if __name__ == "__main__":
    main()

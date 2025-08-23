#!/usr/bin/env python3
"""
Unified Horse Identification Pipeline

Combines the entire pipeline into a single atomic operation:
1. Ingestion (email or directory)
2. Name normalization 
3. Multi-horse detection
4. Identity merging

Includes distributed locking to prevent conflicts across multiple users.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Import pipeline lock utilities
from pipeline_lock import pipeline_lock, PipelineLockExists, update_lock_stage

def get_script_dir():
    """Get the directory where this script is located."""
    return Path(__file__).parent.absolute()

def run_script(script_name, description, stage_name=None):
    """
    Run a pipeline script with error handling.
    
    Args:
        script_name: Name of the Python script to run
        description: Human-readable description for logging
        stage_name: Stage name for lock updates (optional)
    """
    script_path = get_script_dir() / script_name
    
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting: {description}")
    print(f"   Script: {script_name}")
    if stage_name:
        print(f"   Stage: {stage_name}")
        update_lock_stage(stage_name)
    print(f"{'='*60}")
    
    try:
        # Run the script using the same Python interpreter
        result = subprocess.run([
            sys.executable, str(script_path)
        ], check=True, cwd=get_script_dir())
        
        print(f"✅ Completed: {description}")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"   Script: {script_name}")
        print(f"   Exit code: {e.returncode}")
        raise

def prompt_directory_info():
    """Prompt user for directory ingestion information."""
    print("\n📁 Directory-based ingestion selected")
    print("This will process images organized in subdirectories (one per horse).")
    print()
    
    # Get directory path
    while True:
        parent_dir = input("Enter the full path to the directory containing horse subdirectories: ").strip()
        if not parent_dir:
            print("Please enter a directory path.")
            continue
        
        parent_dir = os.path.expanduser(parent_dir)  # Expand ~ to home directory
        
        if not os.path.isdir(parent_dir):
            print(f"Error: '{parent_dir}' is not a valid directory. Please try again.")
            continue
        
        # Check if it contains subdirectories
        subdirs = [d for d in os.listdir(parent_dir) 
                  if os.path.isdir(os.path.join(parent_dir, d)) and not d.startswith('.')]
        
        if not subdirs:
            print(f"Warning: No subdirectories found in '{parent_dir}'")
            confirm = input("Continue anyway? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                continue
        else:
            print(f"Found {len(subdirs)} subdirectories: {', '.join(subdirs[:5])}")
            if len(subdirs) > 5:
                print(f"... and {len(subdirs) - 5} more")
        
        break
    
    # Get date
    while True:
        date_str = input("Enter the date for these photos (YYYYMMDD): ").strip()
        
        try:
            # Validate date format
            datetime.strptime(date_str, '%Y%m%d')
            break
        except ValueError:
            print("Error: Invalid date format. Please use YYYYMMDD (e.g., 20240315).")
    
    return parent_dir, date_str

def run_directory_ingestion(parent_dir=None, date_str=None, interactive=True):
    """Run directory-based ingestion with optional parameters."""
    
    if interactive and (not parent_dir or not date_str):
        parent_dir, date_str = prompt_directory_info()
    
    # Validate parameters for non-interactive mode
    if not parent_dir or not date_str:
        raise ValueError("Directory path and date are required for directory ingestion")
    
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Directory not found: {parent_dir}")
    
    # Run ingest_from_dir.py with input simulation
    script_path = get_script_dir() / "ingest_from_dir.py"
    
    if not script_path.exists():
        raise FileNotFoundError(f"Ingestion script not found: {script_path}")
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting: Directory-based image ingestion")
    print(f"   Script: ingest_from_dir.py")
    print(f"   Directory: {parent_dir}")
    print(f"   Date: {date_str}")
    update_lock_stage("ingest_from_dir")
    print(f"{'='*60}")
    
    try:
        # Create input for the script
        script_input = f"{parent_dir}\n{date_str}\n"
        
        # Run the script with input
        result = subprocess.run([
            sys.executable, str(script_path)
        ], input=script_input, text=True, check=True, cwd=get_script_dir())
        
        print(f"✅ Completed: Directory-based image ingestion")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: Directory-based image ingestion")
        print(f"   Exit code: {e.returncode}")
        raise

def main():
    """Main pipeline execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the complete horse identification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Directory ingestion (interactive)
  %(prog)s --email                            # Email ingestion
  %(prog)s --dir --path /path/to/horses --date 20240315  # Non-interactive directory
  %(prog)s --force                            # Force override existing locks
        """
    )
    
    # Ingestion method
    ingestion_group = parser.add_mutually_exclusive_group()
    ingestion_group.add_argument('--email', action='store_true',
                               help='Use email-based ingestion')
    ingestion_group.add_argument('--dir', action='store_true', 
                               help='Use directory-based ingestion (default)')
    
    # Directory ingestion parameters
    parser.add_argument('--path', type=str,
                       help='Directory path for non-interactive directory ingestion')
    parser.add_argument('--date', type=str,
                       help='Date (YYYYMMDD) for non-interactive directory ingestion')
    
    # Lock management
    parser.add_argument('--force', action='store_true',
                       help='Force start, overriding any existing locks')
    parser.add_argument('--check-lock', action='store_true',
                       help='Check lock status and exit')
    
    args = parser.parse_args()
    
    # Handle lock checking
    if args.check_lock:
        from pipeline_lock import read_lock_file, format_lock_age
        lock_data = read_lock_file()
        if lock_data:
            print("🔒 Pipeline lock is active:")
            print(f"  Created: {lock_data.get('timestamp', 'Unknown')}")
            print(f"  User: {lock_data.get('user', 'Unknown')}@{lock_data.get('hostname', 'Unknown')}")
            print(f"  Operation: {lock_data.get('operation', 'Unknown')}")
            if 'stage' in lock_data:
                print(f"  Stage: {lock_data['stage']}")
            print(f"  Age: {format_lock_age(lock_data.get('timestamp', ''))}")
            sys.exit(1)
        else:
            print("🔓 No pipeline lock found")
            sys.exit(0)
    
    # Determine ingestion method (default to directory)
    use_email = args.email
    use_directory = not use_email  # Default to directory if not email
    
    # Validate directory parameters
    if use_directory and args.path and not args.date:
        parser.error("--date is required when --path is specified")
    if use_directory and args.date and not args.path:
        parser.error("--path is required when --date is specified")
    
    # Non-interactive mode requires both path and date
    interactive_mode = use_directory and not (args.path and args.date)
    
    print("🐴 Horse Identification Pipeline")
    print("=" * 40)
    
    if use_email:
        print("📧 Mode: Email ingestion")
    else:
        print("📁 Mode: Directory ingestion")
        if interactive_mode:
            print("   Interactive: Yes")
        else:
            print("   Interactive: No")
            print(f"   Directory: {args.path}")
            print(f"   Date: {args.date}")
    
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Acquire pipeline lock
        with pipeline_lock("pipeline", "initializing", check_existing=True, force=args.force):
            
            # Stage 1: Ingestion
            if use_email:
                run_script("ingest_from_email.py", "Email ingestion", "ingest_from_email")
            else:
                if interactive_mode:
                    run_directory_ingestion(interactive=True)
                else:
                    run_directory_ingestion(args.path, args.date, interactive=False)
            
            # Stage 2: Name normalization
            run_script("normalize_horse_names.py", "Horse name normalization", "normalize_names")
            
            # Stage 3: Multi-horse detection  
            run_script("multi_horse_detector.py", "Multi-horse detection", "detect_horses")
            
            # Stage 4: Identity merging
            run_script("merge_horse_identities.py", "Horse identity merging", "merge_identities")
            
            # Pipeline completed successfully
            update_lock_stage("completed")
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"🕐 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\nNext steps:")
            print("- Review results: streamlit run manage_horses.py")
            print("- Generate galleries: python generate_gallery.py")
            print("- Extract features: python extract_features.py")
    
    except PipelineLockExists as e:
        print(f"\n🛑 Pipeline execution cancelled: {e}")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\n\n🛑 Pipeline interrupted by user")
        print("Lock has been automatically cleaned up.")
        sys.exit(1)
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed at stage: {e}")
        print("Lock has been automatically cleaned up.")
        print("\nTroubleshooting:")
        print("- Check error messages above")
        print("- Verify all dependencies are installed")
        print("- Ensure config.yml is properly configured")
        print("- Run individual scripts to isolate the issue")
        sys.exit(e.returncode)
    
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("Lock has been automatically cleaned up.")
        print("\nPlease report this issue with the full error traceback.")
        raise

if __name__ == '__main__':
    main()
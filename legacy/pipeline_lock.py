#!/usr/bin/env python3
"""
Pipeline Lock Management Utilities

Provides cross-machine lock management for the horse identification pipeline.
Handles lock creation, validation, cleanup, and interactive override prompts.
"""

import os
import json
import socket
import getpass
import signal
import sys
import atexit
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from contextlib import contextmanager

# --- Load Configuration ---
from config_utils import load_config, get_data_root

config = load_config()
DATA_ROOT = get_data_root(config)
LOCK_FILE = os.path.join(DATA_ROOT, '.pipeline_lock')

class PipelineLockError(Exception):
    """Exception raised when pipeline lock operations fail."""
    pass

class PipelineLockExists(PipelineLockError):
    """Exception raised when trying to create a lock that already exists."""
    pass

def get_lock_metadata(operation: str = "pipeline", stage: Optional[str] = None) -> Dict:
    """Generate lock metadata for the current process."""
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "operation": operation,
        "pid": os.getpid()
    }
    
    if stage:
        metadata["stage"] = stage
    
    return metadata

def format_lock_age(lock_timestamp: str) -> str:
    """Format lock age as a human-readable string."""
    try:
        lock_time = datetime.fromisoformat(lock_timestamp.replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        age = current_time - lock_time
        
        days = age.days
        hours = age.seconds // 3600
        minutes = (age.seconds % 3600) // 60
        
        if days > 0:
            return f"{days} day{'s' if days != 1 else ''}, {hours} hour{'s' if hours != 1 else ''}"
        elif hours > 0:
            return f"{hours} hour{'s' if hours != 1 else ''}, {minutes} minute{'s' if minutes != 1 else ''}"
        else:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
            
    except Exception:
        return "unknown age"

def read_lock_file() -> Optional[Dict]:
    """Read and parse the lock file if it exists."""
    if not os.path.exists(LOCK_FILE):
        return None
    
    try:
        with open(LOCK_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read lock file: {e}")
        return None

def create_lock_file(operation: str = "pipeline", stage: Optional[str] = None) -> None:
    """Create a lock file with metadata."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    
    # Check if lock already exists
    if os.path.exists(LOCK_FILE):
        raise PipelineLockExists("Lock file already exists")
    
    # Create lock with metadata
    metadata = get_lock_metadata(operation, stage)
    
    try:
        with open(LOCK_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except IOError as e:
        raise PipelineLockError(f"Could not create lock file: {e}")

def remove_lock_file() -> None:
    """Remove the lock file if it exists."""
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except IOError as e:
            print(f"Warning: Could not remove lock file: {e}")

def update_lock_stage(stage: str) -> None:
    """Update the current stage in the lock file."""
    lock_data = read_lock_file()
    if not lock_data:
        return
    
    lock_data["stage"] = stage
    lock_data["last_updated"] = datetime.now(timezone.utc).isoformat()
    
    try:
        with open(LOCK_FILE, 'w') as f:
            json.dump(lock_data, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not update lock stage: {e}")

def prompt_override_lock(lock_data: Dict, operation: str = "operation") -> bool:
    """
    Prompt user to override an existing lock with detailed information.
    
    Args:
        lock_data: Lock metadata dictionary
        operation: Description of what operation is being blocked
    
    Returns:
        True if user chooses to override, False otherwise
    """
    print(f"\n⚠️  Pipeline lock detected!")
    print("=" * 60)
    
    # Show lock details
    timestamp = lock_data.get("timestamp", "Unknown")
    user = lock_data.get("user", "Unknown")
    hostname = lock_data.get("hostname", "Unknown")
    lock_operation = lock_data.get("operation", "Unknown")
    stage = lock_data.get("stage")
    age = format_lock_age(timestamp)
    
    print(f"Lock created: {timestamp}")
    print(f"Created by: {user}@{hostname}")
    print(f"Operation: {lock_operation}")
    if stage:
        print(f"Stage: {stage}")
    print(f"Lock age: {age}")
    
    print("\nThis could indicate:")
    print("- Pipeline is currently running on another machine")  
    print("- Previous pipeline was killed or crashed")
    print("- System was powered off during pipeline execution")
    
    # Determine if this is the same user
    current_user = getpass.getuser()
    current_hostname = socket.gethostname()
    same_user = (user == current_user and hostname == current_hostname)
    
    if same_user:
        print(f"\n💡 This lock was created by you on this machine.")
        print("   It might be from a previous crashed process.")
    
    print(f"\n🚨 WARNING: Multiple {operation}s running simultaneously will cause data corruption!")
    print("   Only override if you're certain no pipeline is running elsewhere.")
    
    while True:
        try:
            response = input(f"\nOverride lock and continue with {operation}? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                print("⚡ Lock override confirmed. Proceeding with caution...")
                return True
            elif response in ['n', 'no', '']:
                print("🛑 Operation cancelled to prevent conflicts.")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no (default).")
        except (KeyboardInterrupt, EOFError):
            print("\n🛑 Operation cancelled.")
            return False

def check_and_prompt_for_lock(operation: str = "operation") -> bool:
    """
    Check for existing lock and prompt user if found.
    
    Args:
        operation: Description of what operation is being attempted
    
    Returns:
        True if safe to proceed (no lock or user overrode), False otherwise
    """
    lock_data = read_lock_file()
    if not lock_data:
        return True  # No lock exists, safe to proceed
    
    # Lock exists, prompt user
    return prompt_override_lock(lock_data, operation)

@contextmanager
def pipeline_lock(operation: str = "pipeline", stage: Optional[str] = None, 
                 check_existing: bool = True, force: bool = False):
    """
    Context manager for pipeline locking with automatic cleanup.
    
    Args:
        operation: Type of operation (pipeline, management, etc.)
        stage: Current stage name (optional)
        check_existing: Whether to check for existing locks
        force: Whether to force override existing locks
    
    Raises:
        PipelineLockExists: If lock exists and user doesn't override
    """
    
    # Check for existing lock unless forced
    if check_existing and not force:
        if not check_and_prompt_for_lock(operation):
            raise PipelineLockExists("User chose not to override existing lock")
    
    # Remove any existing lock if we're proceeding
    if os.path.exists(LOCK_FILE):
        remove_lock_file()
    
    # Create new lock
    try:
        create_lock_file(operation, stage)
        
        # Set up cleanup handlers
        def cleanup_handler(signum=None, frame=None):
            remove_lock_file()
            if signum:
                print(f"\n🧹 Cleaned up lock file due to signal {signum}")
                sys.exit(1)
        
        # Register cleanup for normal exit
        atexit.register(remove_lock_file)
        
        # Register cleanup for signals (Ctrl+C, etc.)
        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
        
        print(f"🔒 Pipeline lock acquired for {operation}")
        
        # Yield control back to caller
        yield
        
    finally:
        # Always clean up lock
        remove_lock_file()
        print(f"🔓 Pipeline lock released")

def main():
    """Command line interface for lock management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline lock management utility")
    parser.add_argument('--check', action='store_true', help='Check current lock status')
    parser.add_argument('--remove', action='store_true', help='Remove existing lock (use carefully)')
    parser.add_argument('--force-remove', action='store_true', help='Force remove lock without prompts')
    
    args = parser.parse_args()
    
    if args.check:
        lock_data = read_lock_file()
        if lock_data:
            print("🔒 Pipeline lock is active:")
            print(f"  Created: {lock_data.get('timestamp', 'Unknown')}")
            print(f"  User: {lock_data.get('user', 'Unknown')}@{lock_data.get('hostname', 'Unknown')}")
            print(f"  Operation: {lock_data.get('operation', 'Unknown')}")
            if 'stage' in lock_data:
                print(f"  Stage: {lock_data['stage']}")
            print(f"  Age: {format_lock_age(lock_data.get('timestamp', ''))}")
        else:
            print("🔓 No pipeline lock found")
    
    elif args.force_remove:
        if os.path.exists(LOCK_FILE):
            remove_lock_file()
            print("🗑️  Lock file forcibly removed")
        else:
            print("🔓 No lock file to remove")
    
    elif args.remove:
        lock_data = read_lock_file()
        if lock_data:
            if prompt_override_lock(lock_data, "lock removal"):
                remove_lock_file()
                print("🗑️  Lock file removed")
        else:
            print("🔓 No lock file to remove")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
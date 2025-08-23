import streamlit as st
import pandas as pd
import os
import yaml
from PIL import Image
from datetime import datetime
import sys

# Import pipeline lock utilities
try:
    from pipeline_lock import pipeline_lock, PipelineLockExists
    LOCK_SUPPORT = True
except ImportError:
    LOCK_SUPPORT = False
    st.error("⚠️ Pipeline lock module not found. Multi-user safety disabled.")

# --- Configuration Loading ---
CONFIG_FILE = 'config.yml'
IMAGE_DISPLAY_WIDTH = 300  # Width for displaying images

def load_config():
    """Loads the YAML configuration file."""
    try:
        from config_utils import load_config as load_cfg
        return load_cfg()
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        st.stop()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_manifest_data():
    """Loads the merged manifest CSV file."""
    config = load_config()
    from config_utils import get_data_root
    DATA_ROOT = get_data_root(config)
    MERGED_MANIFEST_FILE = config['paths']['merged_manifest_file'].format(data_root=DATA_ROOT)
    
    try:
        df = pd.read_csv(MERGED_MANIFEST_FILE, dtype={
            'message_id': str,
            'canonical_id': 'Int64',
            'original_canonical_id': 'Int64',
            'filename': str
        })
        return df, MERGED_MANIFEST_FILE
    except FileNotFoundError:
        st.error(f"Error: Merged manifest file not found at {MERGED_MANIFEST_FILE}")
        st.error("Please run the merge identities step first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading manifest: {e}")
        st.stop()

def get_image_dir():
    """Get the image directory path."""
    config = load_config()
    from config_utils import get_data_root
    DATA_ROOT = get_data_root(config)
    return config['paths']['dataset_dir'].format(data_root=DATA_ROOT)

def validate_canonical_id_consistency(df):
    """
    Validate that all images with the same canonical_id have the same normalized_horse_name.
    This is a critical data integrity rule for the system.
    
    Args:
        df (pandas.DataFrame): The manifest dataframe to validate
        
    Returns:
        tuple: (bool, list) - (is_valid, list_of_violations)
    """
    violations = []
    canonical_groups = df.groupby('canonical_id')
    
    for canonical_id, group in canonical_groups:
        # Check normalized_horse_name if it exists, otherwise check horse_name
        name_column = 'normalized_horse_name' if 'normalized_horse_name' in df.columns else 'horse_name'
        unique_names = group[name_column].unique()
        if len(unique_names) > 1:
            violations.append({
                'canonical_id': canonical_id,
                'names': list(unique_names),
                'count': len(group),
                'column': name_column
            })
    
    return len(violations) == 0, violations

def save_manifest_with_validation(df, manifest_file_path):
    """
    Save the manifest file after validating canonical_id consistency.
    Shows error in Streamlit if validation fails.
    
    Args:
        df (pandas.DataFrame): The dataframe to save
        manifest_file_path (str): Path to save the file
        
    Returns:
        bool: True if saved successfully, False if validation failed
    """
    is_valid, violations = validate_canonical_id_consistency(df)
    
    if not is_valid:
        st.error("❌ **Data Integrity Violation Detected!**")
        st.error("All images with the same canonical_id MUST have the same normalized_horse_name.")
        st.error("The following violations were found:")
        
        for violation in violations:
            st.error(f"- Canonical ID {violation['canonical_id']} ({violation['count']} images) has {violation['column']} values: {violation['names']}")
        
        st.error("**Cannot save changes.** Please fix these inconsistencies first.")
        return False
    
    # Validation passed, save the file
    df.to_csv(manifest_file_path, index=False)
    return True

def get_horse_list(df):
    """Get list of horses grouped by canonical_id with status info and usable image counts."""
    # Group by canonical_id and get representative info for each horse
    agg_dict = {
        'horse_name': 'first',  # Take first horse name for the canonical_id
        'status': lambda x: x.fillna('').iloc[0] if not x.fillna('').empty else '',  # Get status
        'filename': 'count'  # Count images
    }
    
    # Add normalized_horse_name if it exists
    if 'normalized_horse_name' in df.columns:
        agg_dict['normalized_horse_name'] = 'first'
    
    horse_groups = df.groupby('canonical_id').agg(agg_dict).reset_index()
    
    # Set up column names
    if 'normalized_horse_name' in df.columns:
        horse_groups.columns = ['canonical_id', 'horse_name', 'status', 'image_count', 'normalized_horse_name']
    else:
        horse_groups.columns = ['canonical_id', 'horse_name', 'status', 'image_count']
    
    # Calculate usable image counts (SINGLE detection AND not EXCLUDE status)
    usable_counts = []
    for _, row in horse_groups.iterrows():
        canonical_id = row['canonical_id']
        horse_images = df[df['canonical_id'] == canonical_id]
        
        # Count usable images: SINGLE detection AND (no status or status != 'EXCLUDE')
        usable_mask = (
            (horse_images['num_horses_detected'] == 'SINGLE') &
            ((horse_images['status'].fillna('') == '') | (horse_images['status'] != 'EXCLUDE'))
        )
        usable_count = usable_mask.sum()
        usable_counts.append(usable_count)
    
    horse_groups['usable_count'] = usable_counts
    
    # Sort by normalized name if available, otherwise by horse name
    if 'normalized_horse_name' in df.columns:
        horse_groups = horse_groups.sort_values('normalized_horse_name')
    else:
        horse_groups = horse_groups.sort_values('horse_name')
    
    return horse_groups

def get_status_display(status):
    """Get colored status display."""
    if pd.isna(status) or status == '':
        return "🟢 Active"
    elif status == 'EXCLUDE':
        return "🔴 EXCLUDE"
    else:
        return f"🟡 {status}"

def acquire_management_lock():
    """Acquire management lock for the session."""
    if not LOCK_SUPPORT:
        return True  # Continue if lock support is not available
    
    # Import additional functions needed for manual lock management
    try:
        from pipeline_lock import create_lock_file, read_lock_file, check_and_prompt_for_lock, remove_lock_file
        import atexit
        import os
        import getpass
        import socket
    except ImportError as e:
        st.error(f"Failed to import lock utilities: {e}")
        return False
    
    # Check if we already have a lock for this session
    if st.session_state.get('management_lock_acquired', False):
        # Verify our lock still exists and is ours
        lock_data = read_lock_file()
        if lock_data:
            current_user = getpass.getuser()
            current_hostname = socket.gethostname()
            current_pid = os.getpid()
            
            # Check if this is our lock
            if (lock_data.get('user') == current_user and 
                lock_data.get('hostname') == current_hostname and
                lock_data.get('operation') == 'management'):
                return True
            else:
                # Someone else's lock - need to handle this
                st.session_state.management_lock_acquired = False
        else:
            # Lock disappeared - need to reacquire
            st.session_state.management_lock_acquired = False
    
    # Try to acquire the management lock
    try:
        # Check for existing locks first
        existing_lock = read_lock_file()
        if existing_lock:
            # Show lock details and get user decision
            st.error("🔒 **Existing Lock Detected**")
            
            with st.expander("🔍 Lock Details", expanded=True):
                timestamp = existing_lock.get("timestamp", "Unknown")
                user = existing_lock.get("user", "Unknown")
                hostname = existing_lock.get("hostname", "Unknown")
                operation = existing_lock.get("operation", "Unknown")
                stage = existing_lock.get("stage")
                
                st.markdown(f"**Created:** {timestamp}")
                st.markdown(f"**User:** {user}@{hostname}")
                st.markdown(f"**Operation:** {operation}")
                if stage:
                    st.markdown(f"**Stage:** {stage}")
            
            st.warning("""
            **This indicates another operation is in progress.**
            Only continue if you're certain the other operation is not running.
            """)
            
            # Create override option
            if st.button("🚨 Override Lock & Continue", type="primary", 
                        help="Only use if you're certain the other operation is not running"):
                # Remove existing lock and proceed
                remove_lock_file()
                st.session_state.lock_overridden = True
                st.rerun()  # Rerun to continue with lock acquisition
            else:
                return False
        
        # Create the management lock
        create_lock_file(operation="management", stage="active")
        st.session_state.management_lock_acquired = True
        st.session_state.management_lock_error = None
        
        # Register cleanup function to remove lock when session ends
        # Note: This is best effort - Streamlit doesn't guarantee cleanup execution
        def cleanup_lock():
            try:
                lock_data = read_lock_file()
                if lock_data and lock_data.get('operation') == 'management':
                    remove_lock_file()
            except:
                pass  # Ignore cleanup errors
        
        atexit.register(cleanup_lock)
        
        return True
        
    except PipelineLockExists as e:
        st.session_state.management_lock_acquired = False
        st.session_state.management_lock_error = str(e)
        return False
    except Exception as e:
        st.session_state.management_lock_acquired = False
        st.session_state.management_lock_error = f"Lock error: {e}"
        return False

def main():
    st.set_page_config(
        page_title="Horse Management",
        page_icon="🐴",
        layout="wide"
    )
    
    # Acquire management lock first
    if not acquire_management_lock():
        st.stop()  # Stop execution if lock cannot be acquired
    
    st.title("🐴 Horse Management")
    st.markdown("Manage horse status and review images by canonical ID")
    
    # Show lock status
    if st.session_state.get('management_lock_acquired', False):
        st.success("🔒 **Management Lock Active** - Protected from concurrent pipeline operations")
    
    # Show lock override status if applicable
    if st.session_state.get('lock_overridden', False):
        st.warning("⚡ **Lock Override Active** - You overrode an existing lock. Use caution!")
    
    # Load data
    df, manifest_file_path = load_manifest_data()
    image_dir = get_image_dir()
    horse_list = get_horse_list(df)
    
    # Initialize session state
    if 'selected_canonical_id' not in st.session_state:
        st.session_state.selected_canonical_id = None
    if 'show_single_only' not in st.session_state:
        st.session_state.show_single_only = True
    if 'selected_images' not in st.session_state:
        st.session_state.selected_images = set()
    if 'last_canonical_id' not in st.session_state:
        st.session_state.last_canonical_id = None
    
    # Reset image selection when switching horses
    if st.session_state.selected_canonical_id != st.session_state.last_canonical_id:
        st.session_state.selected_images = set()
        st.session_state.last_canonical_id = st.session_state.selected_canonical_id
    
    # Sidebar for horse selection
    with st.sidebar:
        # Lock management section
        if LOCK_SUPPORT and st.session_state.get('management_lock_acquired', False):
            st.header("🔒 Lock Management")
            st.success("Management lock active")
            
            if st.button("🔓 Release Lock & Exit", help="Release the management lock and exit the application"):
                try:
                    from pipeline_lock import remove_lock_file, read_lock_file
                    import getpass
                    import socket
                    
                    # Verify this is our lock before removing it
                    lock_data = read_lock_file()
                    if lock_data:
                        current_user = getpass.getuser()
                        current_hostname = socket.gethostname()
                        
                        if (lock_data.get('user') == current_user and 
                            lock_data.get('hostname') == current_hostname and
                            lock_data.get('operation') == 'management'):
                            remove_lock_file()
                            st.session_state.management_lock_acquired = False
                            st.success("Lock released successfully!")
                            st.stop()
                        else:
                            st.error("Cannot release lock - not owned by this session")
                    else:
                        st.warning("No lock found to release")
                        st.session_state.management_lock_acquired = False
                except Exception as e:
                    st.error(f"Error releasing lock: {e}")
            
            st.divider()
        
        st.header("Select Horse")
        
        if horse_list.empty:
            st.warning("No horses found in manifest.")
            return
        
        # Create display options for horses
        horse_options = []
        for _, row in horse_list.iterrows():
            status_display = get_status_display(row['status'])
            # Use normalized_horse_name if available, otherwise fall back to horse_name
            display_name = row.get('normalized_horse_name', row['horse_name'])
            
            # Add indicator for horses with no usable images
            usable_count = row['usable_count']
            total_count = row['image_count']
            
            if usable_count == 0:
                # Add red circle with line through it for no usable images
                option = f"🚫 {display_name} (ID: {row['canonical_id']}) - {usable_count}/{total_count} usable"
            else:
                option = f"{display_name} (ID: {row['canonical_id']}) - {usable_count}/{total_count} usable"
            
            horse_options.append((option, row['canonical_id'], status_display))
        
        # Display horses with status indicators
        selected_option = st.radio(
            "Choose a horse:",
            options=[opt[0] for opt in horse_options],
            format_func=lambda x: x
        )
        
        # Update selected canonical_id
        if selected_option:
            selected_canonical_id = next(opt[1] for opt in horse_options if opt[0] == selected_option)
            st.session_state.selected_canonical_id = selected_canonical_id
            
            # Show status for selected horse
            selected_status = next(opt[2] for opt in horse_options if opt[0] == selected_option)
            st.markdown(f"**Current Status:** {selected_status}")
    
    # Main content area
    if st.session_state.selected_canonical_id is None:
        st.info("👈 Please select a horse from the sidebar to view images and manage status.")
        return
    
    # Get selected horse data
    selected_canonical_id = st.session_state.selected_canonical_id
    horse_images = df[df['canonical_id'] == selected_canonical_id].copy()
    
    if horse_images.empty:
        st.error("No images found for selected horse.")
        return
    
    horse_name = horse_images['horse_name'].iloc[0]
    # Use normalized name for display if available
    display_name = horse_images.get('normalized_horse_name', horse_images['horse_name']).iloc[0] if 'normalized_horse_name' in horse_images.columns else horse_name
    current_status = horse_images['status'].fillna('').iloc[0]
    
    st.header(f"Managing: {display_name} (Canonical ID: {selected_canonical_id})")
    
    # Filtering controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        show_single_only = st.checkbox(
            "Show only SINGLE horse detections",
            value=st.session_state.show_single_only,
            key="show_single_checkbox"
        )
        st.session_state.show_single_only = show_single_only
    
    with col2:
        if show_single_only:
            filtered_count = len(horse_images[horse_images['num_horses_detected'] == 'SINGLE'])
            total_count = len(horse_images)
            st.info(f"Showing {filtered_count} of {total_count} images (SINGLE detections only)")
    
    # Filter images if needed
    display_images = horse_images.copy()
    if show_single_only:
        display_images = display_images[display_images['num_horses_detected'] == 'SINGLE']
    
    # Image Selection Controls
    st.subheader("Image Selection")
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
    
    with col1:
        if st.button("Select All", key="select_all"):
            st.session_state.selected_images = set(display_images['filename'].tolist())
            st.rerun()
    
    with col2:
        if st.button("Clear All", key="clear_all"):
            st.session_state.selected_images = set()
            st.rerun()
    
    with col3:
        if st.button("Invert Selection", key="invert_selection"):
            current_displayed = set(display_images['filename'].tolist())
            st.session_state.selected_images = current_displayed - st.session_state.selected_images
            st.rerun()
    
    with col4:
        selected_count = len(st.session_state.selected_images & set(display_images['filename'].tolist()))
        total_displayed = len(display_images)
        st.info(f"📋 {selected_count} of {total_displayed} images selected")

    # Management interface
    st.subheader("Image Management")
    
    # Create tabs for different operations
    tab1, tab2, tab3, tab4 = st.tabs(["Status Management", "Canonical ID Assignment", "Horse Name Management", "Detection Management"])
    
    with tab1:
        # Status display and selector
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**Current Status:** {get_status_display(current_status)}")
        
        with col2:
            new_status = st.selectbox(
                "Set new status:",
                options=["", "EXCLUDE", "REVIEW"],
                format_func=lambda x: "Active (no status)" if x == "" else x,
                index=0 if current_status == "" else (["", "EXCLUDE", "REVIEW"].index(current_status) if current_status in ["", "EXCLUDE", "REVIEW"] else 0)
            )
        
        # Status action buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Apply Status to All Images", type="secondary", use_container_width=True):
                # Update status for all images with this canonical_id
                try:
                    # Read fresh data to avoid conflicts
                    fresh_df = pd.read_csv(manifest_file_path, dtype={
                        'message_id': str,
                        'canonical_id': 'Int64',
                        'original_canonical_id': 'Int64',
                        'filename': str
                    })
                    
                    # Update status for matching canonical_id
                    mask = fresh_df['canonical_id'] == selected_canonical_id
                    affected_count = mask.sum()
                    
                    fresh_df.loc[mask, 'status'] = new_status
                    
                    # Save back to file with validation
                    if save_manifest_with_validation(fresh_df, manifest_file_path):
                        # Clear cache to reflect changes
                        st.cache_data.clear()
                    else:
                        return  # Stop execution if validation fails
                    
                    st.success(f"✅ Updated status to '{new_status if new_status else 'Active'}' for {affected_count} images")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error updating status: {e}")
        
        with col2:
            selected_displayed_images = st.session_state.selected_images & set(display_images['filename'].tolist())
            button_disabled = len(selected_displayed_images) == 0
            
            if st.button("Apply Status to Selected Images", type="primary", disabled=button_disabled, use_container_width=True):
                # Update status for selected images only
                try:
                    # Read fresh data to avoid conflicts
                    fresh_df = pd.read_csv(manifest_file_path, dtype={
                        'message_id': str,
                        'canonical_id': 'Int64',
                        'original_canonical_id': 'Int64',
                        'filename': str
                    })
                    
                    # Update status for selected images
                    mask = fresh_df['filename'].isin(selected_displayed_images)
                    affected_count = mask.sum()
                    
                    fresh_df.loc[mask, 'status'] = new_status
                    
                    # Save back to file with validation
                    if save_manifest_with_validation(fresh_df, manifest_file_path):
                        # Clear cache to reflect changes
                        st.cache_data.clear()
                    else:
                        return  # Stop execution if validation fails
                    
                    # Clear selection after successful update
                    st.session_state.selected_images = set()
                    
                    st.success(f"✅ Updated status to '{new_status if new_status else 'Active'}' for {affected_count} selected images")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error updating status: {e}")
    
    with tab2:
        # Canonical ID assignment
        st.markdown("**Reassign images to a different canonical ID**")
        st.caption("⚠️ This will move selected images to a different horse identity and update their normalized_horse_name to match. Use carefully!")
        
        # Current canonical ID info
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**Current Canonical ID:** {selected_canonical_id}")
            st.markdown(f"**Current Horse:** {horse_name}")
        
        with col2:
            # Text input for target canonical ID
            target_canonical_id_input = st.text_input(
                "Target Canonical ID:",
                help="Enter the canonical ID to assign selected images to"
            )
            
            # Validate and convert input
            target_canonical_id = None
            target_horse_name = None
            input_error = None
            
            if target_canonical_id_input.strip():
                try:
                    target_canonical_id = int(target_canonical_id_input.strip())
                    
                    # Check if this canonical ID exists
                    target_horse_rows = df[df['canonical_id'] == target_canonical_id]
                    if not target_horse_rows.empty:
                        target_horse_name = target_horse_rows['horse_name'].iloc[0]
                        if target_canonical_id != selected_canonical_id:
                            st.markdown(f"**Target Horse:** {target_horse_name}")
                    else:
                        input_error = f"Canonical ID {target_canonical_id} does not exist"
                        st.error(input_error)
                        target_canonical_id = None
                        
                except ValueError:
                    input_error = "Please enter a valid number"
                    st.error(input_error)
                    target_canonical_id = None
        
        # Canonical ID action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            reassign_all_disabled = bool(target_canonical_id is None or target_canonical_id == selected_canonical_id)
            if st.button("Reassign All Images", type="secondary", disabled=reassign_all_disabled, use_container_width=True):
                try:
                    # Read fresh data to avoid conflicts
                    fresh_df = pd.read_csv(manifest_file_path, dtype={
                        'message_id': str,
                        'canonical_id': 'Int64',
                        'original_canonical_id': 'Int64',
                        'filename': str
                    })
                    
                    # Get target normalized_horse_name from the target canonical_id
                    target_rows = fresh_df[fresh_df['canonical_id'] == target_canonical_id]
                    if not target_rows.empty:
                        if 'normalized_horse_name' in fresh_df.columns:
                            target_normalized_name = target_rows['normalized_horse_name'].iloc[0]
                        else:
                            target_normalized_name = target_rows['horse_name'].iloc[0]
                    else:
                        st.error(f"No rows found for target canonical ID {target_canonical_id}")
                        return
                    
                    # Update canonical_id for matching canonical_id
                    mask = fresh_df['canonical_id'] == selected_canonical_id
                    affected_count = mask.sum()
                    
                    fresh_df.loc[mask, 'canonical_id'] = target_canonical_id
                    
                    # Update normalized_horse_name to match target canonical_id
                    if 'normalized_horse_name' in fresh_df.columns:
                        fresh_df.loc[mask, 'normalized_horse_name'] = target_normalized_name
                        updated_field = 'normalized_horse_name'
                    else:
                        fresh_df.loc[mask, 'horse_name'] = target_normalized_name
                        updated_field = 'horse_name'
                    
                    # Save back to file with validation
                    if save_manifest_with_validation(fresh_df, manifest_file_path):
                        # Clear cache to reflect changes
                        st.cache_data.clear()
                    else:
                        return  # Stop execution if validation fails
                    
                    st.success(f"✅ Reassigned {affected_count} images to canonical ID {target_canonical_id} and updated {updated_field} to '{target_normalized_name}'")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error reassigning canonical ID: {e}")
        
        with col2:
            selected_displayed_images = st.session_state.selected_images & set(display_images['filename'].tolist())
            reassign_selected_disabled = bool(len(selected_displayed_images) == 0 or target_canonical_id is None or target_canonical_id == selected_canonical_id)
            
            if st.button("Reassign Selected Images", type="primary", disabled=reassign_selected_disabled, use_container_width=True):
                try:
                    # Read fresh data to avoid conflicts
                    fresh_df = pd.read_csv(manifest_file_path, dtype={
                        'message_id': str,
                        'canonical_id': 'Int64',
                        'original_canonical_id': 'Int64',
                        'filename': str
                    })
                    
                    # Get target normalized_horse_name from the target canonical_id
                    target_rows = fresh_df[fresh_df['canonical_id'] == target_canonical_id]
                    if not target_rows.empty:
                        if 'normalized_horse_name' in fresh_df.columns:
                            target_normalized_name = target_rows['normalized_horse_name'].iloc[0]
                        else:
                            target_normalized_name = target_rows['horse_name'].iloc[0]
                    else:
                        st.error(f"No rows found for target canonical ID {target_canonical_id}")
                        return
                    
                    # Update selected images
                    mask = fresh_df['filename'].isin(selected_displayed_images)
                    affected_count = mask.sum()
                    
                    fresh_df.loc[mask, 'canonical_id'] = target_canonical_id
                    
                    # Update normalized_horse_name to match target canonical_id
                    if 'normalized_horse_name' in fresh_df.columns:
                        fresh_df.loc[mask, 'normalized_horse_name'] = target_normalized_name
                        updated_field = 'normalized_horse_name'
                    else:
                        fresh_df.loc[mask, 'horse_name'] = target_normalized_name
                        updated_field = 'horse_name'
                    
                    # Save back to file with validation
                    if save_manifest_with_validation(fresh_df, manifest_file_path):
                        # Clear cache to reflect changes
                        st.cache_data.clear()
                    else:
                        return  # Stop execution if validation fails
                    
                    # Clear selection after successful update
                    st.session_state.selected_images = set()
                    
                    st.success(f"✅ Reassigned {affected_count} selected images to canonical ID {target_canonical_id} and updated {updated_field} to '{target_normalized_name}'")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error reassigning canonical ID: {e}")
        
        with col3:
            selected_displayed_images = st.session_state.selected_images & set(display_images['filename'].tolist())
            create_new_disabled = bool(len(selected_displayed_images) == 0)
            
            if st.button("Create New Canonical ID", type="primary", disabled=create_new_disabled, use_container_width=True):
                try:
                    # Read fresh data to avoid conflicts
                    fresh_df = pd.read_csv(manifest_file_path, dtype={
                        'message_id': str,
                        'canonical_id': 'Int64',
                        'original_canonical_id': 'Int64',
                        'filename': str
                    })
                    
                    # Create new canonical_id (use max + 1)
                    new_canonical_id = fresh_df['canonical_id'].max() + 1
                    
                    # Update selected images with new canonical_id
                    mask = fresh_df['filename'].isin(selected_displayed_images)
                    affected_count = mask.sum()
                    
                    fresh_df.loc[mask, 'canonical_id'] = new_canonical_id
                    
                    # Keep the existing normalized_horse_name (or horse_name) unchanged
                    # This allows the user to keep the same name with a new canonical_id
                    
                    # Save back to file with validation
                    if save_manifest_with_validation(fresh_df, manifest_file_path):
                        # Clear cache to reflect changes
                        st.cache_data.clear()
                    else:
                        return  # Stop execution if validation fails
                    
                    # Clear selection after successful update
                    st.session_state.selected_images = set()
                    
                    st.success(f"✅ Created new canonical ID {new_canonical_id} for {affected_count} selected images")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating new canonical ID: {e}")
    
    with tab3:
        # Horse name management
        st.markdown("**Update normalized horse name for all images with this canonical ID**")
        st.caption("⚠️ This changes the horse name for ALL images with the current canonical ID. The canonical ID will not change.")
        
        # Current name info
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Get current normalized horse name (or horse_name if normalized doesn't exist)
            current_normalized_name = horse_images.get('normalized_horse_name', horse_images['horse_name']).iloc[0] if 'normalized_horse_name' in horse_images.columns else horse_name
            
            st.markdown(f"**Current Horse Name:** {horse_name}")
            if 'normalized_horse_name' in horse_images.columns:
                st.markdown(f"**Current Normalized Name:** {current_normalized_name}")
            else:
                st.info("No normalized_horse_name field found. Will update horse_name field.")
        
        with col2:
            # Text input for new normalized name
            new_normalized_name = st.text_input(
                "New normalized horse name:",
                value=current_normalized_name if 'normalized_horse_name' in horse_images.columns else horse_name,
                help="Enter the standardized name for this horse"
            )
            
            # Show preview of change
            if new_normalized_name != current_normalized_name:
                st.markdown(f"**Preview:** {current_normalized_name} → {new_normalized_name}")
        
        # Horse name action buttons - only one button now
        name_change_disabled = bool(new_normalized_name == current_normalized_name or not new_normalized_name.strip())
        
        if st.button("Update All Images with this Canonical ID", type="primary", disabled=name_change_disabled, use_container_width=True):
                try:
                    # Read fresh data to avoid conflicts
                    fresh_df = pd.read_csv(manifest_file_path, dtype={
                        'message_id': str,
                        'canonical_id': 'Int64',
                        'original_canonical_id': 'Int64',
                        'filename': str
                    })
                    
                    # Update normalized_horse_name (or horse_name if normalized doesn't exist)
                    mask = fresh_df['canonical_id'] == selected_canonical_id
                    affected_count = mask.sum()
                    
                    target_column = 'normalized_horse_name' if 'normalized_horse_name' in fresh_df.columns else 'horse_name'
                    fresh_df.loc[mask, target_column] = new_normalized_name.strip()
                    
                    # Save back to file with validation
                    if save_manifest_with_validation(fresh_df, manifest_file_path):
                        # Clear cache to reflect changes
                        st.cache_data.clear()
                    else:
                        return  # Stop execution if validation fails
                    
                    st.success(f"✅ Updated {target_column} to '{new_normalized_name.strip()}' for {affected_count} images")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error updating horse name: {e}")
    
    with tab4:
        # Detection management
        st.markdown("**Update horse detection status for images**")
        st.caption("⚠️ This overrides the automatic detection results. Use carefully to correct obvious detection errors.")
        
        # Current detection info
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Get current detection status for the canonical ID
            detection_counts = horse_images['num_horses_detected'].value_counts()
            st.markdown("**Current Detection Status:**")
            for detection, count in detection_counts.items():
                st.markdown(f"- {detection}: {count} images")
        
        with col2:
            # Detection selector
            new_detection = st.selectbox(
                "Set new detection status:",
                options=["NONE", "SINGLE", "MULTIPLE"],
                help="NONE = No horses detected, SINGLE = One horse detected, MULTIPLE = Multiple horses detected"
            )
            
            st.markdown(f"**Will set to:** {new_detection}")
        
        # Detection action buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Apply Detection to All Images", type="secondary", use_container_width=True):
                # Update detection for all images with this canonical_id
                try:
                    # Read fresh data to avoid conflicts
                    fresh_df = pd.read_csv(manifest_file_path, dtype={
                        'message_id': str,
                        'canonical_id': 'Int64',
                        'original_canonical_id': 'Int64',
                        'filename': str
                    })
                    
                    # Update detection for matching canonical_id
                    mask = fresh_df['canonical_id'] == selected_canonical_id
                    affected_count = mask.sum()
                    
                    fresh_df.loc[mask, 'num_horses_detected'] = new_detection
                    
                    # Save back to file with validation
                    if save_manifest_with_validation(fresh_df, manifest_file_path):
                        # Clear cache to reflect changes
                        st.cache_data.clear()
                    else:
                        return  # Stop execution if validation fails
                    
                    st.success(f"✅ Updated detection to '{new_detection}' for {affected_count} images")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error updating detection: {e}")
        
        with col2:
            selected_displayed_images = st.session_state.selected_images & set(display_images['filename'].tolist())
            button_disabled = len(selected_displayed_images) == 0
            
            if st.button("Apply Detection to Selected Images", type="primary", disabled=button_disabled, use_container_width=True):
                # Update detection for selected images only
                try:
                    # Read fresh data to avoid conflicts
                    fresh_df = pd.read_csv(manifest_file_path, dtype={
                        'message_id': str,
                        'canonical_id': 'Int64',
                        'original_canonical_id': 'Int64',
                        'filename': str
                    })
                    
                    # Update detection for selected images
                    mask = fresh_df['filename'].isin(selected_displayed_images)
                    affected_count = mask.sum()
                    
                    fresh_df.loc[mask, 'num_horses_detected'] = new_detection
                    
                    # Save back to file with validation
                    if save_manifest_with_validation(fresh_df, manifest_file_path):
                        # Clear cache to reflect changes
                        st.cache_data.clear()
                    else:
                        return  # Stop execution if validation fails
                    
                    # Clear selection after successful update
                    st.session_state.selected_images = set()
                    
                    st.success(f"✅ Updated detection to '{new_detection}' for {affected_count} selected images")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error updating detection: {e}")
    
    # Display image gallery
    st.subheader(f"Images ({len(display_images)} shown)")
    
    if display_images.empty:
        st.warning("No images match the current filter criteria.")
        return
    
    # Create image grid (3 columns)
    cols_per_row = 3
    for i in range(0, len(display_images), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(display_images):
                row = display_images.iloc[idx]
                image_path = os.path.join(image_dir, row['filename'])
                filename = row['filename']
                is_selected = filename in st.session_state.selected_images
                
                with col:
                    # Checkbox for image selection
                    checkbox_key = f"img_select_{filename}"
                    selected = st.checkbox(
                        "Select",
                        value=is_selected,
                        key=checkbox_key
                    )
                    
                    # Update selection state
                    if selected and filename not in st.session_state.selected_images:
                        st.session_state.selected_images.add(filename)
                    elif not selected and filename in st.session_state.selected_images:
                        st.session_state.selected_images.remove(filename)
                    
                    try:
                        if os.path.exists(image_path):
                            image = Image.open(image_path)
                            st.image(image, width=IMAGE_DISPLAY_WIDTH, caption=row['filename'])
                        else:
                            st.error(f"Image not found: {row['filename']}")
                            
                        # Show image info
                        detection_status = row.get('num_horses_detected', 'Unknown')
                        current_img_status = row.get('status', '')
                        st.caption(f"Detection: {detection_status}")
                        st.caption(f"Status: {get_status_display(current_img_status)}")
                        st.caption(f"Date: {row.get('email_date', 'Unknown')}")
                        
                    except Exception as e:
                        st.error(f"Error loading {row['filename']}: {e}")

if __name__ == "__main__":
    main()
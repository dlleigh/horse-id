import streamlit as st
import pandas as pd
import os
import yaml
from PIL import Image
from datetime import datetime

# --- Configuration Loading ---
CONFIG_FILE = 'config.yml'
IMAGE_DISPLAY_WIDTH = 300  # Width for displaying images

def load_config():
    """Loads the YAML configuration file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{CONFIG_FILE}' not found. Please ensure it exists in the same directory as the app.")
        st.stop()
    return config

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_manifest_data():
    """Loads the merged manifest CSV file."""
    config = load_config()
    DATA_ROOT = os.path.expanduser(config['paths']['data_root'])
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
    DATA_ROOT = os.path.expanduser(config['paths']['data_root'])
    return config['paths']['dataset_dir'].format(data_root=DATA_ROOT)

def get_horse_list(df):
    """Get list of horses grouped by canonical_id with status info."""
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
        # Sort by normalized name if available
        horse_groups = horse_groups.sort_values('normalized_horse_name')
    else:
        horse_groups.columns = ['canonical_id', 'horse_name', 'status', 'image_count']
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

def main():
    st.set_page_config(
        page_title="Horse Management",
        page_icon="🐴",
        layout="wide"
    )
    
    st.title("🐴 Horse Management")
    st.markdown("Manage horse status and review images by canonical ID")
    
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
            option = f"{display_name} (ID: {row['canonical_id']}) - {row['image_count']} images"
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
    tab1, tab2, tab3 = st.tabs(["Status Management", "Canonical ID Assignment", "Horse Name Management"])
    
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
                    
                    # Save back to file
                    fresh_df.to_csv(manifest_file_path, index=False)
                    
                    # Clear cache to reflect changes
                    st.cache_data.clear()
                    
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
                    
                    # Save back to file
                    fresh_df.to_csv(manifest_file_path, index=False)
                    
                    # Clear cache to reflect changes
                    st.cache_data.clear()
                    
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
                    
                    # Save back to file
                    fresh_df.to_csv(manifest_file_path, index=False)
                    
                    # Clear cache to reflect changes
                    st.cache_data.clear()
                    
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
                    
                    # Save back to file
                    fresh_df.to_csv(manifest_file_path, index=False)
                    
                    # Clear cache to reflect changes
                    st.cache_data.clear()
                    
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
                    
                    # Save back to file
                    fresh_df.to_csv(manifest_file_path, index=False)
                    
                    # Clear cache to reflect changes
                    st.cache_data.clear()
                    
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
                    
                    # Save back to file
                    fresh_df.to_csv(manifest_file_path, index=False)
                    
                    # Clear cache to reflect changes
                    st.cache_data.clear()
                    
                    st.success(f"✅ Updated {target_column} to '{new_normalized_name.strip()}' for {affected_count} images")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error updating horse name: {e}")
    
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
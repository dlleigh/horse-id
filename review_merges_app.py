import streamlit as st
import pandas as pd
import os
import yaml
from datetime import datetime

# --- Configuration Loading ---
CONFIG_FILE = 'config.yml'
IMAGE_DISPLAY_WIDTH = 400 # Width for displaying images in Streamlit

def load_config():
    """Loads the YAML configuration file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Error: Configuration file '{CONFIG_FILE}' not found. Please ensure it exists in the same directory as the app.")
        st.stop()
    return config

config = load_config()
DATA_ROOT = os.path.expanduser(config['paths']['data_root'])
IMAGE_DIR = config['paths']['dataset_dir'].format(data_root=DATA_ROOT)
MERGED_MANIFEST_FILE = config['paths']['merged_manifest_file'].format(data_root=DATA_ROOT)
MERGE_RESULTS_FILE = config['paths']['merge_results_file'].format(data_root=DATA_ROOT) # Updated filename

# --- Data Loading Functions ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_manifest_data(manifest_path):
    """Loads the merged manifest CSV file."""
    try:
        df = pd.read_csv(manifest_path, dtype={
            'message_id': str,
            'canonical_id': 'Int64',
            'original_canonical_id': 'Int64',
            'filename': str
        })
        if 'last_merged_timestamp' in df.columns:
            df['last_merged_timestamp'] = pd.to_datetime(df['last_merged_timestamp'], errors='coerce')
        else:
            df['last_merged_timestamp'] = pd.NaT
        return df
    except FileNotFoundError:
        st.error(f"Error: Manifest file not found at {manifest_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading manifest: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_merge_results_data(results_path):
    """Loads the merge results CSV file."""
    try:
        df = pd.read_csv(results_path, dtype={'message_id_a': str, 'message_id_b': str, 'canonical_id_a': str, 'canonical_id_b': str})
        # Sort by timestamp to review in order of processing, or add other sorting as needed
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: Merge results file not found at {results_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading merge results: {e}")
        return pd.DataFrame()

def save_manifest_data(df, manifest_path):
    """Saves the manifest DataFrame to CSV."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        df.to_csv(manifest_path, index=False)
        st.success(f"Manifest saved to {manifest_path}")
    except Exception as e:
        st.error(f"Error saving manifest: {e}")

def get_images_and_info_for_message_id(manifest_df, msg_id):
    """Retrieves image paths and canonical ID info for a given message_id."""
    if manifest_df.empty or msg_id is None:
        return [], None, None, None
    
    msg_df = manifest_df[manifest_df['message_id'] == str(msg_id)]
    if msg_df.empty:
        return [], None, None, None
        
    image_paths = [os.path.join(IMAGE_DIR, fname) for fname in msg_df['filename'].tolist()]
    # All images from the same message should share these IDs
    current_cid = msg_df['canonical_id'].iloc[0] if not msg_df['canonical_id'].empty else None
    original_cid = msg_df['original_canonical_id'].iloc[0] if not msg_df['original_canonical_id'].empty else None
    horse_name = msg_df['horse_name'].iloc[0] if not msg_df['horse_name'].empty else "N/A"
    return image_paths, current_cid, original_cid, horse_name

# --- Streamlit App UI and Logic ---
st.set_page_config(layout="wide")
st.title("Horse Identity Merge Review Tool")

# Custom CSS to increase monospace font size
st.markdown("""
<style>
# code, pre {
#     font-size: 1.1em !important; /* Adjust the size as needed */
# }
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State ---
if 'manifest_df' not in st.session_state:
    st.session_state.manifest_df = load_manifest_data(MERGED_MANIFEST_FILE)
if 'merge_results_df' not in st.session_state: # Renamed session state variable
    st.session_state.merge_results_df = load_merge_results_data(MERGE_RESULTS_FILE)
if 'current_review_idx' not in st.session_state:
    st.session_state.current_review_idx = 0
if 'feedback_message' not in st.session_state:
    st.session_state.feedback_message = ""


# --- Main Application ---
if st.session_state.manifest_df.empty:
    st.warning("Manifest data is empty. Please ensure the merged manifest file exists and is populated.")
    st.stop()
if st.session_state.merge_results_df.empty:
    st.warning("Merge results file is empty. No pairs to review.")
    st.stop()

# --- Sidebar for Navigation ---
total_pairs = len(st.session_state.merge_results_df)
if total_pairs == 0:
    st.info("No pairs in the merge results file to review.")
    st.stop()

st.sidebar.header("Merge Navigation")

# --- Search and Sequential Navigation ---
search_term = st.sidebar.text_input("Search by Horse Name or Message ID", "")

nav_cols = st.sidebar.columns(2)
if nav_cols[0].button("⬅️ Previous", use_container_width=True) and st.session_state.current_review_idx > 0:
    st.session_state.current_review_idx -= 1
    st.session_state.feedback_message = ""
    st.rerun()

if nav_cols[1].button("Next ➡️", use_container_width=True) and st.session_state.current_review_idx < total_pairs - 1:
    st.session_state.current_review_idx += 1
    st.session_state.feedback_message = ""
    st.rerun()

st.sidebar.write(f"Reviewing Pair: **{st.session_state.current_review_idx + 1} / {total_pairs}**")
st.sidebar.divider()

# --- Filterable Jump-to List ---
if search_term:
    search_term_lower = search_term.lower()
    filtered_df = st.session_state.merge_results_df[
        st.session_state.merge_results_df['horse_name'].str.lower().str.contains(search_term_lower, na=False) |
        st.session_state.merge_results_df['message_id_a'].str.lower().str.contains(search_term_lower, na=False) |
        st.session_state.merge_results_df['message_id_b'].str.lower().str.contains(search_term_lower, na=False)
    ]
else:
    filtered_df = st.session_state.merge_results_df

with st.sidebar.container(height=500):
    for idx, row in filtered_df.iterrows():
        msg_id_a = str(row['message_id_a'])
        msg_id_b = str(row['message_id_b'])
        _, cid_a, _, name_a = get_images_and_info_for_message_id(st.session_state.manifest_df, msg_id_a)
        _, cid_b, _, _ = get_images_and_info_for_message_id(st.session_state.manifest_df, msg_id_b)

        is_merged = (cid_a is not None and cid_a == cid_b)
        status_emoji = "✅" if is_merged else "❌"
        button_label = f"{status_emoji} {name_a}: {msg_id_a[:8]}... vs {msg_id_b[:8]}..."

        if st.button(button_label, key=f"jump_{idx}", use_container_width=True):
            st.session_state.current_review_idx = idx
            st.session_state.feedback_message = f"Jumped to pair {idx + 1}."
            st.rerun()

# --- Main Page Content ---
if st.session_state.current_review_idx > 0 and st.session_state.current_review_idx >= len(st.session_state.merge_results_df):
        st.session_state.current_review_idx -= 1

# Display feedback message
if st.session_state.feedback_message:
    st.info(st.session_state.feedback_message)
    st.session_state.feedback_message = "" # Clear after displaying

# --- Get current pair from similarity log ---
# Ensure index is within bounds after potential filtering/actions
st.session_state.current_review_idx = max(0, min(st.session_state.current_review_idx, total_pairs - 1))

current_pair = st.session_state.merge_results_df.iloc[st.session_state.current_review_idx]

msg_id_a_log = str(current_pair['message_id_a'])
msg_id_b_log = str(current_pair['message_id_b'])

images_a, current_cid_a, original_cid_a, horse_name_a = get_images_and_info_for_message_id(st.session_state.manifest_df, msg_id_a_log)
images_b, current_cid_b, original_cid_b, horse_name_b = get_images_and_info_for_message_id(st.session_state.manifest_df, msg_id_b_log)

log_cid_a = current_pair.get('canonical_id_a', 'N/A') # ID at time of logging
log_cid_b = current_pair.get('canonical_id_b', 'N/A') # ID at time of logging
similarity_score = current_pair.get('max_similarity', 'N/A') # Changed from final_similarity
predicted_match_val = current_pair.get('is_match', 'N/A')
current_match_val = current_cid_a == current_cid_b

# Determine color and text for predicted_match
if str(predicted_match_val).lower() == 'true':
    predicted_match_display = f'<span style="color: green;">True</span>'
elif str(predicted_match_val).lower() == 'false':
    predicted_match_display = f'<span style="color: red;">False</span>'
else:
    predicted_match_display = str(predicted_match_val) # Display as is if not True/False

# Determine color and text for current_match
if current_match_val:
    current_match_display = f'<span style="color: green;">True</span>'
else:
    current_match_display = f'<span style="color: red;">False</span>'

st.markdown("---")
st.subheader(f"Comparison Details (from Merge Results)")
details_cols = st.columns(2)
details_cols[0].markdown(f"""
- **Message ID A:** `{msg_id_a_log}`
- **Message ID B:** `{msg_id_b_log}`
- **Logged Canonical ID A:** `{log_cid_a}`
- **Logged Canonical ID B:** `{log_cid_b}`
""")
details_cols[1].markdown(f"""
- **Horse Name:** {horse_name_a}
- **Currently Merged?** {current_match_display}
- **System Predicted Match:** {predicted_match_display}
- **Similarity Score:** `{similarity_score}`
""", unsafe_allow_html=True)
st.markdown("---")


# --- Display Images and Current Manifest Info ---

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Message A: `{msg_id_a_log}`")
    st.write(f"**Current Canonical ID:** `{current_cid_a if current_cid_a is not None else 'N/A'}`")
    st.write(f"**Original Canonical ID:** `{original_cid_a if original_cid_a is not None else 'N/A'}`")
    if not images_a:
        st.warning("No images found in manifest for this Message ID.")
    for img_path in images_a:
        if os.path.exists(img_path):
            st.image(img_path, width=IMAGE_DISPLAY_WIDTH)
        else:
            st.error(f"Image not found: {img_path}")

with col2:
    st.subheader(f"Message B: `{msg_id_b_log}`")
    st.write(f"**Current Canonical ID:** `{current_cid_b if current_cid_b is not None else 'N/A'}`")
    st.write(f"**Original Canonical ID:** `{original_cid_b if original_cid_b is not None else 'N/A'}`")
    if not images_b:
        st.warning("No images found in manifest for this Message ID.")
    for img_path in images_b:
        if os.path.exists(img_path):
            st.image(img_path, width=IMAGE_DISPLAY_WIDTH)
        else:
            st.error(f"Image not found: {img_path}")

st.markdown("---")
st.subheader("Actions")

# --- Action Buttons ---
action_col1, action_col2, action_col3 = st.columns(3)

with action_col1:
    if st.button("Merge (Mark as SAME Horse)", type="primary", use_container_width=True):
        if current_cid_a is None or current_cid_b is None:
            st.session_state.feedback_message = "Error: Cannot merge. One or both message IDs not found in current manifest."
        elif current_cid_a == current_cid_b:
            st.session_state.feedback_message = f"Already merged under Canonical ID: {current_cid_a}. No action taken."
        else:
            target_id = min(current_cid_a, current_cid_b)
            id_to_change = max(current_cid_a, current_cid_b)
            
            # Update all rows that currently have id_to_change
            rows_to_update_mask = st.session_state.manifest_df['canonical_id'] == id_to_change
            st.session_state.manifest_df.loc[rows_to_update_mask, 'canonical_id'] = target_id
            st.session_state.manifest_df.loc[rows_to_update_mask, 'last_merged_timestamp'] = pd.to_datetime(datetime.now())
            
            save_manifest_data(st.session_state.manifest_df, MERGED_MANIFEST_FILE)
            st.session_state.feedback_message = f"Merged. Rows with Canonical ID {id_to_change} are now {target_id}."
            # Move to next pair after action
            if st.session_state.current_review_idx < total_pairs - 1:
                st.session_state.current_review_idx +=1
        st.rerun()

with action_col2:
    if st.button("Un-merge (Mark as DIFFERENT Horses)", type="secondary", use_container_width=True):
        manifest_changed = False
        # Revert Message A images to original_canonical_id
        msg_a_mask = st.session_state.manifest_df['message_id'] == msg_id_a_log
        if msg_a_mask.any():
            original_ids_a = st.session_state.manifest_df.loc[msg_a_mask, 'original_canonical_id']
            st.session_state.manifest_df.loc[msg_a_mask, 'canonical_id'] = original_ids_a
            st.session_state.manifest_df.loc[msg_a_mask, 'last_merged_timestamp'] = pd.NaT
            manifest_changed = True
            
        # Revert Message B images to original_canonical_id
        msg_b_mask = st.session_state.manifest_df['message_id'] == msg_id_b_log
        if msg_b_mask.any():
            original_ids_b = st.session_state.manifest_df.loc[msg_b_mask, 'original_canonical_id']
            st.session_state.manifest_df.loc[msg_b_mask, 'canonical_id'] = original_ids_b
            st.session_state.manifest_df.loc[msg_b_mask, 'last_merged_timestamp'] = pd.NaT
            manifest_changed = True

        if manifest_changed:
            save_manifest_data(st.session_state.manifest_df, MERGED_MANIFEST_FILE)
            st.session_state.feedback_message = f"Un-merged. Images for Message ID {msg_id_a_log} and {msg_id_b_log} reverted to their original canonical IDs."
            # Move to next pair after action
            if st.session_state.current_review_idx < total_pairs - 1:
                st.session_state.current_review_idx +=1
        else:
            st.session_state.feedback_message = "No changes made. Message IDs might not be in manifest or already reverted."
        st.rerun()

with action_col3:
    if st.button("Skip / No Change", use_container_width=True):
        st.session_state.feedback_message = "Skipped. No changes made."
        if st.session_state.current_review_idx < total_pairs - 1:
            st.session_state.current_review_idx +=1
        st.rerun()

st.markdown("---")
if st.checkbox("Show current manifest data (first 100 rows)"):
    st.dataframe(st.session_state.manifest_df.head(100))

if st.checkbox("Show merge results data (first 100 rows)"):
    st.dataframe(st.session_state.merge_results_df.head(100))

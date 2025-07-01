import pandas as pd
import os
import yaml
import html # Used for escaping HTML characters
import json # For safely handling segmentation data

# --- Load Configuration ---
try:
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found. Please ensure the configuration file exists.")
    exit()

# --- Use the config values ---
DATA_ROOT = os.path.expanduser(config['paths']['data_root'])
IMAGE_DIR = config['paths']['dataset_dir'].format(data_root=DATA_ROOT)
# Use the final merged manifest as the data source
# MANIFEST_FILE = config['detection']['detected_manifest_file'].format(data_root=DATA_ROOT) # Will be dynamic
GALLERY_BASE_FILENAME = "horse_gallery"
GALLERY_OUTPUT_DIR = DATA_ROOT # Assumes DATA_ROOT is a directory like '.../data'


def create_html_gallery(df, output_path, manifest_display_name, current_manifest_key, all_manifest_options):
    """Generates a self-contained static HTML gallery from the manifest dataframe."""
    
    merged_horse_names = set()
    if 'canonical_id' in df.columns and 'original_canonical_id' in df.columns:
        # Ensure IDs are comparable, handling potential NaNs or different types
        df_copy = df[['horse_name', 'canonical_id', 'original_canonical_id']].copy()
        df_copy['canonical_id'] = pd.to_numeric(df_copy['canonical_id'], errors='coerce')
        df_copy['original_canonical_id'] = pd.to_numeric(df_copy['original_canonical_id'], errors='coerce')
        
        merged_rows = df_copy[df_copy['canonical_id'] != df_copy['original_canonical_id']]
        merged_horse_names.update(merged_rows['horse_name'].unique())
    else:
        print("Warning: 'canonical_id' or 'original_canonical_id' column not found. 'Show Merged' filter may not work.")

    unmerged_multi_id_horse_names = set()
    if 'canonical_id' in df.columns and 'horse_name' in df.columns:
        for name, group in df.groupby('horse_name'):
            if group['canonical_id'].nunique() > 1:
                unmerged_multi_id_horse_names.add(name)
    else:
        print("Warning: 'canonical_id' or 'horse_name' column not found. 'Show Unmerged (Multiple IDs)' filter may not work.")

    # Start building the HTML string
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Horse Gallery - {manifest_display_name}</title>
        <style>
            body { font-family: sans-serif; margin: 0; background-color: #f4f4f4; }
            .header { background-color: #333; color: white; padding: 15px; text-align: center; }
            .controls { padding: 20px; text-align: center; background-color: #fff; border-bottom: 1px solid #ddd;}
            .controls label { font-weight: bold; margin-right: 10px; }
            .manifest-nav {{ padding: 10px 20px; background-color: #e9e9e9; text-align: center; border-bottom: 1px solid #ccc; font-size: 0.9em; }}
            #horse-filter, #detection-filter, #size-ratio-filter { padding: 8px; border-radius: 4px; border: 1px solid #ccc; margin-left: 5px;}
            .gallery-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                padding: 20px;
            }
            .gallery-item {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #fff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow: hidden;
                transition: transform 0.2s;
            }
            .gallery-item:hover { transform: scale(1.03); }
            .gallery-item img {
                width: 100%;
                height: 200px;
                object-fit: cover;
                display: block;
            }
            .caption { padding: 10px; font-size: 0.9em; }
            .caption p { margin: 5px 0; }
            .caption .label { font-weight: bold; color: #555; }
            
            /* Modal styles */
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.9);
                overflow: auto; /* display: flex; will be set by JS when modal opens */
                /* display: flex; */ /* Removed to keep modal hidden by default */
                align-items: center;
                justify-content: center;
            }
            .modal-image-wrapper {
                position: relative; /* For positioning the canvas */
                display: flex; /* To center the image if it's smaller than the container */
                align-items: center;
                justify-content: center;
                flex: 1;
                background: #000;
            }
            .modal-container {
                display: flex;
                max-width: 90%;
                max-height: 90vh;
                background: #fff;
                border-radius: 8px;
                overflow: hidden;
            }
            .modal-image {
                max-height: 90vh;
                max-width: 100%;
                object-fit: contain;
            }
            #modalCanvas {
                position: absolute;
                top: 0;
                left: 0;
                pointer-events: none; /* Allows clicking through the canvas to the image if needed */
            }
            .modal-metadata {
                width: 300px;
                padding: 20px;
                background: #fff;
                overflow-y: auto;
            }
            .modal-metadata h3 {
                margin-top: 0;
                margin-bottom: 15px;
                color: #333;
            }
            .modal-metadata p {
                margin: 8px 0;
                font-size: 14px;
            }
            .modal-close {
                position: absolute;
                right: 20px;
                top: 10px;
                color: #fff;
                font-size: 40px;
                font-weight: bold;
                cursor: pointer;
                z-index: 1001;
            }
        </style>
    </head>
    <body>

    <div class="header">
        <h1>Horse Image Gallery</h1>
    </div>

    <div class="manifest-nav">
        <span>Switch Manifest View:</span>
"""
    for key, options in all_manifest_options.items():
        if key == current_manifest_key:
            html_content += f' <strong style="margin: 0 5px;">{options["display"]} (Current)</strong> |'
        else:
            html_content += f' <a href="{GALLERY_BASE_FILENAME}_{key}.html" style="margin: 0 5px;">{options["display"]}</a> |'
    html_content = html_content.rstrip('|') # Remove last pipe
    html_content += """
    </div>
    <div class="controls">
        <label for="horse-filter">Filter by Horse Name:</label>
        <select id="horse-filter" onchange="filterGallery()">
            <option value="all">All Horses</option>
    """

    # --- Generate Filter Dropdown ---
    unique_names = sorted(df['horse_name'].unique())
    for name in unique_names:
        safe_name = html.escape(name)
        html_content += f'<option value="{safe_name}">{safe_name}</option>'
    html_content += '</select>'

    # Add detection filter
    html_content += '''
        <label for="detection-filter" style="margin-left: 20px;">Filter by Detection:</label>
        <select id="detection-filter" onchange="filterGallery()">
            <option value="all">All Detections</option>
            <option value="SINGLE">Single Horse</option>
            <option value="MULTIPLE">Multiple Horses</option>
            <option value="NONE">No Detection</option>
        </select>

        <span style="margin-left: 20px;">
            <input type="checkbox" id="merged-filter" onchange="filterGallery()">
            <label for="merged-filter">Show Merged Horses</label>
        </span>

        <span style="margin-left: 20px;">
            <input type="checkbox" id="unmerged-multi-id-filter" onchange="filterGallery()">
            <label for="unmerged-multi-id-filter">Show Unmerged (Multiple IDs)</label>
        </span>

        <span style="margin-left: 20px;">
            <label for="size-ratio-filter">Min Size Ratio:</label>
            <input type="number" id="size-ratio-filter" step="0.1" min="0" placeholder="e.g., 1.5" oninput="filterGallery()">
        </span>
    </div>
    '''

    # --- Generate Image Gallery ---
    html_content += '<div class="gallery-container" id="gallery">'
    for _, row in df.iterrows():
        # Ensure path is relative from where the HTML file will be to the image
        try:
            image_path = os.path.join(IMAGE_DIR, row['filename'])
            relative_image_path = os.path.relpath(image_path, os.path.dirname(output_path))
        except Exception:
            # Fallback for path errors
            relative_image_path = ""
            
        horse_name_safe = html.escape(row['horse_name'])
        filename_safe = html.escape(row['filename'])
        detection_status = row.get('num_horses_detected', 'N/A')
        size_ratio_val = row.get('size_ratio', 'N/A')
        original_canonical_id_val = row.get('original_canonical_id', 'N/A')
        is_merged_flag = str(row['horse_name'] in merged_horse_names).lower()
        has_multiple_canonical_ids_flag = str(row['horse_name'] in unmerged_multi_id_horse_names).lower()

        # --- Date Formatting ---
        try:
            # email_date is stored as YYYYMMDD, format it to YYYY-MM-DD
            email_date_formatted = pd.to_datetime(str(row.get('email_date')), format='%Y%m%d').strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            email_date_formatted = 'N/A'

        # Additional metadata for display
        message_id_val = row.get('message_id', 'N/A')
        original_filename_safe = html.escape(str(row.get('original_filename', 'N/A')))
        date_added_val = row.get('date_added', 'N/A')
        last_merged_formatted = pd.to_datetime(row.get('last_merged_timestamp')).strftime('%Y-%m-%d') if pd.notna(row.get('last_merged_timestamp')) else 'N/A'
        status_val = html.escape(str(row.get('status', ''))) # Default to empty string if missing

        # --- Get BBox and Mask data ---
        bbox_data = [row.get('bbox_x'), row.get('bbox_y'), row.get('bbox_width'), row.get('bbox_height')]
        bbox_json = html.escape(json.dumps(bbox_data)) if all(pd.notna(x) for x in bbox_data) else ''
        segmentation_mask_val = html.escape(str(row.get('segmentation_mask', '')))

        # Update the gallery-item div content in the main loop
        html_content += f"""
        <div class="gallery-item" 
            data-horse-name="{horse_name_safe}" 
            data-detection="{detection_status}"
            data-canonical-id="{row['canonical_id']}"
            data-filename="{filename_safe}"
            data-email-date="{email_date_formatted}"
            data-size-ratio="{size_ratio_val}"
            data-original-canonical-id="{original_canonical_id_val}"
            data-is-merged="{is_merged_flag}"
            data-has-multiple-canonical-ids="{has_multiple_canonical_ids_flag}"
            data-message-id="{message_id_val}"
            data-original-filename="{original_filename_safe}"
            data-date-added="{date_added_val}"
            data-last-merged="{last_merged_formatted}"
            data-status="{status_val}"
            data-bbox='{bbox_json}'
            data-segmentation-mask='{segmentation_mask_val}'
            onclick="openModal(this)">
            <img src="{relative_image_path}" alt="{filename_safe}" loading="lazy">
            <div class="caption">
                <p><span class="label">Name:</span> {horse_name_safe}</p>
                <p><span class="label">ID:</span> {row['canonical_id']}</p>
                <p><span class="label">Orig. ID:</span> {original_canonical_id_val if pd.notna(original_canonical_id_val) else 'N/A'}</p>
                <p><span class="label">Detection:</span> {detection_status}</p>
                <p><span class="label">Email Date:</span> {email_date_formatted}</p>
                <p><span class="label">Size Ratio:</span> {size_ratio_val if pd.notna(size_ratio_val) else 'N/A'}</p>
            </div>
        </div>
        """
    html_content += '</div>'

    # --- Add JavaScript for Filtering ---
    html_content += """
    <script>
        // Capture initial horse options as soon as the script block starts and the DOM element is available
        const initialHorseOptionsHTML = document.getElementById('horse-filter') ? document.getElementById('horse-filter').innerHTML : '';

        function drawOverlay(canvas, image, bboxData, maskDataStr) {
            const ctx = canvas.getContext('2d');
            const scaleX = image.width / image.naturalWidth;
            const scaleY = image.height / image.naturalHeight;

            // Set canvas size to match the displayed image size
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw Bounding Box
            if (bboxData && bboxData.length === 4) {
                const [x_norm, y_norm, w_norm, h_norm] = bboxData;
                const x = x_norm * image.naturalWidth * scaleX;
                const y = y_norm * image.naturalHeight * scaleY;
                const w = w_norm * image.naturalWidth * scaleX;
                const h = h_norm * image.naturalHeight * scaleY;
                
                ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)'; // Red
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, w, h);
            }

            // Draw Segmentation Mask
            if (maskDataStr) { // Check for non-empty string
                try {
                    // The mask string is a custom format: "x1 y1;x2 y2;..."
                    // It represents a single polygon.
                    const points = maskDataStr.split(';').map(p => {
                        const coords = p.split(' ');
                        return [parseFloat(coords[0]), parseFloat(coords[1])];
                    });
                    // The drawing logic expects a list of polygons.
                    const polygons = [points];
                    ctx.fillStyle = 'rgba(0, 255, 0, 0.4)'; // Green with transparency
                    ctx.strokeStyle = 'rgba(0, 200, 0, 0.9)';
                    ctx.lineWidth = 1;

                    polygons.forEach(polygon => {
                        ctx.beginPath();
                        polygon.forEach((point, index) => {
                            const x = point[0] * scaleX;
                            const y = point[1] * scaleY;
                            if (index === 0) {
                                ctx.moveTo(x, y);
                            } else {
                                ctx.lineTo(x, y);
                            }
                        });
                        ctx.closePath();
                        ctx.fill();
                        ctx.stroke();
                    });
                } catch (e) {
                    console.error("Error parsing segmentation mask:", e, maskDataStr);
                }
            }
        }

        function openModal(element) {
            const modal = document.getElementById("imageModal");
            const modalImg = document.getElementById("modalImage");
            const metadataDiv = document.getElementById("modalMetadataContent");
            const canvas = document.getElementById("modalCanvas");
            const ctx = canvas.getContext('2d');
            
            // Set image
            const img = element.querySelector('img');
            modalImg.src = img.src;
            
            // Set metadata
            const metadata = `
                <p><strong>Horse Name:</strong> ${element.getAttribute('data-horse-name')}</p>
                <p><strong>Canonical ID:</strong> ${element.getAttribute('data-canonical-id')}</p>
                <p><strong>Original Canonical ID:</strong> ${element.getAttribute('data-original-canonical-id') !== 'nan' ? element.getAttribute('data-original-canonical-id') : 'N/A'}</p>                
                <hr>
                <p><strong>Filename:</strong> ${element.getAttribute('data-filename')}</p>
                <p><strong>Original Filename:</strong> ${element.getAttribute('data-original-filename')}</p>
                <p><strong>Message ID:</strong> ${element.getAttribute('data-message-id')}</p>
                <hr>
                <p><strong>Email Date:</strong> ${element.getAttribute('data-email-date')}</p>
                <p><strong>Date Added:</strong> ${element.getAttribute('data-date-added')}</p>
                <p><strong>Last Merged:</strong> ${element.getAttribute('data-last-merged')}</p>
                <hr>
                <p><strong>Detection Status:</strong> ${element.getAttribute('data-detection')}</p>
                <p><strong>Size Ratio:</strong> ${element.getAttribute('data-size-ratio') !== 'nan' ? parseFloat(element.getAttribute('data-size-ratio')).toFixed(2) : 'N/A'}</p>
                <p><strong>Status:</strong> ${element.getAttribute('data-status')}</p>
            `;
            metadataDiv.innerHTML = metadata;
            
            // Clear previous drawings and draw new overlay once image is loaded
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            modalImg.onload = function() {
                const bboxAttr = element.getAttribute('data-bbox');
                const maskAttr = element.getAttribute('data-segmentation-mask');
                let bboxData = null;
                if (bboxAttr) {
                    try {
                        bboxData = JSON.parse(bboxAttr);
                    } catch (e) {
                        console.error("Error parsing bbox data:", e, bboxAttr);
                    }
                }
                drawOverlay(canvas, modalImg, bboxData, maskAttr);
            };
            // If image is already cached, onload might not fire, so call it directly
            if (modalImg.complete) {
                modalImg.onload();
            }

            modal.style.display = "flex";
        }
        
        function closeModal() {
            document.getElementById("imageModal").style.display = "none";
        }

        // Close modal when clicking outside the image
        document.addEventListener('click', function(event) {
            const modal = document.getElementById("imageModal");
            if (event.target === modal) {
                closeModal();
            }
        });

        // Close modal with escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === "Escape") {
                closeModal();
            }
        });

        function filterGallery() {
            const horseFilterSelect = document.getElementById('horse-filter');
            let currentHorseFilterValue = horseFilterSelect.value; // Preserve current selection before modifying options

            const sizeRatioFilterInput = document.getElementById('size-ratio-filter');
            const detectionFilter = document.getElementById('detection-filter').value;
            const mergedFilterActive = document.getElementById('merged-filter').checked;
            const unmergedMultiIdFilterActive = document.getElementById('unmerged-multi-id-filter').checked;
            const galleryItems = document.querySelectorAll('.gallery-item');

            // --- Update horse name dropdown based on checkbox ---
            if (mergedFilterActive || unmergedMultiIdFilterActive) {
                const activeHorseNames = new Set();
                galleryItems.forEach(item => {
                    const isMergedItem = item.getAttribute('data-is-merged') === 'true';
                    const hasMultipleCanonicalIdsItem = item.getAttribute('data-has-multiple-canonical-ids') === 'true';
                    let includeInDropdown = false;

                    if (mergedFilterActive && unmergedMultiIdFilterActive) {
                        includeInDropdown = isMergedItem || hasMultipleCanonicalIdsItem;
                    } else if (mergedFilterActive) {
                        includeInDropdown = isMergedItem;
                    } else if (unmergedMultiIdFilterActive) {
                        includeInDropdown = hasMultipleCanonicalIdsItem;
                    }
                    if (includeInDropdown) {
                        activeHorseNames.add(item.getAttribute('data-horse-name'));
                    }
                });

                // Rebuild options
                let newOptionsHTML = '<option value="all">All Horses</option>';
                const sortedNames = Array.from(activeHorseNames).sort();
                sortedNames.forEach(name => {
                    // Names from data-horse-name are already HTML-escaped by Python
                    newOptionsHTML += `<option value="${name}">${name}</option>`;
                });
                horseFilterSelect.innerHTML = newOptionsHTML;

                // Try to restore selection or default to 'all'
                if (activeHorseNames.has(currentHorseFilterValue) || currentHorseFilterValue === 'all') {
                    horseFilterSelect.value = currentHorseFilterValue;
                } else {
                    horseFilterSelect.value = 'all';
                }
            } else {
                // Restore all horse names from the initial capture
                horseFilterSelect.innerHTML = initialHorseOptionsHTML;
                // Try to restore selection. Check if the value still exists in the restored options.
                let valueStillExists = false;
                for (let i = 0; i < horseFilterSelect.options.length; i++) {
                    if (horseFilterSelect.options[i].value === currentHorseFilterValue) {
                        valueStillExists = true;
                        break;
                    }
                }
                horseFilterSelect.value = valueStillExists ? currentHorseFilterValue : 'all';
            }
            // --- End of horse name dropdown update ---

            const horseFilter = horseFilterSelect.value; // Read the final, potentially updated, value for filtering
            const sizeRatioFilterValue = parseFloat(sizeRatioFilterInput.value);

            galleryItems.forEach(item => {
                const horseName = item.getAttribute('data-horse-name');
                const detection = item.getAttribute('data-detection');
                const itemSizeRatioStr = item.getAttribute('data-size-ratio');
                const itemSizeRatio = parseFloat(itemSizeRatioStr);
                const isMergedItem = item.getAttribute('data-is-merged') === 'true';
                const hasMultipleCanonicalIdsItem = item.getAttribute('data-has-multiple-canonical-ids') === 'true';

                const matchesHorse = horseFilter === 'all' || horseName === horseFilter;

                const matchesMerged = !mergedFilterActive || (mergedFilterActive && isMergedItem);
                const matchesUnmergedMultiId = !unmergedMultiIdFilterActive || (unmergedMultiIdFilterActive && hasMultipleCanonicalIdsItem);

                let matchesDetection;
                if (detectionFilter === 'all') {
                    matchesDetection = true;
                } else if (detectionFilter === 'NONE') {
                    matchesDetection = (detection === 'NONE' || detection === '' || detection === 'N/A' || detection.toLowerCase() === 'nan' || detection === '<NA>');
                } else {
                    matchesDetection = detection === detectionFilter;
                }

                const matchesSizeRatio = isNaN(sizeRatioFilterValue) || (itemSizeRatioStr !== 'N/A' && itemSizeRatioStr.toLowerCase() !== 'nan' && !isNaN(itemSizeRatio) && itemSizeRatio >= sizeRatioFilterValue);

                item.style.display = (matchesHorse && matchesDetection && matchesSizeRatio && matchesMerged && matchesUnmergedMultiId) ? 'block' : 'none';
            });
        }
    </script>
    """

    # Add this before the closing </body> tag
    html_content += """
    <div id="imageModal" class="modal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <div class="modal-container">
            <div class="modal-image-wrapper">
                <img class="modal-image" id="modalImage">
                <canvas id="modalCanvas"></canvas>
            </div>
            <div class="modal-metadata" id="modalMetadata">
                <h3>Image Details</h3>
                <div id="modalMetadataContent"></div>
            </div>
        </div>
    </div>
    """

    # --- Write to File ---
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Successfully generated gallery for '{manifest_display_name}' at: {output_path}")

def main():
    """Main function to generate the gallery."""

    manifest_options = {
        "base": {
            "display": "Base (Initial Import)",
            "path_template": config['paths']['manifest_file']
        },
        "detected": {
            "display": "Detected (After YOLO)",
            "path_template": config['detection']['detected_manifest_file']
        },
        "merged": {
            "display": "Merged (After Similarity)",
            "path_template": config['paths']['merged_manifest_file']
        }
    }

    os.makedirs(GALLERY_OUTPUT_DIR, exist_ok=True)

    for key, options in manifest_options.items():
        manifest_file_path = options["path_template"].format(data_root=DATA_ROOT)
        output_html_path = os.path.join(GALLERY_OUTPUT_DIR, f"{GALLERY_BASE_FILENAME}_{key}.html")

        print(f"\nProcessing manifest for: {options['display']}")
        print(f"Reading manifest from: {manifest_file_path}")
        try:
            df = pd.read_csv(manifest_file_path)
        except FileNotFoundError:
            print(f"Error: Manifest file '{manifest_file_path}' not found. Skipping this gallery.")
            continue

        # Sort the dataframe for a consistent gallery order
        df_sorted = df.sort_values(by=['horse_name', 'canonical_id', 'filename']).reset_index(drop=True)
        create_html_gallery(df_sorted, output_html_path, options["display"], key, manifest_options)


if __name__ == "__main__":
    main()
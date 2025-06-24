import streamlit as st
import os
import subprocess
import threading
import http.server
import socketserver
import time
import json
import urllib.parse
import urllib.request # New import for url2pathname
import functools
import mimetypes
import queue
import requests

# --- Configuration ---
CONTAINER_NAME = "horse-id-lambda-container"
IMAGE_NAME = "horse-id-lambda-image"
HOST_PORT = 8080  # Port on the host to map to the container's 8080
IMAGE_SERVER_PORT = 8001 # Port for our local image hosting server
# Special DNS name that resolves to the host machine from within the container
# For Docker Desktop: 'host.docker.internal'
# For Podman on Linux: 'host.containers.internal'
CONTAINER_HOST_DNS = "host.containers.internal" 

# --- Simple HTTP Server to host the image ---
class ImageServer(threading.Thread):
    """A simple HTTP server that runs in a background thread."""
    def __init__(self, directory, port):
        super().__init__()
        self.directory = directory
        self.port = port
        self.httpd = None
        self.daemon = True # A daemon thread will not prevent the app from exiting

    def run(self):
        # Create a handler that serves files from the specified directory,
        # avoiding a process-wide os.chdir().
        # This requires Python 3.7+
        handler_class = functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=self.directory
        )
        self.httpd = socketserver.TCPServer(("", self.port), handler_class)
        self.httpd.serve_forever()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            print("Image server stopped.")

# --- Container Log Streamer ---
class ContainerLogStreamer(threading.Thread):
    """A thread to stream logs from a Podman container."""
    def __init__(self, container_name, log_queue, stop_event):
        super().__init__()
        self.container_name = container_name
        self.log_queue = log_queue
        self.stop_event = stop_event
        self.process = None

    def run(self):
        try:
            # Use 'podman logs -f' to stream logs in real-time
            self.process = subprocess.Popen(
                ["podman", "logs", "-f", self.container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr into stdout
                text=True, # Decode as text
                bufsize=1, # Line-buffered
                universal_newlines=True # For cross-platform line endings
            )
            for line in iter(self.process.stdout.readline, ''):
                if self.stop_event.is_set():
                    break
                self.log_queue.put(line)
            self.process.stdout.close()
            self.process.wait()
        except Exception as e:
            self.log_queue.put(f"Log streamer error: {e}\n")
        self.stop()

    def stop(self):
        """Signals the thread to stop and terminates the subprocess."""
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()

# --- Main Streamlit App ---
st.set_page_config(layout="wide")
st.title("Horse ID Lambda Test App")

st.write("""
This app provides an interactive way to test the `horse-id-lambda-image`.
1.  **Enter the path** to an image of a horse on your local machine.
2.  The app will **host this image** on a temporary local web server.
3.  Click **"Identify Horse"** to:
    - Start the container.
    - Send a simulated Twilio webhook request to the container.
    - The request will point to the URL of the image hosted by this app.
4.  The **response** from the Lambda function will be displayed below.
""")

image_path_input = st.text_input(
    "Enter the full path to a horse image file (e.g., /path/to/image.jpg or file:///path/to/image.jpg)",
    placeholder="/path/to/your/horse_image.jpg or file:///path/to/your/horse_image.jpg"
)

# Resolve the input path, handling file:/// URIs
resolved_image_path = None
if image_path_input: # Only process if input is not empty
    if image_path_input.startswith('file:///'):
        try:
            parsed_url = urllib.parse.urlparse(image_path_input)
            # url2pathname correctly handles platform-specific path conversions (e.g., Windows drive letters)
            resolved_image_path = urllib.request.url2pathname(parsed_url.path)
            # On Windows, url2pathname might leave a leading slash for drive letters (e.g., /C:/...). Remove it.
            if os.name == 'nt' and len(resolved_image_path) > 2 and resolved_image_path[0] == '/' and resolved_image_path[2] == ':':
                resolved_image_path = resolved_image_path[1:]
        except Exception as e:
            st.error(f"Error parsing file:/// URI: {e}")
            resolved_image_path = None # Ensure it's None if parsing fails
    else:
        resolved_image_path = image_path_input

if resolved_image_path: # Only proceed if a path was successfully resolved
    if os.path.isfile(resolved_image_path):
        st.image(resolved_image_path, caption="Selected Image", width=500)
    else:
        st.warning(f"File path '{resolved_image_path}' is not valid or does not point to a file. Please check the path.")
        resolved_image_path = None # Clear if invalid


# Initialize session state for logs and streamer management
if 'container_logs' not in st.session_state:
    st.session_state.container_logs = []
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = queue.Queue()

# Placeholder for logs
log_placeholder = st.empty()

# Function to update logs in Streamlit UI from the queue
def update_logs():
    """Reads new log lines from the queue and updates the Streamlit UI."""
    while not st.session_state.log_queue.empty():
        st.session_state.container_logs.append(st.session_state.log_queue.get())
    log_placeholder.code("".join(st.session_state.container_logs), language='bash')

if st.button("Identify Horse"):
    if not resolved_image_path: # Use the resolved path for validation
        st.error("Please provide a valid path to an image file first.")
    else:
        # Reset logs for new run
        st.session_state.container_logs = []
        update_logs()

        # 1. Get image details and start the image server
        image_dir = os.path.dirname(resolved_image_path) # Use resolved path
        image_filename = os.path.basename(resolved_image_path) # Use resolved path

        image_server = ImageServer(directory=image_dir, port=IMAGE_SERVER_PORT)
        image_server.start()

        st.info(f"Image server started. Serving from directory: {image_dir}")

        result_area = st.empty()

        log_streamer_thread = None
        log_streamer_stop_event = threading.Event()
        try:
            # 2. Start the Lambda container
            st.info("Starting Lambda container...")
            aws_profile = os.environ.get("AWS_PROFILE", "")
            if not aws_profile:
                st.warning("AWS_PROFILE environment variable not set. AWS commands might fail if not configured otherwise.")

            start_cmd = [
                "podman", "run", "-d", "--rm",
                "-p", f"{HOST_PORT}:8080",
                "-v", f"{os.path.expanduser('~')}/.aws:/root/.aws",
                "-e", "HORSE_ID_DATA_ROOT=/data",
                "-e", f"AWS_PROFILE={aws_profile}",
                "--name", CONTAINER_NAME,
                IMAGE_NAME
            ]
            container_id = subprocess.check_output(start_cmd).decode('utf-8').strip()
            st.success(f"Container '{CONTAINER_NAME}' started with ID: {container_id[:12]}")

            # Start log streamer
            log_streamer_thread = ContainerLogStreamer(
                CONTAINER_NAME, st.session_state.log_queue, log_streamer_stop_event
            )
            log_streamer_thread.start()
            st.info("Log streaming started...")

            # Give container and log streamer a moment to start and produce initial logs
            time.sleep(2)
            update_logs()

            # 3. Construct and send the invocation request
            with st.spinner("Waiting for container to initialize and invoking Lambda function..."):
                time.sleep(3) # Additional wait for runtime inside container

            image_url = f"http://{CONTAINER_HOST_DNS}:{IMAGE_SERVER_PORT}/{urllib.parse.quote(image_filename)}"
            st.info(f"Container will fetch image from: {image_url}")

            # Guess the content type of the image file
            content_type, _ = mimetypes.guess_type(resolved_image_path) # Use resolved path
            if content_type is None:
                content_type = 'application/octet-stream' # A generic default

            mock_twilio_data = urllib.parse.urlencode({
                "SmsMessageSid": "SM_streamlit_test_sid", "AccountSid": "AC_streamlit_test_sid",
                "From": "+15555555555", "To": "+15555555556", "Body": "Image from Streamlit test app.",
                "NumMedia": "1", "MediaUrl0": image_url, "MediaContentType0": content_type
            })

            lambda_payload = {
                "body": mock_twilio_data, "headers": {"Content-Type": "application/x-www-form-urlencoded"},
                "httpMethod": "POST", "isBase64Encoded": False, "path": "/webhook"
            }

            response = requests.post(f"http://localhost:{HOST_PORT}/2015-03-31/functions/function/invocations", json=lambda_payload, timeout=90)
            update_logs() # Update logs after invocation

            result_area.subheader("Lambda Function Response")
            if response.json().get('headers', {}).get('Content-Type') == 'text/xml':
                result_area.code(response.json()['body'], language='xml')
            else:
                result_area.json(response.json())
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.container_logs.append(f"App error: {e}\n")
            update_logs()
        finally:
            st.info(f"Stopping container '{CONTAINER_NAME}'...")
            subprocess.run(["podman", "stop", CONTAINER_NAME], capture_output=True)
            st.success("Container stopped.")
            
            if log_streamer_thread:
                log_streamer_stop_event.set()
                log_streamer_thread.join(timeout=5)
                if log_streamer_thread.is_alive():
                    st.warning("Log streamer thread did not terminate gracefully.")
            
            image_server.stop()
            # Final log update to catch any remaining logs
            time.sleep(1)
            update_logs()
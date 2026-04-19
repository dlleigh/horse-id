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
from twilio.request_validator import RequestValidator

# --- Configuration ---
PROCESSOR_IMAGE_NAME = "horse-id-processor-image" # Renamed for clarity
RESPONDER_IMAGE_NAME = "horse-id-responder-image"
RESPONDER_CONTAINER_NAME = "horse-id-responder-container"
PROCESSOR_CONTAINER_NAME = "horse-id-processor-container"
RESPONDER_HOST_PORT = 8080
PROCESSOR_HOST_PORT = 8081
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
    def __init__(self, container_name, log_queue, stop_event, prefix=""):
        super().__init__()
        self.container_name = container_name
        self.log_queue = log_queue
        self.stop_event = stop_event
        self.process = None
        self.prefix = prefix

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
                self.log_queue.put(self.prefix + line)
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
This app tests the asynchronous two-lambda flow locally.
1.  **Enter the path** to an image of a horse on your local machine.
2.  The app will **host this image** on a temporary local web server.
3.  Click **"Identify Horse"** to simulate the full asynchronous flow:
    - Start two containers: one for the `responder` and one for the `processor`.
    - Invoke the `responder`, which returns an immediate confirmation.
    - The app then invokes the `processor` with the same data.
4.  Logs from both containers are streamed below, showing the entire process.
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
        st.image(resolved_image_path, caption="Selected Image", width=300)
    else:
        st.warning(f"File path '{resolved_image_path}' is not valid or does not point to a file. Please check the path.")
        resolved_image_path = None # Clear if invalid

twilio_from_number = st.text_input(
    "Twilio 'From' Phone Number (e.g., +1234567890)",
    value="+15555555555" # Default value
)

twilio_to_number = st.text_input(
    "Twilio 'To' Phone Number (e.g., +1987654321 - your Twilio number)",
    value="+15555555556" # Default value
)

# Get Twilio Auth Token from environment variable
# This is used for local testing to generate the X-Twilio-Signature
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
if not TWILIO_AUTH_TOKEN:
    st.warning("TWILIO_AUTH_TOKEN environment variable not set. Twilio request validation might fail.")


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

def build_image(image_name, dockerfile_path, context_path="."):
    """Helper function to build a Docker image."""
    st.info(f"Building image '{image_name}' from '{dockerfile_path}'...")
    try:
        build_cmd = ["podman", "build", "-t", image_name, "-f", dockerfile_path, context_path]
        process = subprocess.Popen(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with st.expander(f"Build logs for {image_name}", expanded=False):
            log_output = ""
            for line in iter(process.stdout.readline, ''):
                log_output += line
            st.code(log_output, language='bash')
        process.wait()
        if process.returncode != 0:
            st.error(f"Failed to build image '{image_name}'. Check logs above.")
            return False
        st.success(f"Successfully built image '{image_name}'.")
        return True
    except Exception as e:
        st.error(f"An error occurred while building '{image_name}': {e}")
        return False

if st.button("Identify Horse"):
    if not resolved_image_path: # Use the resolved path for validation
        st.error("Please provide a valid path to an image file first.")
    elif not TWILIO_AUTH_TOKEN:
        st.error("TWILIO_AUTH_TOKEN environment variable is not set. Cannot generate a valid Twilio signature for testing.")
    else:
        # Reset logs for new run
        st.session_state.container_logs = []
        update_logs()

        # Build both images first
        # The main Dockerfile is for the processor.
        if not build_image(RESPONDER_IMAGE_NAME, "Dockerfile.responder"):
            st.stop()
        if not build_image(PROCESSOR_IMAGE_NAME, "Dockerfile.horse_id"):
            st.stop()

        # 1. Get image details and start the image server
        image_dir = os.path.dirname(resolved_image_path) # Use resolved path
        image_filename = os.path.basename(resolved_image_path) # Use resolved path

        image_server = ImageServer(directory=image_dir, port=IMAGE_SERVER_PORT)
        image_server.start()

        st.info(f"Image server started. Serving from directory: {image_dir}")

        result_area = st.empty()

        responder_log_streamer = None
        processor_log_streamer = None
        responder_stop_event = threading.Event()
        processor_stop_event = threading.Event()

        try:
            # 2. Start both Lambda containers
            st.info("Starting Responder and Processor containers...")
            aws_profile = os.environ.get("AWS_PROFILE", "")
            twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID", "AC_streamlit_test_sid")
            # Use the token from the input field
            twilio_token = TWILIO_AUTH_TOKEN 
            if not aws_profile:
                st.warning("AWS_PROFILE environment variable not set. AWS commands might fail if not configured otherwise.")

            # Start Processor Container
            processor_start_cmd = [
                "podman", "run", "-d", "--rm",
                "-p", f"{PROCESSOR_HOST_PORT}:8080",
                "--memory", "6g", # Allocate more memory for the processor
                "-v", f"{os.path.expanduser('~')}/.aws:/root/.aws",
                "-e", "HORSE_ID_DATA_ROOT=/data",
                "-e", f"AWS_PROFILE={aws_profile}",
                "-e", f"TWILIO_ACCOUNT_SID={twilio_sid}",
                "-e", f"TWILIO_AUTH_TOKEN={twilio_token}",
                "--name", PROCESSOR_CONTAINER_NAME,
                PROCESSOR_IMAGE_NAME, # Use the processor image
                "horse_id.horse_id_processor_handler" # Explicitly set handler
            ]
            subprocess.check_call(processor_start_cmd)
            st.success(f"Container '{PROCESSOR_CONTAINER_NAME}' started on port {PROCESSOR_HOST_PORT}.")

            # Start Responder Container
            responder_start_cmd = [
                "podman", "run", "-d", "--rm",
                "-p", f"{RESPONDER_HOST_PORT}:8080",
                "-e", f"TWILIO_AUTH_TOKEN={twilio_token}", # Pass auth token for validation
                "-e", "PROCESSOR_LAMBDA_NAME=horse-id-processor-local-test",
                "--name", RESPONDER_CONTAINER_NAME,
                RESPONDER_IMAGE_NAME, # Use the responder image
                "webhook_responder.webhook_handler"
            ]
            subprocess.check_call(responder_start_cmd)
            st.success(f"Container '{RESPONDER_CONTAINER_NAME}' started on port {RESPONDER_HOST_PORT}.")

            # Start log streamers for both
            responder_log_streamer = ContainerLogStreamer(
                RESPONDER_CONTAINER_NAME, st.session_state.log_queue, responder_stop_event, prefix="[RESPONDER] "
            )
            processor_log_streamer = ContainerLogStreamer(
                PROCESSOR_CONTAINER_NAME, st.session_state.log_queue, processor_stop_event, prefix="[PROCESSOR] "
            )
            responder_log_streamer.start()
            processor_log_streamer.start()
            st.info("Log streaming started for both containers...")
            time.sleep(2)
            update_logs()

            # 3. Construct the payload
            image_url = f"http://{CONTAINER_HOST_DNS}:{IMAGE_SERVER_PORT}/{urllib.parse.quote(image_filename)}"
            content_type, _ = mimetypes.guess_type(resolved_image_path)
            content_type = content_type or 'application/octet-stream'

            mock_twilio_params = {
                "SmsMessageSid": "SM_streamlit_test_sid", "AccountSid": "AC_streamlit_test_sid",
                "From": twilio_from_number, "To": twilio_to_number, "Body": "Image from Streamlit test app.",
                "NumMedia": "1", "MediaUrl0": image_url, "MediaContentType0": content_type
            }
            mock_twilio_data = urllib.parse.urlencode(mock_twilio_params)

            # --- Generate a valid Twilio signature ---
            validator = RequestValidator(twilio_token)
            # The URL must match what the responder will reconstruct from the event.
            mock_host = "streamlit-test.com"
            mock_path = "/webhook"
            mock_url = f"https://{mock_host}{mock_path}"
            signature = validator.compute_signature(mock_url, mock_twilio_params)
            st.info(f"Generated Twilio Signature: {signature}")

            lambda_payload = {
                "body": mock_twilio_data,
                "headers": {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "host": mock_host,
                    "x-twilio-signature": signature
                },
                "httpMethod": "POST", "isBase64Encoded": False, "rawPath": mock_path, "rawQueryString": ""
            }

            # 4. Invoke Responder
            with st.spinner("Step 1: Invoking Responder..."):
                st.info(f"Sending request to Responder at http://localhost:{RESPONDER_HOST_PORT}")
                responder_response = requests.post(f"http://localhost:{RESPONDER_HOST_PORT}/2015-03-31/functions/function/invocations", json=lambda_payload, timeout=30)
                update_logs()
                result_area.subheader("Responder Response")
                result_area.code(responder_response.json().get('body', 'No body in response'), language='xml')

            # 5. Invoke Processor (simulating the async call)
            with st.spinner("Step 2: Simulating async invocation of Processor..."):
                st.info(f"Sending same payload to Processor at http://localhost:{PROCESSOR_HOST_PORT}")
                processor_response = requests.post(f"http://localhost:{PROCESSOR_HOST_PORT}/2015-03-31/functions/function/invocations", json=lambda_payload, timeout=90)

                # The processor handler has finished. Give the log streamer a moment
                # to capture any final log messages before we proceed to cleanup.
                time.sleep(2)
                update_logs()

                st.success("Processor invoked. Check logs for identification results.")

                # The processor's response is just a status, the real result is in the logs (simulating an SMS)

        except Exception as e:
            st.error(f"An error occurred during the test flow: {e}")
            st.session_state.container_logs.append(f"App error: {e}\n")
            update_logs()
        finally:
            # 6. Cleanup
            st.info("Stopping containers...")
            if responder_log_streamer:
                responder_log_streamer.stop()
            if processor_log_streamer:
                processor_log_streamer.stop()

            subprocess.run(["podman", "stop", RESPONDER_CONTAINER_NAME], capture_output=True)
            subprocess.run(["podman", "stop", PROCESSOR_CONTAINER_NAME], capture_output=True)
            st.success("Containers stopped.")
            
            if responder_log_streamer:
                responder_log_streamer.join(timeout=5)
            if processor_log_streamer:
                processor_log_streamer.join(timeout=5)
            
            image_server.stop()
            time.sleep(1)
            update_logs()
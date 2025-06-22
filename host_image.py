import sys
import os
from flask import Flask, send_file, abort

from urllib.parse import urlparse, unquote
app = Flask(__name__)

# Global variable to store the image path
IMAGE_PATH = None

@app.route('/')
def display_image():
    """Serves the image specified by the IMAGE_PATH."""
    if IMAGE_PATH and os.path.exists(IMAGE_PATH):
        try:
            return send_file(IMAGE_PATH)
        except Exception as e:
            print(f"Error sending file: {e}")
            abort(500, description="Error serving the image.")
    else:
        abort(404, description="Image not found or path not configured correctly.")

def main():
    global IMAGE_PATH

    if len(sys.argv) != 2:
        print("Usage: python host_image.py <path_to_image_file_or_uri>")
        sys.exit(1)

    input_path = sys.argv[1]
    image_file_path = ""

    # Check if the input is a file URI and parse it
    if input_path.startswith('file:///'):
        try:
            parsed_url = urlparse(input_path)
            # unquote handles URL-encoded characters like %20
            image_file_path = unquote(parsed_url.path)
        except Exception as e:
            print(f"Error parsing file URI: {e}")
            sys.exit(1)
    else:
        image_file_path = input_path

    if not os.path.isfile(image_file_path):
        print(f"Error: The resolved path '{image_file_path}' is not a file or does not exist.")
        sys.exit(1)

    # Validate image type (optional, but good practice)
    # You might want to add more sophisticated checks for actual image content/mime types
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    if not any(image_file_path.lower().endswith(ext) for ext in allowed_extensions):
        print(f"Warning: The file '{image_file_path}' does not have a common image extension. It might not display correctly.")
        # Decide if you want to exit or proceed
        # sys.exit(1)


    IMAGE_PATH = os.path.abspath(image_file_path)
    print(f"Serving image: {IMAGE_PATH}")
    print("Web application will be available at http://127.0.0.1:5000/")
    app.run(debug=False) # Set debug=True for development, False for production

if __name__ == '__main__':
    main()

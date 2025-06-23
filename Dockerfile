# Use the AWS Lambda Python 3.11 base image (check latest compatible version)
# For PyTorch, you might consider an image with CUDA if you're using GPU inference (unlikely for Lambda CPU)
# or a base Python image and install torch with CPU-only option.
# AWS also provides base images with common ML libraries pre-installed, e.g., public.ecr.aws/lambda/python:3.11-ml
FROM public.ecr.aws/lambda/python:3.13

# Set working directory in the container
WORKDIR /var/task

# Install system dependencies required by OpenCV (cv2)
# These are common libraries for graphics and X11 that are not in minimal Lambda images.
RUN dnf install -y mesa-libGL libXext libSM libICE libXrender fontconfig git


# Copy your requirements file and install dependencies
COPY horse-id-requirements.txt .
# Install dependencies into the function's directory, /var/task.
# The AWS Lambda runtime automatically adds this directory to the PYTHONPATH.
# The original --target "${LAMBDA_RUNTIME_DIR}" installed packages to /var/runtime,
# which is not in the PYTHONPATH, making them unavailable to the Python interpreter.
RUN pip install -r horse-id-requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN python -c "import timm; timm.create_model('hf-hub:BVRA/wildlife-mega-L-384', num_classes=0, pretrained=True)"

# Copy your application code and config file
COPY horse_id.py ${LAMBDA_TASK_ROOT}/
COPY config.yml ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler (app.lambda_handler)
CMD [ "horse_id.lambda_handler" ]
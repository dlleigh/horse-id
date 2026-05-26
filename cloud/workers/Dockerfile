FROM public.ecr.aws/lambda/python:3.13

WORKDIR /var/task

# System dependencies for OpenCV / YOLO
RUN dnf install -y mesa-libGL libXext libSM libICE libXrender fontconfig git

# Install Python dependencies (CPU-only PyTorch)
# Build context is project root, so paths are relative to that
COPY cloud/workers/requirements.txt .
RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt \
    --no-cache-dir \
    --target "${LAMBDA_TASK_ROOT}"

# Pre-download models at build time
RUN mkdir -p /opt/ml/model && \
    python -c "import timm; timm.create_model('hf-hub:BVRA/wildlife-mega-L-384', pretrained=True, cache_dir='/opt/ml/model')"

RUN python -c "from ultralytics import YOLO; YOLO('yolo11x-seg.pt')"

# Copy fine-tuned weights from S3 (if available)
RUN pip install awscli --no-cache-dir && \
    aws s3 cp s3://horse-id-models/finetuned-mega-L-384.pth /opt/ml/model/finetuned-mega-L-384.pth || \
    echo "No fine-tuned weights found in S3, using pretrained model"

# Copy worker code + detection library + config
COPY cloud/workers/*.py ${LAMBDA_TASK_ROOT}/
COPY horse_detection_lib.py ${LAMBDA_TASK_ROOT}/
COPY config.yml ${LAMBDA_TASK_ROOT}/

ENV MODEL_CACHE_DIR="/opt/ml/model"
ENV HF_HUB_OFFLINE="1"
ENV MATPLOTLIB_CONFIGDIR="/tmp/matplotlib_cache"
ENV YOLO_AUTOINSTALL="false"
ENV YOLO_CONFIG_DIR="/tmp/Ultralytics"

CMD [ "handler.lambda_handler" ]

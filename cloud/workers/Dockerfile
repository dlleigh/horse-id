FROM public.ecr.aws/lambda/python:3.13

WORKDIR /var/task

# System dependencies for OpenCV / YOLO
RUN dnf install -y mesa-libGL libXext libSM libICE libXrender fontconfig git

# Install Python dependencies (CPU-only PyTorch)
COPY requirements.txt .
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

# Copy worker code
COPY *.py ${LAMBDA_TASK_ROOT}/

# Copy detection config
COPY ../config.yml ${LAMBDA_TASK_ROOT}/ 2>/dev/null || true

ENV MODEL_CACHE_DIR="/opt/ml/model"
ENV MATPLOTLIB_CONFIGDIR="/tmp/matplotlib_cache"

CMD [ "handler.lambda_handler" ]

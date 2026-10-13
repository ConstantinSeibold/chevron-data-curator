# Chevron for a server: hosted mode, models on the GPU. See docs/hosting/README.md.
#   docker compose build && docker compose up -d
FROM python:3.12-slim

# the CUDA wheels bundle their own CUDA runtime; the host only needs the NVIDIA driver and the
# container toolkit. TORCH_INDEX=https://download.pytorch.org/whl/cpu builds a CPU-only image.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
# files written to the mounted folders belong to this user on the host
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"

COPY pyproject.toml README.md LICENSE /opt/chevron/
COPY chevron /opt/chevron/chevron
RUN pip install --no-cache-dir "/opt/chevron[all]"

RUN groupadd -g "$GID" chevron && useradd -m -u "$UID" -g "$GID" chevron \
    && mkdir -p /srv/chevron/projects /srv/chevron/data /home/chevron/.cache \
    && chown -R chevron:chevron /srv/chevron /home/chevron
USER chevron
ENV CHEVRON_ROOT=/srv/chevron/projects \
    CHEVRON_DATA_ROOT=/srv/chevron/data \
    CHEVRON_HOST=0.0.0.0 \
    CHEVRON_PORT=7870 \
    PYTHONUNBUFFERED=1
EXPOSE 7870
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7870/login', timeout=4)"
ENTRYPOINT ["chevron"]

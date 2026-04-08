# ─────────────────────────────────────────────────────────────────────────────
#  Dockerfile — Particle Simulator  (Jetson Nano, JetPack 4.6.x)
#
#  Base image targets L4T r32.7.1 (JetPack 4.6.1).
#  Check your JetPack version:  cat /etc/nv_tegra_release
#
#  If you are on JetPack 4.5.x use:  nvcr.io/nvidia/l4t-base:r32.5.0
#  If you are on JetPack 4.4.x use:  nvcr.io/nvidia/l4t-base:r32.4.3
# ─────────────────────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/l4t-base:r32.7.1

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    # OpenCV runtime libs
    libopencv-dev \
    python3-opencv \
    # GUI / display (needed for cv2.imshow via X11)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # MediaPipe build / runtime deps
    libprotobuf-dev \
    libhdf5-dev \
    # camera & V4L2
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip ───────────────────────────────────────────────────────────────
RUN python3 -m pip install --upgrade pip setuptools wheel

# ── Python packages ───────────────────────────────────────────────────────────
# numpy first — mediapipe build may need it
RUN pip3 install --no-cache-dir numpy==1.21.6

# mediapipe 0.9.3 is the last release that:
#   • ships an official linux_aarch64 wheel on PyPI
#   • uses the stable mp.solutions.hands API (no .task file required)
#   • runs on Python 3.6 / 3.8 (JetPack 4.6 default)
RUN pip3 install --no-cache-dir mediapipe==0.9.3

# opencv-python: skip if system python3-opencv is sufficient;
# uncomment if you need a newer version
# RUN pip3 install --no-cache-dir opencv-python-headless==4.7.0.72

# ── Application ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .

# Entrypoint
CMD ["python3", "main.py"]

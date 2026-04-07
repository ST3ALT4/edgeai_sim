# Use NVIDIA's official Jetson Nano ML image
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (excluding mediapipe)
RUN pip3 install onnxruntime-gpu numpy

# Copy your project files
COPY . .

# Set display for GUI output
ENV DISPLAY=:0

# Run the simulator
CMD ["python3", "main.py"]

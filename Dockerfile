FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    ffmpeg \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    libsndfile1 \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    python3-dev \
    build-essential \
    libaubio-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy only requirements first to leverage Docker layer caching
COPY requirements-base.txt requirements-heavy.txt ./

# Install base (lighter) dependencies
RUN pip install --no-cache-dir -r requirements-base.txt

# Install heavier dependencies separately to isolate potential failures
RUN pip install --no-cache-dir -r requirements-heavy.txt

# Now copy rest of your app code
COPY . .

# Default command
CMD ["python", "app.py"]

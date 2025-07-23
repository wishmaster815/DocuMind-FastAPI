# ---- STAGE 1: The Builder ----
# This stage builds the Python dependencies
FROM python:3.10-slim as builder

# Install build tools needed to compile some Python packages
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the requirements file
COPY requirements.txt .

# Create a virtual environment to isolate packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model into the builder stage's cache
# We will copy the final cache directory to the final image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# ---- STAGE 2: The Final Image ----
# This stage creates the lean final image
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the model cache from the builder stage
# The default cache for sentence-transformers is /root/.cache/torch
COPY --from=builder /root/.cache /root/.cache

# Copy the application code
COPY . .

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set the command to run your app
# The port should be the one your host expects, often 8000 or 8080
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
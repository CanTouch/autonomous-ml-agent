FROM python:3.12-slim

# System settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install system dependencies (libgomp1 for XGBoost OpenMP, curl for healthcheck, bash)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create runtime directories
RUN mkdir -p /app/data/raw /app/data/processed /app/reports /app/.cache/matplotlib

# Copy application files
COPY . /app/

# Ensure shell scripts have execute permissions
RUN chmod +x /app/workflows/*.sh

# Expose default Streamlit port (8501) and Hugging Face Spaces port (7860)
EXPOSE 8501 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Launch autonomous ML agent Streamlit application
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]

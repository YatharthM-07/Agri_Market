FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY agrimarket-optimizer .

# HF Spaces uses port 7860
EXPOSE 7860

# Environment variable defaults (override at runtime)
ENV PORT=7860
ENV AGRI_TASK=task1
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

# Start the OpenEnv HTTP server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for some ML libraries like XGBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit uses port 8501 by default
EXPOSE 8501

# Healthcheck to ensure the container is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
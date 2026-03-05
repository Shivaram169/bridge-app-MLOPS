# Use Python 3.9
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file
COPY requirements.txt .

# Install your libraries (XGBoost, FastAPI, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files (app.py, drift_detection.py, etc.)
COPY . .

# Expose the port for your API
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Use a lightweight Python image
FROM python:3.9-slim

# Set the work directory inside the container
WORKDIR /app

# Install system dependencies (needed for XGBoost/Scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements first
COPY requirements.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files (app.py, drift_detection.py, etc.)
COPY . .

# Expose the port FastAPI uses
EXPOSE 8000

# Start the application pointing to your app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Use Python 3.11 slim base image for lightweight container
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY client.py server.py verify.py test_model.py ./
COPY data ./data

# Expose port for Flower server
EXPOSE 8080

# Command to run federated learning (server and clients in separate containers)
CMD ["python", "server.py"]
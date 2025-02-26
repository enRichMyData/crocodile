# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install Git and any other necessary packages
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port 8006 (GraphQL endpoint)
EXPOSE 8006

# Run the FastAPI/GraphQL server using uvicorn.
CMD ["uvicorn", "crocodile_upload_file:app", "--host", "0.0.0.0", "--port", "8006"]


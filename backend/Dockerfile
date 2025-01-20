# Use the official Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose the app on port 8000
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "fastAPI:app", "--host", "0.0.0.0", "--port", "8000"]

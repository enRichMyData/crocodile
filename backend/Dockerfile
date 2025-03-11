# Use the official Python 3.11 slim image as the base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the app files directly to the root of the container's working directory
COPY ./backend/app /app

# Install pip and any dependencies from the requirements file
COPY ./backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

# Copy project configuration and requirements from the repository root
COPY pyproject.toml .
COPY ./crocodile ./crocodile

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[app]"

# Expose the port that FastAPI will use
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

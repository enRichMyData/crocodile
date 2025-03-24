FROM jupyter/base-notebook:latest

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm && python -m spacy download en_core_web_trf

# Copy project configuration and requirements from the repository root
COPY pyproject.toml .
COPY ./crocodile ./crocodile

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dependencies]"

# Expose Jupyter Notebook port
EXPOSE 8888

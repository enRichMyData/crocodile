FROM jupyter/base-notebook:latest

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm && python -m spacy download en_core_web_trf

# Expose Jupyter Notebook port
EXPOSE 8888

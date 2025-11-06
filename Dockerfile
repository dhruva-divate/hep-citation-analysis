FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the analysis script
COPY hep_citation_analysis.py .

# Create directory for data and results
RUN mkdir -p /app/data /app/results

# Download the dataset (optional - can be mounted instead)
# RUN wget http://snap.stanford.edu/data/cit-HepTh.txt.gz && \
#     gunzip cit-HepTh.txt.gz && \
#     mv cit-HepTh.txt /app/data/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "hep_citation_analysis.py", "--data", "/app/data/cit-HepTh.txt", "--output-dir", "/app/results"]

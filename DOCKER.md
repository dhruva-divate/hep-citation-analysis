# Docker Usage Guide

This guide explains how to run the HEP-Th Citation Network Analysis using Docker.

## Why Use Docker?

Docker provides:
- **Reproducibility**: Same environment everywhere
- **Isolation**: No conflicts with your system packages
- **Portability**: Easy to share and deploy
- **No installation hassles**: All dependencies bundled

## Prerequisites

Install Docker:
- **Linux**: `sudo apt-get install docker.io docker-compose`
- **Mac/Windows**: Download [Docker Desktop](https://www.docker.com/products/docker-desktop)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/hep-citation-analysis.git
cd hep-citation-analysis

# 2. Create data directory and download dataset
mkdir -p data results
cd data
wget http://snap.stanford.edu/data/cit-HepTh.txt.gz
gunzip cit-HepTh.txt.gz
cd ..

# 3. Build and run
docker-compose up --build

# Results will be in ./results/
```

### Option 2: Using Docker Directly

```bash
# 1. Build the image
docker build -t hep-citation-analysis .

# 2. Run the container
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/results:/app/results \
           hep-citation-analysis

# Results will be in ./results/
```

## Directory Structure

```
your-project/
 data/
    cit-HepTh.txt       # Place dataset here
 results/
    bridging_analysis.png  # Output appears here
 Dockerfile
 docker-compose.yml
 ...
```

## Advanced Usage

### Custom Arguments

```bash
# Using docker-compose
docker-compose run hep-analysis python hep_citation_analysis.py \
    --data /app/data/cit-HepTh.txt \
    --output-dir /app/results \
    --top-percent 5

# Using docker directly
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/results:/app/results \
           hep-citation-analysis \
           python hep_citation_analysis.py --top-percent 5
```

### Interactive Shell

```bash
# Open a shell in the container
docker-compose run hep-analysis /bin/bash

# Or with docker directly
docker run -it -v $(pwd)/data:/app/data \
               -v $(pwd)/results:/app/results \
               hep-citation-analysis /bin/bash
```

### Building Without Cache

```bash
docker-compose build --no-cache
```

## Troubleshooting

### Permission Issues

If you encounter permission issues with output files:

```bash
# Linux/Mac
sudo chown -R $USER:$USER results/

# Or run container as your user
docker run --user $(id -u):$(id -g) \
           -v $(pwd)/data:/app/data \
           -v $(pwd)/results:/app/results \
           hep-citation-analysis
```

### Memory Issues

If the analysis runs out of memory:

```bash
# Increase Docker memory limit (Docker Desktop: Settings > Resources)
# Or limit the analysis
docker run -m 4g -v $(pwd)/data:/app/data \
               -v $(pwd)/results:/app/results \
               hep-citation-analysis
```

### Data Not Found

Ensure `cit-HepTh.txt` is in the `data/` directory:

```bash
ls -lh data/cit-HepTh.txt
# Should show the file (~3MB)
```

## Cleanup

```bash
# Remove containers
docker-compose down

# Remove images
docker rmi hep-citation-analysis

# Remove all (containers, networks, volumes)
docker-compose down -v
```

## Performance Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Docker** | Reproducible, isolated, portable | Slight overhead, larger disk usage |
| **Native** | Fastest, direct file access | Manual dependency management |

**Recommendation**: 
- Use **Docker** for sharing, deployment, and reproducibility
- Use **Native** for development and fastest execution

## Windows-Specific Notes

On Windows PowerShell, use:

```powershell
# Replace $(pwd) with ${PWD}
docker run -v ${PWD}/data:/app/data `
           -v ${PWD}/results:/app/results `
           hep-citation-analysis
```

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: Run Analysis

on: [push]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Download data
        run: |
          mkdir data
          wget -O data/cit-HepTh.txt.gz http://snap.stanford.edu/data/cit-HepTh.txt.gz
          gunzip data/cit-HepTh.txt.gz
      - name: Run analysis
        run: docker-compose up --build
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: analysis-results
          path: results/
```

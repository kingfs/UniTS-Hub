# UniTS-Hub

A unified Docker serving interface for SOTA time-series foundation models.

## Overview

UniTS-Hub provides a standardized FastAPI-based interface to serve multiple state-of-the-art time-series foundation models using a single Docker image. You can switch between models simply by setting an environment variable.

Supported Models:
- **TimesFM** (Google Research)
- **Chronos** (Amazon Science)

## Quick Start

### Run with Docker

```bash
# To run with Chronos (default)
docker run -d -p 8000:8000 --gpus all -e MODEL_TYPE=chronos kingfs/units-hub:latest

# To run with TimesFM
docker run -d -p 8000:8000 --gpus all -e MODEL_TYPE=timesfm kingfs/units-hub:latest
```

### API Usage

Post a JSON request to `/predict`:

```json
{
  "instances": [
    {
      "history": [1.0, 2.0, 3.0, 4.0, 5.0]
    }
  ],
  "task": {
    "horizon": 3
  },
  "parameters": {
    "frequency": "1H"
  }
}
```

## Development

1. Clone the repository:
   ```bash
   git clone https://github.com/kingfs/UniTS-Hub.git
   cd UniTS-Hub
   ```

2. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install dependencies and setup environment:
   ```bash
   uv sync
   ```

4. Download models:
   ```bash
   uv run scripts/download_models.py
   ```

---
Built with ❤️ for the Time-Series community.

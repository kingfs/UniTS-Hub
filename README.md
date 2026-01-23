# UniTS-Hub 🚀

A unified Docker serving interface for SOTA (State-of-the-Art) **Time-Series Foundation Models**.

UniTS-Hub provides a standardized FastAPI-based interface to serve multiple time-series foundation models using a single Docker image. It is designed to be easily integrated into AI workflows like **Dify**, **LangChain**, and **Flowise**.

## ✨ Key Features

- **Unified Interface**: One API for multiple foundation models.
- **SOTA Models Support**: 
  - 📈 **TimesFM** (Google Research)
  - 📊 **Chronos** (Amazon Science)
- **Ready for AI Workflows**: Rich OpenAPI documentation and standard API Key authentication.
- **Easy Deployment**: Fully containerized with Docker.

## 🛠️ Quick Start

### 1. Model Preparation
Ensure you have the models downloaded in a local directory (e.g., `./models/timesfm` and `./models/chronos`).

### 2. Run with Docker
```bash
# Set your API Key
export API_KEY=your-secret-key

# Run with TimesFM (Recommended)
docker run -d -p 8000:8000 \
  -e MODEL_TYPE=timesfm \
  -e API_KEY=$API_KEY \
  -v $(pwd)/models:/app/models \
  kingfs/unitshub:latest
```

## 🔐 Authentication
UniTS-Hub uses **X-API-Key** header authentication for all `/predict` requests.

- **Header Name**: `X-API-Key`
- **Configuration**: Set the `API_KEY` environment variable in your `.env` or deployment script.

## 📝 API Usage

### Interactive Docs
Once the service is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Example Request (curl)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "instances": [
      {
        "history": [10.5, 12.1, 11.8, 13.2, 12.9],
        "metadata": {"seq_id": "sensor_01"}
      }
    ],
    "task": {
      "horizon": 5
    },
    "parameters": {
      "freq": "H"
    }
  }'
```

## 🤖 Dify Integration
UniTS-Hub is designed to work seamlessly as a **Dify Tool**:

1. Open your Dify dashboard.
2. Go to **Tools** -> **Create Custom Tool**.
3. Use the **URL** method and point to `http://your-server-ip:8000/openapi.json`.
4. Configure **Authentication**:
   - Type: `API Key`
   - Header Name: `X-API-Key`
   - Value: Your configured `API_KEY`.
5. Now you can use time-series forecasting in your Dify workflows!

## ⚙️ Configuration
Use environment variables or a `.env` file to configure the service:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_TYPE` | Model to serve (`timesfm` or `chronos`) | `chronos` |
| `MODELS_DIR` | Path to the directory containing models | `/app/models` |
| `API_KEY` | Secret key for API authentication | `unitshub-secret` |

## 🏗️ Local Development

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Setup Environment**:
   ```bash
   uv sync
   ```
3. **Run Locally**:
   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---
Built with ❤️ for the Time-Series community.

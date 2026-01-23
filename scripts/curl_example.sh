#!/bin/bash

# API Configuration
URL="http://localhost:8000/predict"
API_KEY="unitshub-secret"

# JSON Payload
# Note: history should be a list of floats
PAYLOAD='{
  "instances": [
    {
      "history": [681.16, 682.62, 683.5, 684.1, 685.2],
      "metadata": {"sequence_id": "sensor_01"}
    }
  ],
  "task": {
    "horizon": 12
  },
  "parameters": {
    "freq": "D"
  }
}'

# Execute Curl
curl -X POST "$URL" \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" \
     -d "$PAYLOAD"

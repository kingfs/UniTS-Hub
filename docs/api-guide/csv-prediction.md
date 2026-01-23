# CSV Prediction Guide 📊

UniTS-Hub supports direct CSV file uploads for time-series forecasting. This is useful when you have large datasets or prefer not to format data into JSON blocks.

## 🚀 Quick Start (CSV)

### 1. Endpoint
- **URL**: `/predict/csv`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

### 2. Form Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | The `.csv` file containing your time-series data. |
| `target_column` | String | Yes | The header name of the column you want to forecast. |
| `horizon` | Integer | Yes | Number of future time points to predict. |
| `freq` | String | No | Frequency of the data (e.g., `H`, `D`, `5min`). Default is `auto`. |

## 🛠️ Efficient Data Handling
UniTS-Hub uses **Polars** for high-performance CSV processing. It handles large files efficiently and automatically drops missing values (`null`s) before passing the sequence to the foundation models.

## 💻 Example using Curl

```bash
curl -X POST http://localhost:8000/predict/csv \
     -H "Authorization: Bearer your-secret-key" \
     -F "file=@your_data.csv" \
     -F "target_column=TOTAIRFL" \
     -F "horizon=12"
```

## 🐍 Example using Python

See the full example script in `scripts/predict_csv_example.py`.

```python
import requests

files = {'file': open('data.csv', 'rb')}
data = {
    'target_column': 'TOTAIRFL',
    'horizon': 12
}
response = requests.post("http://localhost:8000/predict/csv", 
                         headers={"Authorization": "Bearer your-key"}, 
                         files=files, 
                         data=data)
print(response.json())
```

## ⚠️ Important Notes
- **Univariate**: Current foundation models (TimesFM, Chronos) are univariate. You must specify **one** column to forecast.
- **No Timestamps**: You don't need to include timestamps in the `target_column`. The model only takes the sequence of numerical values.
- **Cleaning**: Any non-numerical rows or null values in the target column will be automatically removed.

import requests
import os

# API Configuration
URL = "http://localhost:8000/predict/csv"
API_KEY = "unitshub-secret"
HEADERS = {
    "Authorization": f'Bearer {API_KEY}'
}

# 1. Create a sample CSV file for demonstration
csv_filename = "sample_data.csv"
with open(csv_filename, "w") as f:
    f.write("Time,TOTAIRFL,TOTFUELFL\n")
    f.write("2020-6-11 8:0:0,681.16,91.51\n")
    f.write("2020-6-11 8:0:5,682.62,91.53\n")
    f.write("2020-6-11 8:0:10,683.50,91.55\n")
    f.write("2020-6-11 8:0:15,684.10,91.57\n")
    f.write("2020-6-11 8:0:20,685.20,91.59\n")

def main():
    # 2. Prepare the file and data
    files = {
        'file': (csv_filename, open(csv_filename, 'rb'), 'text/csv')
    }
    data = {
        'target_column': 'TOTAIRFL', # The column you want to forecast
        'horizon': 12,              # Forecast horizon
        'freq': 'auto'               # Frequency
    }

    # 3. Call the API
    try:
        print(f"Uploading {csv_filename} to {URL}...")
        response = requests.post(URL, headers=HEADERS, files=files, data=data)
        response.raise_for_status()
        
        result = response.json()
        print(f"Model used: {result['model']}")
        
        for i, forecast in enumerate(result['forecasts']):
            print(f"Forecast for sequence {i}:")
            print(f"Mean: {forecast['mean']}")
            
    except Exception as e:
        print(f"Error: {e}")
        if 'response' in locals() and response.text:
            print(f"Response: {response.text}")
    finally:
        # Clean up
        if os.path.exists(csv_filename):
            os.remove(csv_filename)

if __name__ == "__main__":
    main()

# src/gpu_benchmark/database.py
import requests
import datetime
import torch

# Hardcoded Supabase credentials (anon key is designed to be public)
SUPABASE_URL = "https://jftqjabhnesfphpkoilc.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmdHFqYWJobmVzZnBocGtvaWxjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ5NzI4NzIsImV4cCI6MjA2MDU0ODg3Mn0.S0ZdRIauUyMhdVJtYFNquvnlW3dV1wxERy7YrurZyag"

def country_code_to_flag(country_code):
    """Convert country code to flag emoji."""
    if len(country_code) != 2 or not country_code.isalpha():
        return "🏳️"  # White flag for unknown
    
    # Convert each letter to regional indicator symbol
    # A-Z: 0x41-0x5A -> regional indicators: 0x1F1E6-0x1F1FF
    return ''.join(chr(ord(c.upper()) - ord('A') + ord('🇦')) for c in country_code)

def get_country_flag():
    """Get country flag emoji based on IP."""
    try:
        country_response = requests.get("https://ipinfo.io/json")
        country_code = country_response.json().get("country", "Unknown")
        return country_code_to_flag(country_code)
    except Exception as e:
        print(f"Error getting country info: {e}")
        return "🏳️"  # White flag for unknown

def upload_benchmark_results(model_name: str, primary_metric_value: int, max_temp: float, avg_temp: float, cloud_provider: str = "Private", **kwargs):
    """Upload benchmark results to Supabase database.
    
    Args:
        model_name: Name of the model ("stable-diffusion-1-5", "qwen3-0-6b") to determine the target table.
        primary_metric_value: Value for the primary metric (e.g., images generated or generations processed),
                              which will be stored in the 'result' column.
        max_temp: Maximum GPU temperature recorded.
        avg_temp: Average GPU temperature recorded.
        cloud_provider: Cloud provider name (default: "Private").
        **kwargs: Additional fields to upload (e.g., gpu_power_watts, gpu_memory_total).
        
    Returns:
        tuple: (success, message, record_id)
    """
    
    table_name = ""
    metric_column_name = "result" # Generic column name for the primary metric

    if model_name == "stable-diffusion-1-5":
        table_name = "stable-diffusion-1-5"
    elif model_name == "qwen3-0-6b":
        table_name = "qwen3-0-6b"
    elif model_name == "nanogpt-train":
        table_name = "nanogpt-train"
    else:
        err_msg = f"Unsupported model_name '{model_name}' for database upload."
        print(f"❌ {err_msg}")
        return False, err_msg, None

    # Get country flag
    flag_emoji = get_country_flag()
    
    # Prepare benchmark results
    benchmark_data = {
        "created_at": datetime.datetime.now().isoformat(),
        "gpu_type": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "N/A",
        metric_column_name: primary_metric_value, # Using "result" as the column name
        "max_heat": int(max_temp) if max_temp is not None else None,
        "avg_heat": int(avg_temp) if avg_temp is not None else None,
        "country": flag_emoji,
        "provider": cloud_provider
    }
    
    # Add additional fields if provided.
    additional_fields_expected = [
        "gpu_power_watts", "gpu_memory_total", "platform", 
        "acceleration", "torch_version",
        # Training benchmark specific fields
        "iterations", "tokens_per_sec", "train_loss"
    ]
    
    for field in additional_fields_expected:
        if field in kwargs and kwargs[field] is not None:
            benchmark_data[field] = kwargs[field]
    
    api_url = f"{SUPABASE_URL}/rest/v1/{table_name}" # Dynamic table name
    
    try:
        response = requests.post(
            api_url,
            json=benchmark_data,
            headers={
                "Content-Type": "application/json",
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Prefer": "return=representation"
            }
        )
        
        if response.status_code in (200, 201):
            try:
                record_data = response.json()
                if isinstance(record_data, list) and len(record_data) > 0:
                    record_id = record_data[0].get('id')
                    print(f"✅ Results uploaded successfully to benchmark results!")
                    print(f"Your ID at www.unitedcompute.ai/gpu-benchmark: {record_id}")
                    return True, "Upload successful", record_id
                else:
                    print(f"✅ Upload successful, but couldn't retrieve ID from response: {record_data}")
                    return True, "Upload successful, but couldn't retrieve ID", None
            except ValueError as e: # Catch JSON decoding errors
                print(f"✅ Upload reported success (status {response.status_code}), but failed to parse JSON response: {e}. Response text: '{response.text}'")
                return True, f"Upload successful (status {response.status_code}), but error parsing response", None
        else:
            error_details = f"Status Code: {response.status_code}. Response Body: '{response.text}'. Headers: {response.headers}"
            error_message = f"Failed to upload results to Supabase. {error_details}"
            print(f"❌ Database Upload Error: {error_message}")
            if response.status_code == 400:
                print("Hint (400 Bad Request): This might be due to a mismatch between the data sent and the table schema in Supabase (e.g., wrong data types for columns, missing required columns that are not nullable, or malformed JSON). Check the 'Response Body' above for specific column errors from Supabase.")
            elif response.status_code == 401:
                print("Hint (401 Unauthorized): Check if the Supabase ANON_KEY is correct and has the necessary INSERT permissions for the table. Review Row Level Security (RLS) policies on the table.")
            elif response.status_code == 403:
                 print("Hint (403 Forbidden): The request was understood, but refused. This often relates to permissions, possibly RLS policies or service-level API key permissions for insert operations on the target table.")
            elif response.status_code == 404:
                print(f"Hint (404 Not Found): Check if the table_name '{table_name}' is correct and the API endpoint '{api_url}' is valid. The table might not exist or the URL path could be wrong.")
            return False, error_message, None
            
    except requests.exceptions.ConnectionError as e:
        error_message = f"Network Connection Error: Failed to connect to Supabase at {SUPABASE_URL}. Details: {e}"
        print(f"❌ {error_message}")
        print("Troubleshooting: Check your internet connection and firewall settings. Ensure Supabase services are operational.")
        return False, error_message, None
    except requests.exceptions.Timeout as e:
        error_message = f"Request Timeout: The request to Supabase timed out. URL: {api_url}. Details: {e}"
        print(f"❌ {error_message}")
        print("Troubleshooting: Check your network connection. The Supabase server might be overloaded or slow to respond.")
        return False, error_message, None
    except requests.exceptions.RequestException as e: # Catches other requests-related errors (e.g., invalid URL)
        error_message = f"Request Error: An error occurred during the request to Supabase. URL: {api_url}. Details: {type(e).__name__} - {e}"
        print(f"❌ {error_message}")
        return False, error_message, None
    except Exception as e:
        import traceback
        error_message = f"Unexpected Error: An unexpected Python error occurred during database upload. Details: {type(e).__name__} - {e}"
        print(f"❌ {error_message}")
        traceback.print_exc()
        return False, error_message, None
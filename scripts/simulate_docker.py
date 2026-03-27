import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
import pandas as pd
from datetime import datetime, timezone

# ==========================================
# 1. CONFIGURATION & DEMO SETTINGS
# ==========================================
API_URL = "http://localhost:8000/predict"
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mock_anomaly_telemetry.csv")

# 🎛️ WHICH ANOMALY DO YOU WANT TO DEMO?
# 1 = The "Infinite Loop" (Payment API CPU Spike)
# 2 = The "Crash Loop" (API Gateway Flatline)
SCENARIO_TO_TEST = 1

# Terminal color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
YELLOW = '\033[93m'

def run_simulation():
    if not os.path.exists(CSV_PATH):
        print(f"{RED}Error: Could not find {CSV_PATH}{RESET}")
        return

    print(f"{YELLOW}Loading 30-Dimensional Telemetry Data...{RESET}")
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    
    # Fast-forward to 10 seconds before the chosen anomaly
    if SCENARIO_TO_TEST == 1:
        start_row = 990  # Anomaly 1 hits at exactly 1000
        print(f"{YELLOW}Demo Mode 1: Hunting for CPU Spikes...{RESET}\n")
    else:
        start_row = 1390 # Anomaly 2 hits at exactly 1400
        print(f"{YELLOW}Demo Mode 2: Hunting for Dead Pods...{RESET}\n")
        
    print("Press Ctrl+C to stop.\n")
    
    for index, row in df.iloc[start_row:].iterrows():
        metrics_dict = row.to_dict()
        
        # We now send the entire cluster's state at once
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pod_id": "cluster-wide-monitor", 
            "metrics": metrics_dict
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            response_data = response.json()
            
            is_anomaly = response_data.get("anomaly_detected", False)
            confidence = response_data.get("confidence_score", 0.0)
            
            if is_anomaly:
                print(f"{RED}[ALERT] [{payload['timestamp']}] CLUSTER ANOMALY DETECTED! (Error Score: {confidence}){RESET}")
                print(f"{RED}        Triggering Automation Script for recovery...{RESET}\n")
                time.sleep(2) 
            else:
                print(f"{GREEN}[OK]    [{payload['timestamp']}] Cluster Healthy. (Error Score: {confidence}){RESET}")
                
        except requests.exceptions.ConnectionError:
            print(f"{RED}Failed to connect to API. Is 'python api.py' running?{RESET}")
            break
            
        time.sleep(1)

if __name__ == "__main__":
    try:
        run_simulation()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Simulation stopped by user.{RESET}")
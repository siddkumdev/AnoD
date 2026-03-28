import os
import sys
import time
import random
import pandas as pd
import requests 
from datetime import datetime, timezone

# ==========================================
# 0. PATH SETUP & IMPORTS
# ==========================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from src import config

# ==========================================    
# 1. CONFIGURATION
# ==========================================
# Pointing to the RAW data so the UI looks normal and the Broker does the scaling
CSV_PATH = os.path.join(ROOT_DIR, "data", "raw_telemetry_stream.csv")

# NOTE: Make sure your broker.py is actually running on 8000 (or change this to 8001 if needed)
BROKER_URL = "http://127.0.0.1:8000/predict" 

# 🎛️ DEMO DIAL: 1 = CPU Spike (Payment), 2 = Dead Pod (Gateway)
SCENARIO_TO_TEST = 1

# Terminal Colors
CYAN = '\033[96m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

SERVICES = config.MONITORED_SERVICES

def run_simulation():
    if not os.path.exists(CSV_PATH):
        print(f"{RED}Error: Could not find {CSV_PATH}. Did you run generate_mock_data.py?{RESET}")
        return

    print(f"{YELLOW}Loading RAW Cluster Telemetry Data...{RESET}")
    df = pd.read_csv(CSV_PATH)
    df.fillna(0.0, inplace=True)
    
    start_row = 990 if SCENARIO_TO_TEST == 1 else 1390
    print(f"\n{CYAN}===================================================={RESET}")
    print(f"{CYAN}🚀 STARTING LIVE CLUSTER STREAM (API ARCHITECTURE){RESET}")
    print(f"{CYAN}===================================================={RESET}\n")
    
    # --- RECOVERY STATE TRACKERS ---
    rebooting_service = None
    reboot_timer = 0
    anomaly_mask_until = 0 
    
    for index, row in df.iloc[start_row:].iterrows():
        metrics_dict = row.to_dict()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # ==========================================
        # 🩹 THE RECOVERY MASK (SIMULATING THE FIX)
        # ==========================================
        if rebooting_service and index < anomaly_mask_until:
            safe_rebooting_svc = rebooting_service.replace("-", "_")
            if reboot_timer > 0:
                metrics_dict[f"{safe_rebooting_svc}_cpu_usage"] = 0.0
                metrics_dict[f"{safe_rebooting_svc}_mem_usage"] = 0.0
                metrics_dict[f"{safe_rebooting_svc}_net_rx_kbps"] = 0.0
                metrics_dict[f"{safe_rebooting_svc}_net_tx_kbps"] = 0.0
                metrics_dict[f"{safe_rebooting_svc}_is_ready"] = 0.0
                reboot_timer -= 1
            else:
                metrics_dict[f"{safe_rebooting_svc}_cpu_usage"] = 25.0 + random.uniform(-2, 2)
                metrics_dict[f"{safe_rebooting_svc}_mem_usage"] = 40.0 + random.uniform(-1, 1)
                metrics_dict[f"{safe_rebooting_svc}_net_rx_kbps"] = 1500.0 + random.uniform(-100, 100)
                metrics_dict[f"{safe_rebooting_svc}_net_tx_kbps"] = 800.0 + random.uniform(-50, 50)
                metrics_dict[f"{safe_rebooting_svc}_is_ready"] = 1.0
                
                current_restarts = metrics_dict.get(f"{safe_rebooting_svc}_restart_count", 0.0)
                metrics_dict[f"{safe_rebooting_svc}_restart_count"] = current_restarts + 1.0
        
        # ==========================================
        # 🖥️ TERMINAL DASHBOARD RENDERER
        # ==========================================
        print(f"⏱️  {timestamp}")
        for svc in SERVICES:
            safe_svc = svc.replace("-", "_")
            
            cpu = metrics_dict.get(f"{safe_svc}_cpu_usage", 0.0)
            mem = metrics_dict.get(f"{safe_svc}_mem_usage", 0.0)
            rx = metrics_dict.get(f"{safe_svc}_net_rx_kbps", 0.0)
            tx = metrics_dict.get(f"{safe_svc}_net_tx_kbps", 0.0)
            ready = int(metrics_dict.get(f"{safe_svc}_is_ready", 0.0))
            
            if svc == rebooting_service and reboot_timer > 0:
                status = f"{MAGENTA}🔄 REBOOTING{RESET}"
                cpu_color = MAGENTA
            elif ready == 0 and cpu == 0.0:
                status = f"{RED}🔴 DOWN     {RESET}"
                cpu_color = RED
            else:
                status = f"{GREEN}🟢 RUNNING  {RESET}"
                cpu_color = RED if cpu > 90 else (YELLOW if cpu > 70 else RESET)
            
            print(f"  {status} | 📦 {svc:<14} | CPU: {cpu_color}{cpu:>5.1f}%{RESET} | MEM: {mem:>4.1f}MB | NET(RX/TX): {rx:>4.0f}/{tx:>4.0f} | RDY: {ready}")
            
        # ==========================================
        # 🌐 SEND DATA TO THE API BROKER
        # ==========================================
        try:
            # Package the metrics exactly how your broker expects them
            payload = {"metrics": metrics_dict}
            
            # Send the POST request to your FastAPI broker
            response = requests.post(BROKER_URL, json=payload, timeout=2)
            
            if response.status_code == 200:
                result = response.json()
                is_anomaly = result.get("anomaly_detected")
                confidence = result.get("confidence_score")
                # --- NEW: Extract per-service scores ---
                service_scores = result.get("service_scores", {})
                
                print("-" * 105) # Widened for the new columns
                
                # 1. Print the Global Verdict
                if is_anomaly and index >= anomaly_mask_until:
                    print(f"{RED}🚨 BROKER ALERT! Anomaly Detected (Global Loss: {confidence:.4f}){RESET}")
                    # ... [rest of your alert logic] ...
                else:
                    status_text = f"{GREEN}✅ Cluster Healthy" if not is_anomaly else f"{YELLOW}⏳ Healing"
                    print(f"{status_text} (Global Loss: {confidence:.4f}){RESET}")

                # 2. NEW: Print the Root Cause Analysis (Individual MSE Scores)
                print(f"{CYAN}🔍 Root Cause Analysis (Individual MSE):{RESET}")
                rca_line = ""
                for svc, score in service_scores.items():
                    # If a specific service's MSE is high, highlight it in Yellow
                    color = YELLOW if score > (config.ANOMALY_THRESHOLD / 2) else RESET
                    rca_line += f" | {svc}: {color}{score:.5f}{RESET}"
                print(f"   {rca_line} |")
                print("-" * 105 + "\n")                    
            else:
                print(f"{YELLOW}⚠️ Broker returned status code: {response.status_code}{RESET}")
                
        except requests.exceptions.ConnectionError:
            print("-" * 85)
            print(f"{RED}🔌 Error: Could not connect to broker at {BROKER_URL}{RESET}")
            print(f"{YELLOW}Hint: Ensure 'python broker.py' is running in a separate terminal window!{RESET}\n")
            
        time.sleep(1)

if __name__ == "__main__":
    run_simulation()
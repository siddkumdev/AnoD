import os
import sys
import numpy as np
import pandas as pd

# Pathing fix so it finds the src folder perfectly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

# ==========================================
# 🎛️ THE TUNING DIALS
# ==========================================
HEALTHY_ROWS = 10000
ANOMALY_ROWS = 2000

def get_base_metrics(num_rows):
    """Generates a perfectly healthy baseline dataframe for all 30 Fusion metrics."""
    data = {}
    
    for metric in config.EXPECTED_METRICS:
        if metric.endswith("_cpu_usage"):
            col_data = np.random.normal(loc=25.0, scale=5.0, size=num_rows)
            data[metric] = np.clip(col_data, 1.0, 100.0) 
            
        elif metric.endswith("_mem_usage"):
            col_data = np.random.normal(loc=40.0, scale=2.0, size=num_rows)
            data[metric] = np.clip(col_data, 1.0, 100.0)
            
        elif metric.endswith("_net_rx_kbps"):
            # Healthy receive traffic (e.g., ~1024 KB/s)
            col_data = np.random.normal(loc=1024.0, scale=100.0, size=num_rows)
            data[metric] = np.clip(col_data, 0.0, 10000.0)
            
        elif metric.endswith("_net_tx_kbps"):
            # Healthy transmit traffic (e.g., ~512 KB/s)
            col_data = np.random.normal(loc=512.0, scale=50.0, size=num_rows)
            data[metric] = np.clip(col_data, 0.0, 10000.0)
            
        elif metric.endswith("_restart_count"):
            # Healthy pods don't restart
            data[metric] = np.zeros(num_rows)
            
        elif metric.endswith("_is_ready"):
            # 1.0 means True (Healthy and ready to receive traffic)
            data[metric] = np.ones(num_rows)
            
        else:
            # Fallback for safety
            data[metric] = np.zeros(num_rows)
            
    return pd.DataFrame(data, columns=config.EXPECTED_METRICS)

def generate_healthy_csv(filepath):
    print(f"Generating {HEALTHY_ROWS} rows of healthy 30-metric baseline data...")
    df = get_base_metrics(HEALTHY_ROWS)
    df.to_csv(filepath, index=False)
    print(f"✅ Saved healthy data to {filepath}")

def generate_anomaly_csv(filepath):
    print(f"Generating {ANOMALY_ROWS} rows of anomaly test data...")
    df = get_base_metrics(ANOMALY_ROWS)
    
    # --- INJECT CHAOS ---
    start_idx = ANOMALY_ROWS // 2
    end_idx = start_idx + 300
    
    # SCENARIO A: The "Infinite Loop" on Payment API
    # High CPU, but Network drops to zero because it's frozen
    if "payment_api_cpu_usage" in df.columns:
        print("💉 Injecting Scenario A: Infinite Loop (payment_api)...")
        df.loc[start_idx:end_idx, "payment_api_cpu_usage"] = 99.5
        df.loc[start_idx:end_idx, "payment_api_net_rx_kbps"] = 2.0  # Basically dead
        df.loc[start_idx:end_idx, "payment_api_net_tx_kbps"] = 0.5
        
# SCENARIO B: The "Crash Loop" on API Gateway
    # Happens a little bit after Scenario A
    crash_start = end_idx + 100
    crash_end = crash_start + 200
    if "api_gateway_is_ready" in df.columns and crash_start < ANOMALY_ROWS:
        print("💥 Injecting Scenario B: Crash Loop (api_gateway)...")
        
        # 1. The pod becomes unready
        df.loc[crash_start:crash_end, "api_gateway_is_ready"] = 0.0
        
        # 2. THE FIX: Dead pods do not use hardware. Flatline everything to 0.
        df.loc[crash_start:crash_end, "api_gateway_cpu_usage"] = 0.0
        df.loc[crash_start:crash_end, "api_gateway_mem_usage"] = 0.0
        df.loc[crash_start:crash_end, "api_gateway_net_rx_kbps"] = 0.0
        df.loc[crash_start:crash_end, "api_gateway_net_tx_kbps"] = 0.0
        
        # 3. The restart count starts climbing
        for i, row_idx in enumerate(range(crash_start, crash_end)):
            df.loc[row_idx, "api_gateway_restart_count"] = float(i // 50 + 1)

    df.to_csv(filepath, index=False)
    print(f"✅ Saved anomaly data to {filepath}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    healthy_path = os.path.join(current_dir, "mock_healthy_telemetry.csv")
    anomaly_path = os.path.join(current_dir, "mock_anomaly_telemetry.csv")
    
    generate_healthy_csv(healthy_path)
    generate_anomaly_csv(anomaly_path)
    print("🎉 All 30-Dimensional mock data generation complete!")
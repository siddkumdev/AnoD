import os

# ==========================================
# 1. THE API CONTRACT (FEATURE VECTOR)
# ==========================================
# This is the exact list of metrics your ML model expects, in this EXACT order.
# If you train your model on 36 features, you must always provide 36 features.
EXPECTED_METRICS = [
    # --- Cart API Metrics ---
    "cart_api_cpu_usage",
    "cart_api_mem_usage",
    "cart_api_net_rx_kbps",
    "cart_api_net_tx_kbps",
    "cart_api_restart_count",
    "cart_api_is_ready",
    
    # --- Payment API Metrics ---
    "payment_api_cpu_usage",
    "payment_api_mem_usage",
    "payment_api_net_rx_kbps",
    "payment_api_net_tx_kbps",
    "payment_api_restart_count",
    "payment_api_is_ready",
    
    # --- Auth Service Metrics ---
    "auth_service_cpu_usage",
    "auth_service_mem_usage",
    "auth_service_net_rx_kbps",
    "auth_service_net_tx_kbps",
    "auth_service_restart_count",
    "auth_service_is_ready",
    
    # --- Inventory API Metrics ---
    "inventory_api_cpu_usage",
    "inventory_api_mem_usage",
    "inventory_api_net_rx_kbps",
    "inventory_api_net_tx_kbps",
    "inventory_api_restart_count",
    "inventory_api_is_ready",
    

    # --- API Gateway Metrics ---
    "api_gateway_cpu_usage",
    "api_gateway_mem_usage",
    "api_gateway_net_rx_kbps",
    "api_gateway_net_tx_kbps",
    "api_gateway_restart_count",
    "api_gateway_is_ready"
]

# The mathematical value to inject if a service is assassinated by the Chaos script
# and its metric completely disappears from the Prometheus stream.
MISSING_DATA_DEFAULT = 0.0  

# ==========================================
# 2. MICROSERVICE REGISTRY
# ==========================================
# The list of target pods Role 2 (Infra) is deploying. 
# Used to attribute an anomaly to a specific service so Role 3 knows what to restart.
MONITORED_SERVICES = [
    "cart-api",
    "payment-api",
    "auth-service",
    "inventory-api",
    "api-gateway"
]

# ==========================================
# 3. ML MODEL SETTINGS
# ==========================================
# Filepath for saving/loading your PyTorch weights or Scikit-Learn pickle file
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "saved_models", "anomaly_model_v2.pth") # Bumped to v2

# If using a Neural Network (like an Autoencoder), this is the reconstruction error threshold.
# Anything above this error score is flagged as an anomaly.
ANOMALY_THRESHOLD = 0.088188 # You will likely need to recalibrate this after retraining

# ==========================================
# 4. TELEMETRY SETTINGS
# ==========================================
# The local URL where Role 2 will eventually host the Prometheus server.
# You will leave this alone during Phase 1 while you use mock CSV data.
PROMETHEUS_URL = "http://localhost:9090"
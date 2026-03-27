import os

# ==========================================
# 1. THE API CONTRACT (FEATURE VECTOR)
# ==========================================
# This is the exact list of metrics your ML model expects, in this EXACT order.
# If you train your model on 10 features, you must always provide 10 features.
EXPECTED_METRICS = [
    "cart_api_cpu_usage",
    "cart_api_mem_usage",
    "payment_api_cpu_usage",
    "payment_api_mem_usage",
    "auth_service_cpu_usage",
    "auth_service_mem_usage",
    "inventory_api_cpu_usage",
    "inventory_api_mem_usage",
    "frontend_cpu_usage",
    "frontend_mem_usage"
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
    "frontend"
]

# ==========================================
# 3. ML MODEL SETTINGS
# ==========================================
# Filepath for saving/loading your PyTorch weights or Scikit-Learn pickle file
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "saved_models", "anomaly_model_v1.pth")

# If using a Neural Network (like an Autoencoder), this is the reconstruction error threshold.
# Anything above this error score is flagged as an anomaly.
ANOMALY_THRESHOLD = 0.005201

# ==========================================
# 4. TELEMETRY SETTINGS
# ==========================================
# The local URL where Role 2 will eventually host the Prometheus server.
# You will leave this alone during Phase 1 while you use mock CSV data.
PROMETHEUS_URL = "http://localhost:9090"
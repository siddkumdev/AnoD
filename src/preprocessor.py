import pandas as pd
from src import config
def enforce_feature_contract(df: pd.DataFrame) -> pd.DataFrame:
    """
    This is your protection against the 'Chaos Trap'.
    It ensures the dataframe has EXACTLY the columns expected by the model,
    in the EXACT correct order.
    """
    # 1. Check for missing columns and inject the default value (0.0)
    for col in config.EXPECTED_METRICS:
        if col not in df.columns:
            # If the Chaos script killed 'payment_api', its CPU metric won't exist.
            # We pad it with 0.0 so the math doesn't break.
            df[col] = config.MISSING_DATA_DEFAULT
            
    # 2. Drop any extra garbage columns we don't care about (like timestamps)
    # and lock in the exact order.
    enforced_df = df[config.EXPECTED_METRICS]
    
    # Optional but recommended for Neural Networks: Normalize the data (0 to 1)
    # Assuming CPU percentages are 0-100, we divide by 100.
    # If your mock data is already 0.0 to 1.0, you can remove this line.
    enforced_df = enforced_df / 100.0 
    
    return enforced_df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. SYNTHETIC RAW DATA GENERATION
# ==========================================
np.random.seed(42)
base_time = pd.date_range(start="2026-09-29 08:00:00", periods=100, freq="15s")

# Generate timestamps with duplicate network retries
timestamps = list(base_time) + [base_time[10], base_time[25]]
temp_fahrenheit = np.random.normal(loc=175.0, scale=2.5, size=102)
pressure_psi = np.random.normal(loc=72.5, scale=3.0, size=102)

# Inject common industrial sensor anomalies
temp_fahrenheit[15] = 999.0  # Outlier: Hardware voltage spike
temp_fahrenheit[45] = -150.0  # Outlier: Sensor disconnect (-150 F impossible)
temp_fahrenheit[30:34] = np.nan  # Packet loss: Missing temperature sequence
pressure_psi[60:63] = np.nan  # Packet loss: Missing pressure sequence

df_raw = pd.DataFrame({
    "timestamp": timestamps,
    "sensor_id": "REACTOR_TEMP_01",
    "temperature_f": temp_fahrenheit,
    "pressure_psi": pressure_psi
})


# ==========================================
# 2. DATA PIPELINE PROCESSING FUNCTION
# ==========================================
def process_industrial_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning, validation, unit conversion, imputation, and resampling pipeline for industrial time-series data.
    """
    # A. Deduplication on composite keys
    df_clean = df.drop_duplicates(subset=["timestamp", "sensor_id"]).copy()  # same timestamp and same sensor ID in different rows make no sense --> duplicate data
    df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

    # B. Unit Standardization: °F to °C, PSI to Bar
    # T(°C) = (T(°F) - 32) * 5/9
    # P(bar) = P(psi) * 0.0689476
    df_clean["temperature_c"] = (df_clean["temperature_f"] - 32.0) * (5.0 / 9.0)
    df_clean["pressure_bar"] = df_clean["pressure_psi"] * 0.0689476
    df_clean.drop(columns=["temperature_f", "pressure_psi"], inplace=True)

    # C. Outlier Detection: Domain Boundaries & Statistical Range Check
    # Reactor physical boundaries: [50°C, 120°C]
    lower_bound_c, upper_bound_c = 50.0, 120.0
    outlier_mask = (
            (df_clean["temperature_c"] < lower_bound_c) |
            (df_clean["temperature_c"] > upper_bound_c)
    )
    df_clean.loc[outlier_mask, "temperature_c"] = np.nan  # Mark out-of-bounds temperature readings as NaN for next imputation step

    # D. Missing Data Imputation based on variable physics
    # Temperature (continuous thermal state) -> Linear Interpolation
    df_clean["temperature_c"] = df_clean["temperature_c"].interpolate(method="linear")
    # Pressure (steady state) -> Forward Fill with last sensed value. CAVEAT: if NaNs are at the start, there is no "last sensed value", hence a backward fill pass is also needed to grab the first reading and propagate it backwards
    df_clean["pressure_bar"] = df_clean["pressure_bar"].ffill().bfill()

    # E. Timestamp Alignment & Resampling to a Strict 30-Second Grid
    df_clean.set_index("timestamp", inplace=True)
    df_resampled = df_clean[["temperature_c", "pressure_bar"]].resample("30s").mean()
    # Fill any empty grid slots resulting from downsampling gaps using temporal interpolation
    df_resampled = df_resampled.interpolate(method="time")

    return df_resampled


# Run processing pipeline
df_processed = process_industrial_telemetry(df_raw)

# ==========================================
# 3. VISUAL VALIDATION & COMPARISON
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Temperature Plot
axes[0].plot((df_raw["timestamp"]), (df_raw["temperature_f"] - 32) * 5 / 9,
             'r.', label='Raw Sensor Signals (with outliers/gaps)', alpha=0.6, markersize=8)
axes[0].plot(df_processed.index, df_processed["temperature_c"],
             'b-o', label='Cleaned & Resampled (30s Grid)', linewidth=1.5, markersize=4)
axes[0].set_ylabel("Temperature (°C)")
axes[0].set_title("Industrial Telemetry Pipeline: Temperature Refinement")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(loc="upper right")

# Pressure Plot
axes[1].plot(df_raw["timestamp"], df_raw["pressure_psi"] * 0.0689476,
             'm.', label='Raw Pressure Signals (PSI converted)', alpha=0.6, markersize=8)
axes[1].plot(df_processed.index, df_processed["pressure_bar"],
             'k-s', label='Cleaned & Resampled (30s Grid)', linewidth=1.5, markersize=4)
axes[1].set_xlabel("Timestamp")
axes[1].set_ylabel("Pressure (bar)")
axes[1].set_title("Industrial Telemetry Pipeline: Pressure Refinement")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()
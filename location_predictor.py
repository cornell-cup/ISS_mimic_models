import kagglehub
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

class SinusoidalRegressor:
    def __init__(self, period_guess=92.9):
        self.params = None
        self.period_guess = period_guess
        self.t_min = 0

    def _sin_func(self, t, A, omega, phi, C):
        return A * np.sin(omega * t + phi) + C
        
    def fit(self, X, y):
        # Calculate time in minutes from a baseline
        t = (X['day_of_year'] * 24 * 60 + X['hour_of_day'] * 60 + X['minute']).values
        if len(t) > 0:
            self.t_min = t[0]
        t = t - self.t_min
        
        A_guess = (np.max(y) - np.min(y)) / 2
        C_guess = np.mean(y)
        omega_guess = 2 * np.pi / self.period_guess
        
        p0 = [A_guess, omega_guess, 0, C_guess]
        try:
            self.params, _ = curve_fit(self._sin_func, t, y.values, p0=p0, maxfev=5000)
        except Exception:
            self.params = p0

    def predict(self, X):
        t = (X['day_of_year'] * 24 * 60 + X['hour_of_day'] * 60 + X['minute']).values
        t = t - self.t_min
        return self._sin_func(t, *self.params)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
# Replace this with your actual file path
path = kagglehub.dataset_download("vaibhavrawat277/iss-real-time-tracker-10s-interval-dataset")
print("Path to dataset files:", path)

# Load the dataset
df = pd.read_csv(f"{path}/iss_data.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# Make sure timestamp is parsed as datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by time — important!
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(df.head())

# ── 2. CREATE TARGET (lat/lon 1 hour later) ───────────────────────────────────
# Find the row that is ~1 hour ahead for each row
TARGET_MINUTES = 60
TOLERANCE_MINUTES = 5  # allow ±5 min window when matching

def find_future_position(df, target_minutes=60, tolerance=5):
    """
    For each row, find the row closest to target_minutes ahead.
    Returns a dataframe with future_lat and future_lon columns.
    """
    future_lats = []
    future_lons = []

    for i, row in df.iterrows():
        target_time = row["timestamp"] + pd.Timedelta(minutes=target_minutes)
        window = df[
            (df["timestamp"] >= target_time - pd.Timedelta(minutes=tolerance)) &
            (df["timestamp"] <= target_time + pd.Timedelta(minutes=tolerance))
        ]
        if len(window) > 0:
            # Pick the closest match
            closest = window.iloc[(window["timestamp"] - target_time).abs().argsort()[:1]]
            future_lats.append(closest["latitude"].values[0])
            future_lons.append(closest["longitude"].values[0])
        else:
            future_lats.append(np.nan)
            future_lons.append(np.nan)

    df["future_lat"] = future_lats
    df["future_lon"] = future_lons
    return df

print("\nCreating 1-hour-ahead targets...")
df = find_future_position(df)

# Drop rows where we couldn't find a future match
df = df.dropna(subset=["future_lat", "future_lon"])
print(f"Rows after matching: {df.shape[0]}")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
# Extract time-based features
df["hour_of_day"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day_of_year"] = df["timestamp"].dt.dayofyear

# Convert lat/lon to sine/cosine to handle wrap-around
# (longitude wraps -180 to 180, latitude wraps -90 to 90)
df["lon_sin"] = np.sin(np.radians(df["longitude"]))
df["lon_cos"] = np.cos(np.radians(df["longitude"]))
df["lat_sin"] = np.sin(np.radians(df["latitude"]))
df["lat_cos"] = np.cos(np.radians(df["latitude"]))

# ── 4. DEFINE FEATURES & TARGETS ──────────────────────────────────────────────
FEATURES = [
    "latitude", "longitude",
    "lat_sin", "lat_cos",
    "lon_sin", "lon_cos",
    "altitude_km", "speed_mph",
    "hour_of_day", "minute", "day_of_year"
]

# Only keep columns that exist in your dataset
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"\nUsing features: {FEATURES}")

X = df[FEATURES]
y_lat = df["future_lat"]
y_lon = df["future_lon"]

# ── 5. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
# Important: use shuffle=False to avoid data leakage across time!
X_train, X_test, y_lat_train, y_lat_test, y_lon_train, y_lon_test = train_test_split(
    X, y_lat, y_lon, test_size=0.2, shuffle=False
)

print(f"\nTraining rows: {len(X_train)}, Test rows: {len(X_test)}")

# ── 6. TRAIN MODEL ────────────────────────────────────────────────────────────
print("\nTraining model...")

model_lat = SinusoidalRegressor()
model_lon = SinusoidalRegressor()

model_lat.fit(X_train, y_lat_train)
model_lon.fit(X_train, y_lon_train)

# --- Add these lines to print the values ---
print("\n📈 Trained Parameters (A, omega, phi, C):")
print(f"Latitude Model : {model_lat.params}")
print(f"Longitude Model: {model_lon.params}")


# ── 7. EVALUATE ───────────────────────────────────────────────────────────────
pred_lat = model_lat.predict(X_test)
pred_lon = model_lon.predict(X_test)

mae_lat = mean_absolute_error(y_lat_test, pred_lat)
mae_lon = mean_absolute_error(y_lon_test, pred_lon)

print("\n📍 Results:")
print(f"  Latitude  MAE: {mae_lat:.2f}°  (~{mae_lat * 111:.0f} km)")
print(f"  Longitude MAE: {mae_lon:.2f}°  (varies by latitude)")

# # ── 8. KID-FRIENDLY PREDICTION EXAMPLE ───────────────────────────────────────
print("\n🚀 Example Prediction:")
sample = X_test.iloc[0:1]
p_lat = model_lat.predict(sample)[0]
p_lon = model_lon.predict(sample)[0]
actual_lat = y_lat_test.iloc[0]
actual_lon = y_lon_test.iloc[0]

print(f"  Current position : {sample['latitude'].values[0]:.2f}°, {sample['longitude'].values[0]:.2f}°")
print(f"  Predicted (1hr)  : {p_lat:.2f}°, {p_lon:.2f}°")
print(f"  Actual (1hr)     : {actual_lat:.2f}°, {actual_lon:.2f}°")

# # ── 9. FEATURE IMPORTANCE PLOT ────────────────────────────────────────────────
# Feature importance plot is omitted because Sinusoidal Regression relies purely on time.
print("\n✅ Omitting feature importance plot (not applicable for sine wave regression).")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("\n✅ Feature importance plot saved.")
plt.show(block=False)

# ── 10. INTERACTIVE PREDICTION ────────────────────────────────────────────────
def predict_custom_location():
    print("\n--- Try it out! Enter your own custom location ---")
    print("(Type 'q' or 'quit' to exit)")
    
    # Calculate median altitude and speed from the dataset to use as defaults
    median_altitude = df.get("altitude_km", pd.Series([420.0])).median()
    median_speed = df.get("speed_mph", pd.Series([17000.0])).median()
    
    while True:
        try:
            lat_input = input("\nEnter Latitude (-90 to 90): ")
            if lat_input.strip().lower() in ['q', 'quit']:
                break
                
            lon_input = input("Enter Longitude (-180 to 180): ")
            if lon_input.strip().lower() in ['q', 'quit']:
                break
                
            in_lat = float(lat_input)
            in_lon = float(lon_input)
            
            # Simple validation
            if not (-90 <= in_lat <= 90) or not (-180 <= in_lon <= 180):
                print("Please enter valid latitude (-90 to 90) and longitude (-180 to 180)!")
                continue
                
            now = pd.Timestamp.now(tz='UTC')
            
            # Build feature dictionary
            sample_data = {
                "latitude": in_lat,
                "longitude": in_lon,
                "lat_sin": np.sin(np.radians(in_lat)),
                "lat_cos": np.cos(np.radians(in_lat)),
                "lon_sin": np.sin(np.radians(in_lon)),
                "lon_cos": np.cos(np.radians(in_lon)),
                "altitude_km": median_altitude,
                "speed_mph": median_speed,
                "hour_of_day": now.hour,
                "minute": now.minute,
                "day_of_year": now.dayofyear
            }
            
            # Keep only the features exactly as in FEATURES
            sample_df = pd.DataFrame([{f: sample_data.get(f, 0) for f in FEATURES}])
            
            pred_future_lat = model_lat.predict(sample_df)[0]
            pred_future_lon = model_lon.predict(sample_df)[0]
            
            print(f"\n🔮 Prediction from custom position ({in_lat}°, {in_lon}°):")
            print(f"  Expected position in ~1 hour: {pred_future_lat:.2f}°, {pred_future_lon:.2f}°")
            
        except ValueError:
            print("Invalid input! Please enter numeric values.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error predicting: {e}")

if __name__ == "__main__":
    import sys
    if sys.stdin.isatty():
        predict_custom_location()
    else:
        print("\nℹ️ Non-interactive terminal detected. Skipping interactive prediction prompt.")
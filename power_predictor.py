"""
ISS Live Telemetry Data Collector
===================================
Logs ISS position, solar panel angles, and power generation to a CSV file.
Data comes directly from NASA's public Lightstreamer feed.

Install dependencies:
    pip install py-iss-telemetry

Run:
    python iss_data_collector.py

    By default it logs every 10 seconds until you press Ctrl+C.
    A new row is appended to iss_telemetry_log.csv each interval.
"""

import time
import csv
import os
from datetime import datetime, timezone

try:
    import pyisstelemetry
except ImportError:
    raise SystemExit(
        "Missing dependency. Run:  pip install py-iss-telemetry"
    )

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

OUTPUT_FILE   = "iss_telemetry_log.csv"
LOG_INTERVAL  = 10          # seconds between rows
MAX_ROWS      = None        # set to e.g. 5000 to auto-stop; None = run forever

# ── TELEMETRY PARAMETER NAMES (from ISS-Mimic / NASA ISSLIVE) ─────────────────
#
# Full dictionary: https://iss-mimic.github.io/Mimic/
# Each name is a NASA telemetry mnemonic that maps to a live data point.
#
# POSITION
PARAM_LAT       = "USLAB000052"   # ISS Latitude  (degrees)
PARAM_LON       = "USLAB000053"   # ISS Longitude (degrees)
PARAM_ALT       = "USLAB000054"   # ISS Altitude  (km)

# SOLAR ARRAY ANGLES — Beta Gimbal Assembly (BGA) angle for each of 8 arrays
# Each BGA rotates its array to track the sun. Range: 0–360°
PARAM_BGA = {
    "BGA_1A": "S4000003",    # Starboard 4, array 1A
    "BGA_1B": "S4000006",    # Starboard 4, array 1B
    "BGA_2A": "S6000003",    # Starboard 6, array 2A
    "BGA_2B": "S6000006",    # Starboard 6, array 2B
    "BGA_3A": "P4000003",    # Port 4, array 3A
    "BGA_3B": "P4000006",    # Port 4, array 3B
    "BGA_4A": "P6000003",    # Port 6, array 4A
    "BGA_4B": "P6000006",    # Port 6, array 4B
}

# SOLAR ALPHA ROTARY JOINT (SARJ) — rotates the whole outboard truss
PARAM_SARJ = {
    "SARJ_Starboard": "S0000007",
    "SARJ_Port":      "S0000008",
}

# POWER — Channel Voltage & Current for 8 power channels
# Power (W) = Voltage (V) × Current (A)  — we log both so you can compute it
PARAM_POWER = {
    "Ch1_Voltage": "EPSAL0001K",
    "Ch1_Current": "EPSAL0002K",
    "Ch2_Voltage": "EPSAL0003K",
    "Ch2_Current": "EPSAL0004K",
    "Ch3_Voltage": "EPSAL0005K",
    "Ch3_Current": "EPSAL0006K",
    "Ch4_Voltage": "EPSAL0007K",
    "Ch4_Current": "EPSAL0008K",
}

# ── COLLECT ALL PARAM NAMES ───────────────────────────────────────────────────

ALL_PARAMS = (
    [PARAM_LAT, PARAM_LON, PARAM_ALT]
    + list(PARAM_BGA.values())
    + list(PARAM_SARJ.values())
    + list(PARAM_POWER.values())
)

# Build a reverse lookup: mnemonic → friendly column name
FRIENDLY = {
    PARAM_LAT: "latitude",
    PARAM_LON: "longitude",
    PARAM_ALT: "altitude_km",
}
FRIENDLY.update({v: k for k, v in PARAM_BGA.items()})
FRIENDLY.update({v: k for k, v in PARAM_SARJ.items()})
FRIENDLY.update({v: k for k, v in PARAM_POWER.items()})

COLUMNS = ["timestamp"] + [FRIENDLY[p] for p in ALL_PARAMS]

# ── HELPER: COMPUTE TOTAL POWER ───────────────────────────────────────────────

def compute_channel_power(row: dict) -> dict:
    """Add a computed total power column (sum of V×I for each channel)."""
    total = 0.0
    for ch in range(1, 5):
        v_key = f"Ch{ch}_Voltage"
        i_key = f"Ch{ch}_Current"
        try:
            watts = float(row.get(v_key, 0)) * float(row.get(i_key, 0))
            row[f"Ch{ch}_Power_W"] = round(watts, 2)
            total+=watts
        except (TypeError, ValueError):
            row[f"Ch{ch}_Power_W"] = None
    row["Total_Power_W"] = round(total, 2)
    return row

# ── MAIN LOGGER ───────────────────────────────────────────────────────────────

def main():
    # Add computed power columns to CSV header
    power_cols = [f"Ch{ch}_Power_W" for ch in range(1, 5)] + ["Total_Power_W"]
    all_columns = COLUMNS + power_cols

    file_exists = os.path.isfile(OUTPUT_FILE)

    print("🚀 Connecting to NASA ISS live telemetry feed...")
    stream = pyisstelemetry.TelemetryStream()

    # Give the websocket a moment to receive initial values
    time.sleep(3)
    print(f"✅ Connected! Logging to '{OUTPUT_FILE}' every {LOG_INTERVAL}s")
    print("   Press Ctrl+C to stop.\n")

    rows_written = 0

    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")

        if not file_exists:
            writer.writeheader()

        try:
            while True:
                raw = stream.get_tm()

                # raw is a list of dicts: [{"name": "USLAB000052", "value": "51.6"}, ...]
                tm = {item["name"]: item["value"] for item in raw}

                # Build a clean row using friendly names
                row = {"timestamp": datetime.now(timezone.utc).isoformat()}
                for mnemonic in ALL_PARAMS:
                    col = FRIENDLY[mnemonic]
                    row[col] = tm.get(mnemonic)

                # Add computed power
                row = compute_channel_power(row)

                writer.writerow(row)
                f.flush()
                rows_written += 1

                # Print a live summary to the console
                lat  = row.get("latitude",    "?")
                lon  = row.get("longitude",   "?")
                alt  = row.get("altitude_km", "?")
                pwr  = row.get("Total_Power_W", "?")
                bga1 = row.get("BGA_1A", "?")

                print(
                    f"[{rows_written:>5}] {row['timestamp']}"
                    f"  |  lat={lat:>8}  lon={lon:>9}  alt={alt:>6} km"
                    f"  |  power={pwr:>8} W  |  BGA_1A={bga1:>6}°"
                )

                if MAX_ROWS and rows_written >= MAX_ROWS:
                    print(f"\n✅ Reached {MAX_ROWS} rows. Done.")
                    break

                time.sleep(LOG_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n⏹  Stopped by user. {rows_written} rows saved to '{OUTPUT_FILE}'.")
        finally:
            stream.disconnect()

if __name__ == "__main__":
    main()
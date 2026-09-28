import pandas as pd
import numpy as np

# --- Data Loading & Cleaning ---
raw_data = {
    "RPM": [3000, 3100, 3150, 3000, 2900, 3200, 3300, 3000],
    "Temp_C": [450, 465, 480, 455, np.nan, 510, 525, 460],
    "Vibration_mm": [1.2, 1.3, 1.5, 1.2, 1.1, 2.1, 2.5, 1.3],
    "Oil_Pressure": [40, 39, 38, 40, 41, 35, 32, 39],
    "Sensor_Error": [0, 0, 0, 0, 1, 0, 0, 0],
}
hours = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]

# ساخت دیتافریم
turbine_df = pd.DataFrame(raw_data, index=hours)

# حذف ستون اضافی
turbine_df = turbine_df.drop("Sensor_Error", axis=1)

# پر کردن داده‌های گمشده سنسور دما با دیتای قبلی
turbine_df["Temp_C"] = turbine_df["Temp_C"].ffill()


# --- Feature Engineering ---
# محاسبه شاخص تنش
turbine_df["Stress_Index"] = (turbine_df["Temp_C"] * turbine_df["Vibration_mm"]) / 1000

turbine_df["Status"] = turbine_df["Stress_Index"].apply(lambda x: "Danger" if x > 1.0 else "Normal")


# --- Analysis & Reporting ---
print("=== Critical Conditions (Danger) ===")
danger_data = turbine_df.loc[turbine_df["Status"] == "Danger"]
print(danger_data)

print("\n=== Top Stressed Hours (Descending) ===")
# مرتب‌سازی نزولی
sorted_df = turbine_df.sort_values(by="Stress_Index", ascending=False)
print(sorted_df[["Stress_Index", "Status"]].head())

print("\n=== Correlation Analysis ===")
# محاسبه همبستگی
correlation = turbine_df["RPM"].corr(turbine_df["Vibration_mm"])
print(f"Correlation between RPM and Vibration: {correlation:.2f}")

print("\n=== System Status Summary ===")
# شمارش وضعیت‌ها فقط روی ستون Status
print(turbine_df["Status"].value_counts())
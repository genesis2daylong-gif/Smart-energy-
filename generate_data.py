"""สร้างข้อมูลจำลองการใช้ไฟห้องหอพัก 1 ห้อง (รายชั่วโมง) -> energy_data.csv"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
DAYS = 75
ANOMALY_DAYS = {20, 41, 63}  # วันที่แอร์เปิดทั้งวัน (ไว้ทดสอบการแจ้งเตือน)

hours = pd.date_range("2026-08-01", periods=DAYS * 24, freq="h")
rows = []

for ts in hours:
    h, day = ts.hour, (ts - hours[0]).days
    # อุณหภูมิ: ร้อนสุดช่วงบ่าย + สุ่มเล็กน้อย
    temp = 29 + 5 * np.sin((h - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.8)
    hot = temp > 31

    # แอร์ ~1 kW: เปิดตอนกลางคืน และช่วงบ่ายถ้าร้อน
    ac_on = (h >= 21 or h <= 6) or (hot and 13 <= h <= 17 and rng.random() < 0.5)
    if day in ANOMALY_DAYS:
        ac_on = True
    ac = (0.9 + rng.normal(0, 0.05)) if ac_on else 0.0

    fan = 0.04 if (7 <= h <= 20 and rng.random() < 0.6) else 0.0
    laptop = 0.06 if (9 <= h <= 23 and rng.random() < 0.8) else 0.0
    tv = 0.10 if (19 <= h <= 22 and rng.random() < 0.4) else 0.0
    kettle = 0.15 if (h in (7, 21) and rng.random() < 0.5) else 0.0
    fridge = 0.05 + rng.normal(0, 0.005)

    for name, kwh in [("air_conditioner", ac), ("fan", fan), ("laptop", laptop),
                      ("tv", tv), ("kettle", kettle), ("fridge", fridge)]:
        rows.append((ts, name, round(max(kwh, 0), 3), round(temp, 1)))

df = pd.DataFrame(rows, columns=["datetime", "appliance", "kwh", "temperature"])
df.to_csv("energy_data.csv", index=False)
print(f"saved energy_data.csv ({len(df)} rows)")

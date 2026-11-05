"""สร้างข้อมูลจำลองห้องหอพัก 1 ห้อง ส.ค. - พ.ย. 2026 (รายชั่วโมง, 15 อุปกรณ์) -> energy_data.csv

เรื่องราวของข้อมูล: นักศึกษาค่อย ๆ ลดการใช้ไฟ (เปิดแอร์น้อยลง ปิดไฟ/ถอดปลั๊กมากขึ้น)
  - ก.ย. และ ต.ค. ปรับให้ผลรวมตรงกับบิลจริง (151 และ 127 kWh) และชั่วโมงแอร์ตรงกับที่ผู้ใช้ให้มา (8 และ 5 ชม./วัน)
  - ส.ค. และ พ.ย. เป็นข้อมูลจำลองล้วน (แอร์ ~10 และ ~4 ชม./วัน)
บันทึกเฉพาะชั่วโมงที่อุปกรณ์ทำงาน (kwh > 0) เพื่อให้ไฟล์เล็ก
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
REAL_TOTALS = {9: 151.0, 10: 127.0}          # kWh ต่อเดือน (บิลจริง)
AC_HOURS = {8: 10, 9: 8, 10: 5, 11: 4}       # ชั่วโมงแอร์เฉลี่ยต่อวัน
AC_KW = 0.22                                 # กำลังไฟแอร์เฉลี่ยตอนเปิด (kW, ประมาณ)
ANOMALY_DAYS = {"2026-09-19", "2026-10-04", "2026-10-17"}   # แอร์เปิดทั้งวัน
BASE_TEMP = {8: 29.5, 9: 28.8, 10: 28.0, 11: 27.0}
# ความน่าจะเป็นของการใช้ในแต่ละชั่วโมง/วัน ลดลงตามเดือน (นิสัยดีขึ้น)
P_FAN = {8: .80, 9: .70, 10: .50, 11: .40}
P_LIGHT = {8: .95, 9: .90, 10: .80, 11: .75}
P_TV = {8: .50, 9: .40, 10: .30, 11: .25}
P_KETTLE = {8: .60, 9: .50, 10: .45, 11: .40}
P_DRYER = {8: .40, 9: .35, 10: .30, 11: .30}
CHARGE_HOURS = {8: 7, 9: 5, 10: 4, 11: 3}    # ถอดที่ชาร์จเร็วขึ้น
PREF = [22, 23, 0, 1, 2, 3, 4, 5, 6, 21, 14, 15, 13, 16, 17, 12, 20, 19, 18, 11, 10, 9, 8, 7]

days = pd.date_range("2026-08-01", "2026-11-04")
ANOMALY_AC_HOURS = 24   # วันผิดปกติ: เปิดแอร์ทั้งวัน
# ปรับชั่วโมงแอร์ของวันปกติ ให้ค่าเฉลี่ยทั้งเดือน (รวมวันผิดปกติ) ตรงกับที่ตั้งไว้
normal_target = {}
for m, target in AC_HOURS.items():
    n_days = int((days.month == m).sum())
    n_anom = sum(1 for d in ANOMALY_DAYS if int(d[5:7]) == m)
    normal_target[m] = (target * n_days - ANOMALY_AC_HOURS * n_anom) / (n_days - n_anom)
rows = []
for day in days:
    m = day.month
    n = int(round(rng.normal(normal_target[m], 1.0)))
    ac_n = ANOMALY_AC_HOURS if day.strftime("%Y-%m-%d") in ANOMALY_DAYS else int(np.clip(n, 2, 12))
    heater_h = rng.choice([6, 20]) if rng.random() < .35 else None
    wash_h = rng.choice([10, 16]) if rng.random() < 1 / 7 else None
    iron_h = rng.choice([7, 20]) if rng.random() < .15 else None
    rice = rng.random() < .5
    for h in range(24):
        ts = day + pd.Timedelta(hours=h)
        temp = BASE_TEMP[m] + 4.5 * np.sin((h - 9) / 24 * 2 * np.pi) + rng.normal(0, .8)
        out = {}
        if h in PREF[:ac_n]:
            out["air_conditioner"] = AC_KW + rng.normal(0, .03)
        if 7 <= h <= 20 and rng.random() < P_FAN[m]:
            out["fan"] = .045
        if 18 <= h <= 23 and rng.random() < P_LIGHT[m]:
            out["led_light"] = .018
        out["fridge"] = .04 + rng.normal(0, .006)
        out["wifi_router"] = .010
        if 19 <= h <= 23 and rng.random() < P_TV[m]:
            out["tv"] = .08
        if 9 <= h <= 23 and rng.random() < .8:
            out["laptop"] = .05
        if (h >= 23 or h < CHARGE_HOURS[m] - 1) and rng.random() < .9:
            out["phone_charger"] = .010
        if h in (7, 21) and rng.random() < P_KETTLE[m]:
            out["kettle"] = .20
        if h in (12, 19) and rng.random() < .15:
            out["microwave"] = .10
        if h == 18 and rice:
            out["rice_cooker"] = .30
        if h == heater_h:
            out["water_heater"] = .52
        if h == 7 and rng.random() < P_DRYER[m]:
            out["hair_dryer"] = .10
        if h == wash_h:
            out["washing_machine"] = .50
        if h == iron_h:
            out["iron"] = .20
        for name, kwh in out.items():
            if kwh > 0:
                rows.append((ts, name, kwh, round(temp, 1)))

df = pd.DataFrame(rows, columns=["datetime", "appliance", "kwh", "temperature"])

# ก.ย./ต.ค.: ปรับแอร์ให้ผลรวมรายเดือนตรงกับบิลจริง
for month, target in REAL_TOTALS.items():
    sel = df["datetime"].dt.month == month
    is_ac = df["appliance"] == "air_conditioner"
    other = df.loc[sel & ~is_ac, "kwh"].sum()
    ac_sum = df.loc[sel & is_ac, "kwh"].sum()
    f = (target - other) / ac_sum
    df.loc[sel & is_ac, "kwh"] *= f
    print(f"เดือน {month}: ไฟอุปกรณ์อื่น {other:.1f} kWh, ตัวคูณแอร์ {f:.2f}")

df["kwh"] = df["kwh"].round(3)
df = df[df["kwh"] > 0]
df.to_csv("energy_data.csv", index=False)
print(f"saved energy_data.csv: {len(df)} rows, {df['appliance'].nunique()} appliances, "
      f"{df['datetime'].dt.normalize().nunique()} days")

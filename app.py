"""Smart Energy Saving - prototype (Streamlit)
รัน: streamlit run app.py
มี 3 โหมด: (1) กรอกอุปกรณ์ (2) กรอกหน่วยไฟรายเดือน (3) ข้อมูลรายชั่วโมงจากไฟล์ CSV
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Smart Energy Saving", page_icon="⚡", layout="wide")
st.title("⚡ AI-Powered Energy Saving Assistant")


# ====================== โหมด 1: กรอกหน่วยไฟเอง ======================
def analyze_months(raw, rate):
    """รับตารางที่ผู้ใช้กรอก -> คืนตารางสรุป + ค่าประมาณแอร์ (ถ้าคำนวณได้)"""
    df = raw.rename(columns={
        "เดือน": "month", "หน่วยไฟ (kWh)": "kwh",
        "จำนวนวัน": "days", "ชั่วโมงแอร์/วัน": "ac_h"}).copy()
    for c in ["kwh", "days", "ac_h"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["kwh", "days"])
    df = df[(df["kwh"] > 0) & (df["days"] > 0)].reset_index(drop=True)
    if df.empty:
        return None
    df["month"] = df["month"].fillna("").astype(str)
    df["per_day"] = df["kwh"] / df["days"]
    df["cost"] = df["kwh"] * rate
    df["kwh_change_pct"] = df["kwh"].pct_change() * 100

    # ประมาณ: ไฟต่อวัน = ไฟพื้นฐาน + (kWh ต่อชั่วโมงแอร์ x ชั่วโมงแอร์)
    slope = intercept = None
    fit = df.dropna(subset=["ac_h"])
    if len(fit) >= 2 and np.ptp(fit["ac_h"]) > 0:
        s, i = np.polyfit(fit["ac_h"], fit["per_day"], 1)
        if s > 0 and i >= 0:
            slope, intercept = float(s), float(i)
    return {"table": df, "slope": slope, "intercept": intercept}


def manual_mode(rate):
    st.subheader("🧮 กรอกหน่วยไฟ (kWh) ของแต่ละเดือน")
    st.caption("ดูหน่วยไฟได้จากบิลค่าไฟหรือมิเตอร์ แก้ตัวเลขในตารางได้เลย "
               "กด + ท้ายตารางเพื่อเพิ่มเดือน (ใส่ยิ่งหลายเดือน ผลยิ่งน่าเชื่อถือ)")
    default = pd.DataFrame({
        "เดือน": ["ก.ย. 2026", "ต.ค. 2026"],
        "หน่วยไฟ (kWh)": [151.0, 127.0],
        "จำนวนวัน": [30, 31],
        "ชั่วโมงแอร์/วัน": [8.0, 5.0],
    })
    edited = st.data_editor(default, num_rows="dynamic", hide_index=True, key="months")
    res = analyze_months(edited, rate)
    if res is None:
        st.info("กรอกหน่วยไฟและจำนวนวันอย่างน้อย 1 เดือนเพื่อเริ่มคำนวณ")
        return
    t, slope, intercept = res["table"], res["slope"], res["intercept"]
    last = t.iloc[-1]

    # ---- สรุปเดือนล่าสุด ----
    st.markdown(f"### สรุปเดือนล่าสุด: {last['month']}")
    c1, c2, c3 = st.columns(3)
    d_kwh = f"{last['kwh_change_pct']:+.0f}% เทียบเดือนก่อน" if len(t) > 1 else None
    c1.metric("หน่วยไฟ", f"{last['kwh']:.0f} kWh", d_kwh, delta_color="inverse")
    c2.metric("ค่าไฟโดยประมาณ", f"{last['cost']:,.0f} บาท")
    c3.metric("เฉลี่ยต่อวัน", f"{last['per_day']:.1f} kWh")

    # ---- เปรียบเทียบรายเดือน ----
    show = t[["month", "kwh", "days", "per_day", "cost"]].rename(columns={
        "month": "เดือน", "kwh": "kWh", "days": "วัน",
        "per_day": "kWh/วัน", "cost": "ค่าไฟ (บาท)"})
    st.dataframe(show.round(1), hide_index=True)
    st.plotly_chart(px.bar(t, x="month", y="kwh", text="kwh",
                           title="หน่วยไฟรายเดือน (kWh)",
                           labels={"month": "เดือน", "kwh": "kWh"}))

    # ---- แจ้งเตือน ----
    if len(t) > 1:
        prev = t.iloc[-2]
        if last["per_day"] > prev["per_day"] * 1.2:
            st.error(f"🚨 เดือนล่าสุดใช้ไฟต่อวันสูงกว่าเดือนก่อน "
                     f"{(last['per_day'] / prev['per_day'] - 1) * 100:.0f}%")
        elif last["per_day"] < prev["per_day"] * 0.9:
            st.success(f"✅ ใช้ไฟต่อวันลดลง "
                       f"{(1 - last['per_day'] / prev['per_day']) * 100:.0f}% เมื่อเทียบกับเดือนก่อน")
        else:
            st.info("การใช้ไฟต่อวันใกล้เคียงเดือนก่อน")

    st.markdown("### 🔍 แอร์กินไฟเท่าไร (ค่าประมาณ)")
    if slope is None:
        st.info("ใส่ \"ชั่วโมงแอร์/วัน\" อย่างน้อย 2 เดือนที่ไม่เท่ากัน "
                "ระบบจึงจะประมาณสัดส่วนไฟของแอร์ได้")
        st.markdown("### 🔮 คาดการณ์เดือนหน้า (ค่าเฉลี่ย)")
        pred_day = float(t["per_day"].tail(2).mean())
        days = st.number_input("จำนวนวันของเดือนหน้า", 28, 31, 30)
        st.metric("คาดการณ์", f"{pred_day * days:.0f} kWh",
                  f"≈ {pred_day * days * rate:,.0f} บาท")
        return

    ac_last = float(last["ac_h"]) if pd.notna(last["ac_h"]) else 0.0
    ac_kwh_day = slope * ac_last
    share = ac_kwh_day / last["per_day"] * 100
    a, b, c = st.columns(3)
    a.metric("แอร์ 1 ชั่วโมง ≈", f"{slope:.2f} kWh", f"{slope * rate:.1f} บาท")
    b.metric("ไฟพื้นฐานอื่น ๆ ต่อวัน", f"{intercept:.1f} kWh")
    c.metric("สัดส่วนแอร์เดือนล่าสุด", f"{share:.0f}%")
    st.caption("ประมาณจากความต่างของชั่วโมงแอร์และหน่วยไฟระหว่างเดือน "
               "(ยังไม่ได้ปรับตามอากาศ/จำนวนคน) เป็นค่าประมาณ ไม่ใช่ค่าที่วัดจริง")

    st.markdown("### 🔮 คาดการณ์เดือนหน้า")
    f1, f2 = st.columns(2)
    next_ac = f1.number_input("ชั่วโมงแอร์/วัน ที่คาดว่าจะเปิด", 0.0, 24.0, ac_last, 0.5)
    next_days = f2.number_input("จำนวนวันของเดือนหน้า", 28, 31, 30)
    pred_kwh = (intercept + slope * next_ac) * next_days
    st.metric("คาดการณ์", f"{pred_kwh:.0f} kWh", f"≈ {pred_kwh * rate:,.0f} บาท")

    st.markdown("### 💡 ถ้าลดการใช้แอร์")
    cut = st.slider("ลดแอร์ลงวันละ (ชั่วโมง)", 0.0, 4.0, 1.0, 0.5)
    cut = min(cut, ac_last)
    saved_kwh = slope * cut * 30
    st.success(f"ลดแอร์วันละ {cut:g} ชม. ประหยัดได้ประมาณ **{saved_kwh:.0f} kWh/เดือน "
               f"≈ {saved_kwh * rate:,.0f} บาท/เดือน** "
               f"(คิดเป็น {saved_kwh / (last['per_day'] * 30) * 100:.0f}% ของเดือนล่าสุด)")



# ====================== โหมด 0: กรอกเครื่องใช้ไฟฟ้า (จำนวน + ชั่วโมง/วัน) ======================
C_NAME, C_W, C_QTY, C_H, C_DUTY = ("อุปกรณ์", "กำลังไฟ (วัตต์)", "จำนวน (ชิ้น)",
                                   "ชั่วโมงใช้/วัน", "ตัวคูณการทำงาน (0-1)")

# ค่าตั้งต้นโดยประมาณของอุปกรณ์ในหอ (แก้ได้ในตาราง ควรเช็กจากป้ายหลังเครื่อง)
APPLIANCE_DEFAULTS = pd.DataFrame({
    C_NAME: ["พัดลม", "หลอดไฟ LED", "ตู้เย็น", "ทีวี", "แล็ปท็อป",
             "ที่ชาร์จมือถือ", "กาต้มน้ำ", "ไมโครเวฟ", "หม้อหุงข้าว",
             "เครื่องทำน้ำอุ่น", "ไดร์เป่าผม", "อื่น ๆ (ใส่วัตต์เอง)"],
    C_W: [50, 9, 100, 80, 50, 10, 1500, 1000, 600, 3500, 1200, 0],
    C_QTY: [1, 2, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    C_H: [10, 5, 24, 0, 6, 3, 0.2, 0, 0, 0, 0, 0],
    C_DUTY: [1, 1, 0.35, 1, 1, 1, 1, 1, 1, 1, 1, 1],
})

# ค่าประมาณทั่วไปของแอร์ (ไม่ใช่ค่าจากแหล่งทางการ ควรเช็กป้ายเครื่อง)
AC_WATTS = {"ไม่ทราบ": 900, "9,000": 800, "12,000": 1100, "18,000": 1700, "24,000": 2300}
# สัดส่วนเวลาที่คอมเพรสเซอร์ทำงานจริงโดยเฉลี่ยขณะเปิดแอร์
AC_DUTY = {"ไม่ทราบ": 0.4, "อินเวอร์เตอร์": 0.35, "ธรรมดา": 0.6}


def estimate_appliances(raw, days, bill_kwh=0.0, calibrate=True):
    """kWh/เดือน = วัตต์ x จำนวน x ตัวคูณ x ชั่วโมง/วัน x วัน / 1000
    ถ้ามียอดบิลจริง จะปรับค่าแอร์ให้ผลรวมตรงกับบิล"""
    df = raw.rename(columns={C_NAME: "name", C_W: "watt", C_QTY: "qty",
                             C_H: "h", C_DUTY: "duty"}).copy()
    df["name"] = df["name"].fillna("").astype(str)
    for c in ["watt", "qty", "h", "duty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["watt"] = df["watt"].clip(lower=0)
    df["qty"] = df["qty"].clip(lower=0)
    df["h"] = df["h"].clip(0, 24)
    df["duty"] = df["duty"].clip(0, 1)
    df = df[(df["watt"] > 0) & (df["qty"] > 0) & (df["h"] > 0)].reset_index(drop=True)
    if df.empty:
        return None

    df["kw"] = df["watt"] * df["qty"] * df["duty"] / 1000      # กำลังไฟเฉลี่ยตอนใช้งาน (kW)
    df["is_ac"] = df["name"].str.contains("แอร์")
    total_day = float((df["kw"] * df["h"]).sum())
    est_before = total_day * days

    factor, how = 1.0, None
    if bill_kwh > 0 and calibrate:
        ac_day = float((df["kw"] * df["h"])[df["is_ac"]].sum())
        other_day = total_day - ac_day
        target_day = bill_kwh / days
        if ac_day > 0 and target_day > other_day:
            factor, how = (target_day - other_day) / ac_day, "ac"
            df.loc[df["is_ac"], "kw"] *= factor
        else:
            factor, how = target_day / total_day, "all"
            df["kw"] *= factor

    df["kwh_day"] = df["kw"] * df["h"]
    df["kwh_month"] = df["kwh_day"] * days
    df["share"] = df["kwh_month"] / df["kwh_month"].sum() * 100
    return {"table": df, "est_before": est_before,
            "est_after": float(df["kwh_month"].sum()), "factor": factor, "how": how}


def appliance_mode(rate):
    st.subheader("🔌 กรอกเครื่องใช้ไฟฟ้าในห้อง")
    st.caption("ใส่ **จำนวน** และ **ชั่วโมงใช้ต่อวัน** ของแต่ละอุปกรณ์ (ไม่ต้องรู้เรื่องวัตต์) "
               "ค่าวัตต์เป็นค่าโดยประมาณทั่วไป แก้ได้ ส่วนแอร์/ตู้เย็นไม่ได้กินไฟเต็มกำลังตลอดเวลา "
               "จึงมี \"ตัวคูณการทำงาน\" ช่วยลดค่าลง กด + ท้ายตารางเพื่อเพิ่มอุปกรณ์อื่น")
    st.markdown("#### ❄️ แอร์")
    a1, a2, a3, a4 = st.columns(4)
    btu = a1.selectbox("ขนาด (BTU) ดูที่ป้ายเครื่อง", list(AC_WATTS))
    ac_type = a2.selectbox("ชนิด", list(AC_DUTY))
    ac_qty = a3.number_input("จำนวน (เครื่อง)", 0, 5, 1)
    ac_h = a4.number_input("ชั่วโมงเปิด/วัน", 0.0, 24.0, 8.0, 0.5)
    with st.expander("ปรับค่ากำลังไฟของแอร์เอง (ไม่บังคับ)"):
        ac_w = st.number_input("กำลังไฟตอนทำงาน (วัตต์)", 0, 5000, AC_WATTS[btu], 50,
                               key=f"acw_{btu}")
        ac_duty = st.number_input("ตัวคูณการทำงาน (0-1)", 0.0, 1.0, AC_DUTY[ac_type], 0.05,
                                  key=f"acd_{ac_type}")
    st.markdown("#### 🔌 เครื่องใช้ไฟฟ้าอื่น ๆ")
    edited = st.data_editor(
        APPLIANCE_DEFAULTS, num_rows="dynamic", hide_index=True, key="appl",
        column_config={
            C_NAME: st.column_config.TextColumn(C_NAME),
            C_W: st.column_config.NumberColumn(C_W, min_value=0, step=1),
            C_QTY: st.column_config.NumberColumn(C_QTY, min_value=0, step=1),
            C_H: st.column_config.NumberColumn(C_H, min_value=0, max_value=24, step=0.5),
            C_DUTY: st.column_config.NumberColumn(C_DUTY, min_value=0.0, max_value=1.0, step=0.05),
        })
    c1, c2 = st.columns(2)
    days = c1.number_input("จำนวนวันในเดือน", 28, 31, 30)
    bill = c2.number_input("หน่วยไฟในบิลจริง (kWh) ไม่บังคับ", 0.0, 3000.0, 0.0, 1.0)
    calibrate = True
    if bill > 0:
        calibrate = st.checkbox("ปรับค่าแอร์ให้ผลรวมตรงกับบิลจริง", value=True)

    ac_row = pd.DataFrame({C_NAME: ["แอร์"], C_W: [ac_w], C_QTY: [ac_qty],
                           C_H: [ac_h], C_DUTY: [ac_duty]})
    full = pd.concat([ac_row, edited], ignore_index=True)
    res = estimate_appliances(full, days, bill, calibrate)
    if res is None:
        st.info("ใส่กำลังไฟ จำนวน และชั่วโมงอย่างน้อย 1 อุปกรณ์เพื่อเริ่มคำนวณ")
        return
    t = res["table"]
    total = res["est_after"]

    # ---- สรุป ----
    st.markdown("### ผลการคำนวณ")
    m1, m2, m3 = st.columns(3)
    m1.metric("ใช้ไฟต่อเดือน (ประมาณ)", f"{total:.0f} kWh")
    m2.metric("ค่าไฟโดยประมาณ", f"{total * rate:,.0f} บาท")
    m3.metric("เฉลี่ยต่อวัน", f"{total / days:.1f} kWh")

    if bill > 0:
        err = (res["est_before"] - bill) / bill * 100
        st.info(f"เทียบกับบิลจริง {bill:.0f} kWh: ค่าที่คำนวณจากข้อมูลที่กรอก (ก่อนปรับเทียบ) "
                f"ได้ {res['est_before']:.0f} kWh ({err:+.0f}%)")
        if calibrate and res["how"] == "ac":
            st.write(f"ปรับค่าแอร์เป็น **{res['factor']:.2f} เท่า** ของที่ตั้งไว้ "
                     "เพื่อให้ผลรวมตรงกับบิล")
            if not 0.5 <= res["factor"] <= 1.6:
                st.warning("ค่าปรับต่างจากที่ตั้งไว้มาก ลองเช็กชั่วโมงใช้งานของแอร์ "
                           "หรืออุปกรณ์อื่นที่กรอกอีกครั้ง")
        elif calibrate and res["how"] == "all":
            st.warning("ปรับแอร์อย่างเดียวไม่พอ (ไม่มีแอร์ หรืออุปกรณ์อื่นรวมแล้วเกินบิล) "
                       f"จึงปรับทุกอุปกรณ์ตามสัดส่วนเป็น {res['factor']:.2f} เท่า")

    show = (t.assign(cost=t["kwh_month"] * rate)
              .sort_values("kwh_month", ascending=False)
              [["name", "qty", "h", "kwh_month", "cost", "share"]]
              .rename(columns={"name": "อุปกรณ์", "qty": "จำนวน", "h": "ชม./วัน",
                               "kwh_month": "kWh/เดือน", "cost": "บาท/เดือน",
                               "share": "สัดส่วน (%)"}))
    st.dataframe(show.round(1), hide_index=True)
    st.plotly_chart(px.pie(t, names="name", values="kwh_month",
                           title="สัดส่วนการใช้ไฟตามอุปกรณ์"))

    # ---- คำแนะนำ ----
    st.markdown("### 💡 คำแนะนำสำหรับห้องของคุณ")
    top = t.sort_values("kwh_month", ascending=False).iloc[0]
    st.write(f"• **{top['name']}** กินไฟมากที่สุด {top['share']:.0f}% "
             f"(≈ {top['kwh_month'] * rate:,.0f} บาท/เดือน) เป็นอุปกรณ์ที่ควรลดเป็นอันดับแรก")
    ac = t[t["is_ac"]]
    if not ac.empty:
        ac_row = ac.iloc[0]
        st.write(f"• แอร์ 1 ชั่วโมง ≈ {ac_row['kw']:.2f} kWh ≈ {ac_row['kw'] * rate:.1f} บาท "
                 f"ลดวันละ 1 ชั่วโมงประหยัดได้ประมาณ **{ac_row['kw'] * days * rate:,.0f} บาท/เดือน**")
        st.write("• ตั้งอุณหภูมิ 26°C และใช้ timer ช่วยลดชั่วโมงที่แอร์ทำงานได้")
    st.write("• ถอดปลั๊กอุปกรณ์ที่ไม่ใช้ เช่น ที่ชาร์จ/ทีวี เพื่อลดไฟแฝง")

    # ---- จำลอง ----
    st.markdown("### 🎛️ ถ้าลดชั่วโมงการใช้งาน")
    top3 = t.sort_values("kwh_month", ascending=False).head(3)
    cut_day_kwh = 0.0
    for i, r in top3.iterrows():
        step = 0.5 if r["h"] >= 1 else 0.1
        default = min(1.0, float(r["h"])) if r["is_ac"] else 0.0
        cut = st.slider(f"ลด {r['name']} วันละ (ชั่วโมง)", 0.0, float(r["h"]),
                        default, step, key=f"cut_{i}")
        cut_day_kwh += r["kw"] * cut
    saved = cut_day_kwh * days
    st.success(f"ประหยัดได้ประมาณ **{saved:.0f} kWh/เดือน ≈ {saved * rate:,.0f} บาท/เดือน** "
               f"({saved / total * 100:.0f}% ของทั้งหมด)")

    # ---- คาดการณ์เดือนหน้า ----
    st.markdown("### 🔮 คาดการณ์เดือนหน้า")
    nd = st.number_input("จำนวนวันของเดือนหน้า", 28, 31, 30)
    base = t["kwh_day"].sum() * nd
    plan = (t["kwh_day"].sum() - cut_day_kwh) * nd
    f1, f2 = st.columns(2)
    f1.metric("ถ้าใช้เหมือนเดิม", f"{base:.0f} kWh", f"≈ {base * rate:,.0f} บาท")
    f2.metric("ถ้าทำตามแผนลดด้านบน", f"{plan:.0f} kWh", f"≈ {plan * rate:,.0f} บาท",
              delta_color="off")
    st.caption("เป็นค่าประมาณจากข้อมูลที่กรอก ไม่ได้คำนึงถึงสภาพอากาศหรือพฤติกรรมที่เปลี่ยนไป")


# ====================== ตัวเลือกโหมด ======================
mode = st.sidebar.radio("เลือกโหมด", ["🔌 กรอกอุปกรณ์ (แนะนำ)", "🧮 กรอกค่าไฟรายเดือน",
                                       "📊 ข้อมูลรายชั่วโมง (CSV)"])
rate = st.sidebar.number_input("ค่าไฟ (บาท/หน่วย)", value=4.5, step=0.1)

if mode.startswith("🔌"):
    appliance_mode(rate)
    st.stop()
if mode.startswith("🧮"):
    manual_mode(rate)
    st.stop()


# ====================== โหมด 2: ข้อมูลรายชั่วโมง (CSV) ======================
uploaded = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV (ไม่บังคับ)", type="csv")
df = pd.read_csv(uploaded if uploaded else "energy_data.csv", parse_dates=["datetime"])

df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour

daily = (df.groupby("date")
           .agg(kwh=("kwh", "sum"), temp=("temperature", "mean"))
           .reset_index())
daily["date"] = pd.to_datetime(daily["date"])

last30 = daily.tail(30)
month_kwh = last30["kwh"].sum()
c1, c2, c3 = st.columns(3)
c1.metric("ใช้ไฟ 30 วันล่าสุด", f"{month_kwh:.0f} kWh")
c2.metric("ค่าไฟโดยประมาณ", f"{month_kwh * rate:,.0f} บาท")
c3.metric("เฉลี่ยต่อวัน", f"{last30['kwh'].mean():.1f} kWh")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 พยากรณ์ & แจ้งเตือน", "💡 คำแนะนำ"])

with tab1:
    view = st.radio("ช่วงเวลา", ["รายวัน", "รายสัปดาห์", "รายเดือน"], horizontal=True)
    rule = {"รายวัน": "D", "รายสัปดาห์": "W", "รายเดือน": "MS"}[view]
    trend = daily.set_index("date")["kwh"].resample(rule).sum().reset_index()
    st.plotly_chart(px.line(trend, x="date", y="kwh", markers=True,
                            title=f"การใช้ไฟ{view} (kWh)"))

    left, right = st.columns(2)
    by_app = df.groupby("appliance")["kwh"].sum().reset_index()
    left.plotly_chart(px.pie(by_app, names="appliance", values="kwh",
                             title="สัดส่วนไฟตามอุปกรณ์"))
    by_hour = df.groupby("hour")["kwh"].mean().reset_index()
    right.plotly_chart(px.bar(by_hour, x="hour", y="kwh",
                              title="ชั่วโมงที่ใช้ไฟสูง (เฉลี่ย kWh)"))


def make_features(d):
    return pd.DataFrame({
        "dow": d["date"].dt.dayofweek,
        "temp": d["temp"],
        "lag1": d["kwh"].shift(1),
        "lag2": d["kwh"].shift(2),
    })


feat = make_features(daily)
valid = feat.dropna().index
X, y = feat.loc[valid], daily.loc[valid, "kwh"]
X_train, y_train, X_test, y_test = X[:-7], y[:-7], X[-7:], y[-7:]

model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))

next_day = daily["date"].iloc[-1] + pd.Timedelta(days=1)
x_next = pd.DataFrame([{
    "dow": next_day.dayofweek,
    "temp": daily["temp"].tail(7).mean(),
    "lag1": daily["kwh"].iloc[-1],
    "lag2": daily["kwh"].iloc[-2],
}])
pred = model.predict(x_next)[0]

with tab2:
    a, b = st.columns(2)
    a.metric(f"พยากรณ์ {next_day:%d %b}", f"{pred:.1f} kWh", f"≈ {pred * rate:.0f} บาท")
    b.metric("ความคลาดเคลื่อนโมเดล (MAE)", f"{mae:.2f} kWh/วัน")

    thr = daily["kwh"].mean() + 2 * daily["kwh"].std()
    daily["alert"] = daily["kwh"] > thr
    fig = px.line(daily, x="date", y="kwh", title="การใช้ไฟรายวัน + จุดผิดปกติ")
    fig.add_hline(y=thr, line_dash="dash", annotation_text="เกณฑ์แจ้งเตือน")
    spikes = daily[daily["alert"]]
    fig.add_scatter(x=spikes["date"], y=spikes["kwh"], mode="markers",
                    marker=dict(color="red", size=10), name="ผิดปกติ")
    st.plotly_chart(fig)

    if daily["alert"].iloc[-7:].any():
        st.error("🚨 7 วันล่าสุดมีวันที่ใช้ไฟสูงผิดปกติ")
    else:
        st.success("✅ 7 วันล่าสุดการใช้ไฟปกติ")

with tab3:
    top = by_app.sort_values("kwh", ascending=False).iloc[0]
    share = top["kwh"] / by_app["kwh"].sum() * 100
    peak_h = int(by_hour.sort_values("kwh", ascending=False).iloc[0]["hour"])
    st.write(f"• **{top['appliance']}** กินไฟ {share:.0f}% ของทั้งหมด เป็นอุปกรณ์ที่ควรลดเป็นอันดับแรก")
    st.write(f"• ช่วงเวลาใช้ไฟสูงสุดคือประมาณ **{peak_h}:00 น.**")

    if top["appliance"] == "air_conditioner":
        ac_hours = df[(df["appliance"] == "air_conditioner") & (df["kwh"] > 0)].shape[0] / daily.shape[0]
        ac_kwh_day = top["kwh"] / daily.shape[0]
        saved = ac_kwh_day * 0.1 * 30 * rate  # สมมติ: เพิ่มอุณหภูมิ 1°C ลดไฟ ~10%
        st.write(f"• แอร์เปิดเฉลี่ย {ac_hours:.1f} ชม./วัน ลองตั้ง 26°C หรือใช้ timer "
                 f"อาจประหยัดได้ประมาณ **{saved:,.0f} บาท/เดือน** (ค่าประมาณ)")
    st.write("• ถอดปลั๊กอุปกรณ์ที่ไม่ใช้ เช่น ทีวี/ที่ชาร์จ เพื่อลดไฟแฝง")

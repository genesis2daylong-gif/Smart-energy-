"""Smart Energy Saving - prototype (Streamlit)
รัน: streamlit run app.py
มี 2 โหมด: (1) กรอกหน่วยไฟรายเดือนเอง  (2) ข้อมูลรายชั่วโมงจากไฟล์ CSV
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


# ====================== ตัวเลือกโหมด ======================
mode = st.sidebar.radio("เลือกโหมด", ["🧮 กรอกค่าไฟเอง", "📊 ข้อมูลรายชั่วโมง (CSV)"])
rate = st.sidebar.number_input("ค่าไฟ (บาท/หน่วย)", value=4.5, step=0.1)

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

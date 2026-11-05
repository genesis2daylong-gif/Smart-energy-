"""Smart Energy Saving - prototype (Streamlit)
รัน: streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Smart Energy Saving", page_icon="⚡", layout="wide")
st.title("⚡ AI-Powered Energy Saving Assistant")

# ---------- 1) โหลดข้อมูล ----------
uploaded = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV (ไม่บังคับ)", type="csv")
df = pd.read_csv(uploaded if uploaded else "energy_data.csv", parse_dates=["datetime"])
rate = st.sidebar.number_input("ค่าไฟ (บาท/หน่วย)", value=4.5, step=0.1)

df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour

daily = (df.groupby("date")
           .agg(kwh=("kwh", "sum"), temp=("temperature", "mean"))
           .reset_index())
daily["date"] = pd.to_datetime(daily["date"])

# ---------- 2) ตัวเลขสรุป + ค่าไฟ ----------
last30 = daily.tail(30)
month_kwh = last30["kwh"].sum()
c1, c2, c3 = st.columns(3)
c1.metric("ใช้ไฟ 30 วันล่าสุด", f"{month_kwh:.0f} kWh")
c2.metric("ค่าไฟโดยประมาณ", f"{month_kwh * rate:,.0f} บาท")
c3.metric("เฉลี่ยต่อวัน", f"{last30['kwh'].mean():.1f} kWh")

# ---------- 3) แดชบอร์ด ----------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 พยากรณ์ & แจ้งเตือน", "💡 คำแนะนำ"])

with tab1:
    view = st.radio("ช่วงเวลา", ["รายวัน", "รายสัปดาห์", "รายเดือน"], horizontal=True)
    rule = {"รายวัน": "D", "รายสัปดาห์": "W", "รายเดือน": "ME"}[view]
    trend = daily.set_index("date")["kwh"].resample(rule).sum().reset_index()
    st.plotly_chart(px.line(trend, x="date", y="kwh", markers=True,
                            title=f"การใช้ไฟ{view} (kWh)"), use_container_width=True)

    left, right = st.columns(2)
    by_app = df.groupby("appliance")["kwh"].sum().reset_index()
    left.plotly_chart(px.pie(by_app, names="appliance", values="kwh",
                             title="สัดส่วนไฟตามอุปกรณ์"), use_container_width=True)
    by_hour = df.groupby("hour")["kwh"].mean().reset_index()
    right.plotly_chart(px.bar(by_hour, x="hour", y="kwh",
                              title="ชั่วโมงที่ใช้ไฟสูง (เฉลี่ย kWh)"), use_container_width=True)

# ---------- 4) โมเดลพยากรณ์ ----------
def make_features(d):
    f = pd.DataFrame({
        "dow": d["date"].dt.dayofweek,
        "temp": d["temp"],
        "lag1": d["kwh"].shift(1),
        "lag2": d["kwh"].shift(2),
    })
    return f

feat = make_features(daily)
valid = feat.dropna().index
X, y = feat.loc[valid], daily.loc[valid, "kwh"]
X_train, y_train, X_test, y_test = X[:-7], y[:-7], X[-7:], y[-7:]

model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))

# พยากรณ์พรุ่งนี้
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

    # ตรวจจับความผิดปกติ: > mean + 2*std
    thr = daily["kwh"].mean() + 2 * daily["kwh"].std()
    daily["alert"] = daily["kwh"] > thr
    fig = px.line(daily, x="date", y="kwh", title="การใช้ไฟรายวัน + จุดผิดปกติ")
    fig.add_hline(y=thr, line_dash="dash", annotation_text="เกณฑ์แจ้งเตือน")
    spikes = daily[daily["alert"]]
    fig.add_scatter(x=spikes["date"], y=spikes["kwh"], mode="markers",
                    marker=dict(color="red", size=10), name="ผิดปกติ")
    st.plotly_chart(fig, use_container_width=True)

    if daily["alert"].iloc[-7:].any():
        st.error("🚨 7 วันล่าสุดมีวันที่ใช้ไฟสูงผิดปกติ")
    else:
        st.success("✅ 7 วันล่าสุดการใช้ไฟปกติ")

# ---------- 5) คำแนะนำจากข้อมูลจริงของผู้ใช้ ----------
with tab3:
    top = by_app.sort_values("kwh", ascending=False).iloc[0]
    share = top["kwh"] / by_app["kwh"].sum() * 100
    peak_h = int(by_hour.sort_values("kwh", ascending=False).iloc[0]["hour"])
    st.write(f"• **{top['appliance']}** กินไฟ {share:.0f}% ของทั้งหมด เป็นอุปกรณ์ที่ควรลดเป็นอันดับแรก")
    st.write(f"• ช่วงเวลาใช้ไฟสูงสุดคือประมาณ **{peak_h}:00 น.**")

    if top["appliance"] == "air_conditioner":
        ac_hours = df[(df["appliance"] == "air_conditioner") & (df["kwh"] > 0)].shape[0] / daily.shape[0]
        saved = ac_hours * 0.1 * 0.9 * 30 * rate  # สมมติเพิ่มอุณหภูมิ 1°C ลดไฟ ~10%
        st.write(f"• แอร์เปิดเฉลี่ย {ac_hours:.1f} ชม./วัน ลองตั้ง 26°C หรือใช้ timer "
                 f"อาจประหยัดได้ประมาณ **{saved:,.0f} บาท/เดือน** (ค่าประมาณ)")
    st.write("• ถอดปลั๊กอุปกรณ์ที่ไม่ใช้ เช่น ทีวี/ที่ชาร์จ เพื่อลดไฟแฝง")

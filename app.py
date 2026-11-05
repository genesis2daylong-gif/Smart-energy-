"""Smart Energy Saving - AI-powered energy saving assistant (prototype)
รัน: streamlit run app.py

ข้อมูลเข้าได้ 3 ทาง (กรอกเอง / อัปโหลด CSV / ข้อมูลตัวอย่าง) แต่ทุกทางถูกแปลงเป็น
"โปรไฟล์อุปกรณ์" + "ประวัติรายเดือน" แบบเดียวกัน แล้วใช้ชุดวิเคราะห์เดียวกันทั้งหมด
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Smart Energy Saving", page_icon="⚡", layout="centered")
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 760px;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px; padding: 14px 16px;}
h5 {margin-top: 1.4rem; font-weight: 600;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2E7D6B", "#5FB49C", "#9BD3C2", "#F2B84B", "#E07A5F", "#8D99AE", "#B8C4BB"]

# ---------------------------------------------------------------- ค่าตั้งต้น
C_NAME, C_QTY, C_H, C_W, C_DUTY = ("อุปกรณ์", "จำนวน (ชิ้น)", "ชั่วโมง/วัน",
                                   "กำลังไฟ (วัตต์)", "ตัวคูณการทำงาน (0-1)")
# ค่าวัตต์เป็นค่าประมาณทั่วไป (ไม่ใช่ค่าจากแหล่งทางการ) ควรเช็กป้ายหลังเครื่อง
APPLIANCE_DEFAULTS = pd.DataFrame({
    C_NAME: ["พัดลม", "หลอดไฟ LED", "ตู้เย็น", "ทีวี", "แล็ปท็อป", "ที่ชาร์จมือถือ",
             "กาต้มน้ำ", "ไมโครเวฟ", "หม้อหุงข้าว", "เครื่องทำน้ำอุ่น", "ไดร์เป่าผม",
             "อื่น ๆ (ใส่วัตต์เอง)"],
    C_QTY: [1, 2, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    C_H: [10, 5, 24, 0, 6, 3, 0.2, 0, 0, 0, 0, 0],
    C_W: [50, 9, 100, 80, 50, 10, 1500, 1000, 600, 3500, 1200, 0],
    C_DUTY: [1, 1, 0.35, 1, 1, 1, 1, 1, 1, 1, 1, 1],
})
AC_WATTS = {"ไม่ทราบ": 900, "9,000": 800, "12,000": 1100, "18,000": 1700, "24,000": 2300}
AC_DUTY = {"ไม่ทราบ": 0.4, "อินเวอร์เตอร์": 0.35, "ธรรมดา": 0.6}

M_NAME, M_KWH, M_DAYS, M_AC = "เดือน", "หน่วยไฟ (kWh)", "จำนวนวัน", "ชั่วโมงแอร์/วัน"
MONTH_DEFAULTS = pd.DataFrame({
    M_NAME: ["ก.ย. 2026", "ต.ค. 2026"], M_KWH: [151.0, 127.0],
    M_DAYS: [30, 31], M_AC: [8.0, 5.0]})

THAI = {"air_conditioner": "แอร์", "fan": "พัดลม", "fridge": "ตู้เย็น",
        "laptop": "แล็ปท็อป", "tv": "ทีวี", "kettle": "กาต้มน้ำ",
        "led_light": "หลอดไฟ LED", "phone_charger": "ที่ชาร์จมือถือ",
        "microwave": "ไมโครเวฟ", "rice_cooker": "หม้อหุงข้าว",
        "water_heater": "เครื่องทำน้ำอุ่น", "hair_dryer": "ไดร์เป่าผม",
        "wifi_router": "เราเตอร์ Wi-Fi", "washing_machine": "เครื่องซักผ้า", "iron": "เตารีด"}


# ================================================================ ชุดคำนวณกลาง
def is_ac_name(n):
    n = str(n).lower().strip()
    return "แอร์" in n or "air" in n or n == "ac"


def clean_months(raw):
    """ตารางบิลรายเดือน -> month, kwh, days, ac_h, per_day"""
    df = raw.rename(columns={M_NAME: "month", M_KWH: "kwh", M_DAYS: "days", M_AC: "ac_h"}).copy()
    for c in ["kwh", "days", "ac_h"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["kwh", "days"])
    df = df[(df["kwh"] > 0) & (df["days"] > 0)].reset_index(drop=True)
    df["month"] = df["month"].fillna("").astype(str)
    df["per_day"] = df["kwh"] / df["days"]
    return df


def clean_others(raw):
    df = raw.rename(columns={C_NAME: "name", C_QTY: "qty", C_H: "hours",
                             C_W: "watt", C_DUTY: "duty"}).copy()
    df["name"] = df["name"].fillna("").astype(str)
    for c in ["qty", "hours", "watt", "duty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["hours"] = df["hours"].clip(0, 24)
    df["duty"] = df["duty"].clip(0, 1)
    df = df[(df["qty"] > 0) & (df["hours"] > 0) & (df["watt"] > 0)].reset_index(drop=True)
    df["kw"] = df["watt"] * df["qty"] * df["duty"] / 1000
    return df


def learn_ac_kw(months, other_day):
    """เรียนรู้กำลังไฟแอร์ (kWh ต่อชั่วโมงที่เปิด) จากบิลทุกเดือนที่มีชั่วโมงแอร์
    โมเดล: ไฟต่อวัน = ไฟอุปกรณ์อื่น + k x ชั่วโมงแอร์ (กำลังสองน้อยที่สุด)"""
    if months is None or months.empty:
        return None
    m = months.dropna(subset=["ac_h"])
    m = m[m["ac_h"] > 0]
    if m.empty:
        return None
    k = float((m["ac_h"] * (m["per_day"] - other_day)).sum() / (m["ac_h"] ** 2).sum())
    return k if 0.05 <= k <= 5 else None


def check_months(months, other_day, ac_kw_before, ac_kw_after):
    """เทียบค่าที่ระบบประมาณกับบิลจริงรายเดือน"""
    m = months.dropna(subset=["ac_h"]).copy()
    if m.empty:
        return None
    m["est_before"] = (other_day + ac_kw_before * m["ac_h"]) * m["days"]
    m["est_after"] = (other_day + ac_kw_after * m["ac_h"]) * m["days"]
    m["err_before"] = (m["est_before"] - m["kwh"]) / m["kwh"] * 100
    m["err_after"] = (m["est_after"] - m["kwh"]) / m["kwh"] * 100
    return m


def manual_profile(ac_kw_preset, ac_h, others_raw, months_raw):
    """แปลงข้อมูลที่กรอกเองเป็นโปรไฟล์กลาง"""
    o = clean_others(others_raw)
    months = clean_months(months_raw) if months_raw is not None else None
    other_day = float((o["kw"] * o["hours"]).sum()) if not o.empty else 0.0
    k = learn_ac_kw(months, other_day)
    ac_kw = k if k else ac_kw_preset
    rows = []
    if ac_h > 0 and ac_kw > 0:
        rows.append({"name": "แอร์", "kw": ac_kw, "hours": float(ac_h), "is_ac": True})
    for _, r in o.iterrows():
        rows.append({"name": r["name"], "kw": r["kw"], "hours": r["hours"], "is_ac": False})
    if not rows:
        return None
    prof = pd.DataFrame(rows)
    prof["kwh_day"] = prof["kw"] * prof["hours"]
    check = (check_months(months, other_day, ac_kw_preset, ac_kw)
             if months is not None and len(months) else None)
    return {"profile": prof, "months": months, "daily": None, "check": check,
            "learned": k is not None, "ac_kw": ac_kw if ac_h > 0 else None}


def parse_csv(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ("datetime", "appliance", "kwh") if c not in df.columns]
    if missing:
        raise ValueError("ไฟล์ขาดคอลัมน์: " + ", ".join(missing) +
                         " (ต้องมี datetime, appliance, kwh ส่วน temperature ไม่บังคับ)")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df["temperature"] = (pd.to_numeric(df["temperature"], errors="coerce")
                         if "temperature" in df.columns else np.nan)
    df = df.dropna(subset=["datetime", "appliance", "kwh"])
    df = df[df["kwh"] >= 0]
    if df.empty:
        raise ValueError("อ่านข้อมูลไม่ได้ ลองตรวจรูปแบบวันที่และตัวเลขในไฟล์")
    if df["datetime"].dt.normalize().nunique() < 2:
        raise ValueError("ต้องมีข้อมูลอย่างน้อย 2 วัน")
    return df


def csv_profile(raw):
    """แปลง CSV (รายชั่วโมงหรือรายวัน) เป็นโปรไฟล์กลาง"""
    d = parse_csv(raw)
    d["name"] = d["appliance"].map(lambda a: THAI.get(str(a), str(a)))
    d["date"] = d["datetime"].dt.normalize()
    n_days = d["date"].nunique()
    hourly = d["datetime"].dt.hour.nunique() > 1

    prof = (d.groupby("name")["kwh"].sum() / n_days).rename("kwh_day").reset_index()
    if hourly:
        on = d[d["kwh"] > 0].groupby("name").size() / n_days
        prof["hours"] = prof["name"].map(on)
    else:
        prof["hours"] = np.nan
    prof["kw"] = np.where(prof["hours"] > 0, prof["kwh_day"] / prof["hours"], np.nan)
    prof["is_ac"] = prof["name"].map(is_ac_name)
    prof = prof[prof["kwh_day"] > 0].reset_index(drop=True)

    # อุณหภูมิเฉลี่ยต่อวัน: เฉลี่ยจาก "แต่ละชั่วโมง" (ไม่ถ่วงด้วยจำนวนแถวของอุปกรณ์)
    temp_day = (d.groupby("datetime")["temperature"].first()
                  .groupby(lambda x: x.normalize()).mean())
    daily = pd.DataFrame({"kwh": d.groupby("date")["kwh"].sum(),
                          "temp": temp_day}).rename_axis("date").reset_index()
    d["ym"] = d["datetime"].dt.to_period("M").astype(str)
    months = d.groupby("ym").agg(kwh=("kwh", "sum"), days=("date", "nunique")).reset_index()
    months = months.rename(columns={"ym": "month"})
    months["per_day"] = months["kwh"] / months["days"]
    if hourly:
        ac = d[d["name"].map(is_ac_name) & (d["kwh"] > 0)].groupby("ym").size()
        months["ac_h"] = months["month"].map(ac).fillna(0) / months["days"]
    else:
        months["ac_h"] = np.nan
    return {"profile": prof, "months": months, "daily": daily, "check": None,
            "learned": False, "ac_kw": None}


def sample_template():
    rows = []
    for t in pd.date_range("2026-11-01", periods=48, freq="h"):
        night = t.hour >= 22 or t.hour <= 5
        rows.append((t, "air_conditioner", 0.31 if night else 0.0, 29.0))
        rows.append((t, "fan", 0.04 if 8 <= t.hour <= 20 else 0.0, 29.0))
    return pd.DataFrame(rows, columns=["datetime", "appliance", "kwh", "temperature"])


@st.cache_data
def load_sample():
    return pd.read_csv("energy_data.csv")


# ================================================================ ส่วนแสดงผล
def style(fig, h=300):
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=h, legend_title_text="")
    return fig


def ai_tab(daily, rate):
    if daily is None or len(daily) < 30:
        st.info("ส่วนนี้ใช้ข้อมูลรายวันอย่างน้อย 30 วัน (อัปโหลด CSV รายชั่วโมง/รายวัน "
                "หรือลองข้อมูลตัวอย่าง) ข้อมูลที่กรอกเองมีเดือนละค่าเดียว จึงยังไม่พอสำหรับโมเดล AI")
        return
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    use_temp = d["temp"].notna().all()
    # ทำนาย "สัดส่วนเทียบค่าเฉลี่ย 7 วันก่อนหน้า" แทนค่า kWh ตรง ๆ เพื่อให้จับแนวโน้มที่ลดลง/เพิ่มขึ้นได้
    base = d["kwh"].shift(1).rolling(7).mean()
    feat = pd.DataFrame({"dow": d["date"].dt.dayofweek,
                         "r1": d["kwh"].shift(1) / base, "r2": d["kwh"].shift(2) / base})
    if use_temp:
        feat["temp"] = d["temp"]
    ratio = d["kwh"] / base
    ok = feat.dropna().index.intersection(ratio.dropna().index)
    X, y = feat.loc[ok], ratio.loc[ok]
    if len(X) < 14:
        st.info("ข้อมูลยังน้อยเกินไปสำหรับการพยากรณ์")
        return
    n_test = 7
    model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X[:-n_test], y[:-n_test])
    base_t = base.loc[ok].values[-n_test:]
    actual = d.loc[ok, "kwh"].values[-n_test:]
    mae = mean_absolute_error(actual, model.predict(X[-n_test:]) * base_t)
    mae_naive = mean_absolute_error(actual, base_t)   # วิธีง่าย ๆ: ใช้ค่าเฉลี่ย 7 วันก่อน
    nxt = d["date"].iloc[-1] + pd.Timedelta(days=1)
    base_next = float(d["kwh"].tail(7).mean())
    row = {"dow": nxt.dayofweek, "r1": d["kwh"].iloc[-1] / base_next,
           "r2": d["kwh"].iloc[-2] / base_next}
    if use_temp:
        row["temp"] = d["temp"].tail(7).mean()
    pred = float(model.predict(pd.DataFrame([row])[X.columns])[0]) * base_next

    a, b = st.columns(2)
    a.metric(f"พยากรณ์ {nxt:%d %b}", f"{pred:.1f} kWh", f"≈ {pred * rate:.0f} บาท", delta_color="off")
    b.metric("ความคลาดเคลื่อน (MAE)", f"{mae:.2f} kWh/วัน",
             f"{mae / d['kwh'].mean() * 100:.0f}% ของค่าเฉลี่ย", delta_color="off")
    verdict = ("ดีกว่า" if mae < mae_naive else "ยังไม่ดีกว่า")
    st.caption("Random Forest ทำนายไฟวันถัดไปจากวันในสัปดาห์ ไฟ 2 วันก่อนหน้า และอุณหภูมิ (ถ้ามี) "
               "ทดสอบกับ 7 วันสุดท้ายที่โมเดลไม่เคยเห็น เทียบกับวิธีง่าย ๆ (ใช้ค่าเฉลี่ย 7 วันก่อน) "
               f"ที่คลาดเคลื่อน {mae_naive:.2f} kWh/วัน โมเดล{verdict}วิธีง่าย ๆ ในชุดข้อมูลนี้")

    thr = d["kwh"].mean() + 2 * d["kwh"].std()
    d["flag"] = d["kwh"] > thr
    fig = px.line(d, x="date", y="kwh", color_discrete_sequence=[PALETTE[0]],
                  labels={"date": "", "kwh": "kWh/วัน"})
    fig.add_hline(y=thr, line_dash="dash", line_color="#8D99AE")
    sp = d[d["flag"]]
    fig.add_scatter(x=sp["date"], y=sp["kwh"], mode="markers", name="ผิดปกติ",
                    marker=dict(color="#E07A5F", size=10))
    st.plotly_chart(style(fig))
    if d["flag"].iloc[-7:].any():
        st.error("🚨 7 วันล่าสุดมีวันที่ใช้ไฟสูงผิดปกติ")
    else:
        st.success("✅ 7 วันล่าสุดใช้ไฟปกติ")


def render(res, rate):
    prof, months, daily = res["profile"], res["months"], res["daily"]
    total_day = float(prof["kwh_day"].sum())
    m30 = total_day * 30

    c1, c2, c3 = st.columns(3)
    c1.metric("ใช้ไฟ / เดือน", f"{m30:.0f} kWh")
    c2.metric("ค่าไฟ / เดือน", f"{m30 * rate:,.0f} บาท")
    c3.metric("เฉลี่ย / วัน", f"{total_day:.1f} kWh")
    st.caption("คิดที่ 30 วัน จากพฤติกรรมการใช้ปัจจุบัน")

    t_over, t_dev, t_save, t_ai = st.tabs(["ภาพรวม", "อุปกรณ์", "ลดค่าไฟ", "AI พยากรณ์"])

    # ---------- ภาพรวม ----------
    with t_over:
        if months is not None and len(months):
            fig = px.bar(months, x="month", y="per_day", text_auto=".1f",
                         color_discrete_sequence=[PALETTE[0]],
                         labels={"month": "", "per_day": "kWh ต่อวัน"})
            st.plotly_chart(style(fig, 260))
            show = months[["month", "kwh", "days", "per_day"]].assign(cost=months["kwh"] * rate)
            show.columns = ["เดือน", "kWh", "วัน", "kWh/วัน", "บาท"]
            st.dataframe(show.round(1), hide_index=True)
            if len(months) > 1:
                last, prev = months.iloc[-1], months.iloc[-2]
                chg = (last["per_day"] / prev["per_day"] - 1) * 100
                if chg > 20:
                    st.error(f"🚨 เดือนล่าสุดใช้ไฟต่อวันสูงกว่าเดือนก่อน {chg:.0f}%")
                elif chg < -10:
                    st.success(f"✅ ใช้ไฟต่อวันลดลง {-chg:.0f}% เมื่อเทียบกับเดือนก่อน")
                else:
                    st.info("การใช้ไฟต่อวันใกล้เคียงเดือนก่อน")
        else:
            st.info("ใส่บิลย้อนหลังเพื่อดูแนวโน้มรายเดือน")

        chk = res["check"]
        if chk is not None and len(chk):
            st.markdown("##### ความตรงกับบิลจริง")
            tbl = pd.DataFrame({"เดือน": chk["month"], "บิลจริง (kWh)": chk["kwh"],
                                "ระบบประมาณ (kWh)": chk["est_after"],
                                "ต่าง (%)": chk["err_after"]})
            st.dataframe(tbl.round(1), hide_index=True)
            if res["learned"]:
                st.caption(f"ระบบเรียนรู้จากบิลของคุณว่าแอร์ 1 ชั่วโมง ≈ {res['ac_kw']:.2f} kWh "
                           f"(ก่อนเรียนรู้คลาดเคลื่อนเฉลี่ย {chk['err_before'].abs().mean():.0f}% "
                           f"หลังเรียนรู้ {chk['err_after'].abs().mean():.0f}%) "
                           "หมายเหตุ: ใช้บิลชุดเดียวกันเรียนรู้และเทียบ จึงยังไม่ใช่การทดสอบอิสระ")
            else:
                st.caption("ใส่ชั่วโมงแอร์อย่างน้อย 1 เดือนที่มีบิล ระบบจึงจะเรียนรู้กำลังไฟแอร์ได้")

    # ---------- อุปกรณ์ ----------
    with t_dev:
        p = prof.copy()
        p["share"] = p["kwh_day"] / total_day * 100
        pie = p.loc[p["share"] >= 3, ["name", "kwh_day"]]
        rest = p.loc[p["share"] < 3, "kwh_day"].sum()
        if rest > 0:  # รวมอุปกรณ์ที่สัดส่วนน้อยเป็น "อื่น ๆ" ให้กราฟอ่านง่าย
            pie = pd.concat([pie, pd.DataFrame({"name": ["อื่น ๆ"], "kwh_day": [rest]})],
                            ignore_index=True)
        fig = px.pie(pie, names="name", values="kwh_day", hole=0.5,
                     color_discrete_sequence=PALETTE)
        fig.update_traces(textinfo="percent")
        st.plotly_chart(style(fig, 320))
        tbl = p.sort_values("kwh_day", ascending=False)
        tbl = pd.DataFrame({"อุปกรณ์": tbl["name"], "kWh/เดือน": tbl["kwh_day"] * 30,
                            "บาท/เดือน": tbl["kwh_day"] * 30 * rate, "สัดส่วน (%)": tbl["share"]})
        st.dataframe(tbl.round(1), hide_index=True)

    # ---------- ลดค่าไฟ ----------
    with t_save:
        ranked = prof.sort_values("kwh_day", ascending=False).reset_index(drop=True)
        top = ranked.iloc[0]
        st.write(f"• **{top['name']}** กินไฟมากที่สุด {top['kwh_day'] / total_day * 100:.0f}% "
                 f"(≈ {top['kwh_day'] * 30 * rate:,.0f} บาท/เดือน) ควรลดเป็นอันดับแรก")
        ac = prof[prof["is_ac"] & prof["kw"].notna()]
        if not ac.empty:
            kw = float(ac.iloc[0]["kw"])
            st.write(f"• แอร์ 1 ชั่วโมง ≈ {kw:.2f} kWh ≈ {kw * rate:.1f} บาท "
                     f"ลดวันละ 1 ชั่วโมงประหยัดได้ประมาณ **{kw * 30 * rate:,.0f} บาท/เดือน**")
            st.write("• ตั้งแอร์ 26°C และใช้ timer ช่วยลดชั่วโมงที่เครื่องทำงาน")
        st.write("• ถอดปลั๊กอุปกรณ์ที่ไม่ใช้ เพื่อลดไฟแฝง")

        st.markdown("##### ถ้าลดการใช้งาน")
        cut_day = 0.0
        for i, r in ranked.head(3).iterrows():
            if pd.notna(r["hours"]) and r["hours"] > 0 and pd.notna(r["kw"]):
                h = float(r["hours"])
                step = 0.5 if h >= 1 else 0.1
                default = min(1.0, h) if r["is_ac"] else 0.0
                cut = st.slider(f"ลด{r['name']}วันละ (ชั่วโมง)", 0.0, h, default, step,
                                key=f"cut_{i}")
                cut_day += r["kw"] * cut
            else:
                pct = st.slider(f"ลด{r['name']} (%)", 0, 100, 0, 5, key=f"cut_{i}")
                cut_day += r["kwh_day"] * pct / 100
        saved = cut_day * 30
        st.success(f"ประหยัดได้ประมาณ **{saved:.0f} kWh/เดือน ≈ {saved * rate:,.0f} บาท/เดือน** "
                   f"({saved / m30 * 100:.0f}%)")

        st.markdown("##### คาดการณ์เดือนหน้า")
        nd = st.number_input("จำนวนวันของเดือนหน้า", 28, 31, 30)
        base, plan = total_day * nd, (total_day - cut_day) * nd
        a, b = st.columns(2)
        a.metric("ถ้าใช้เหมือนเดิม", f"{base:.0f} kWh", f"≈ {base * rate:,.0f} บาท", delta_color="off")
        b.metric("ถ้าลดตามแผน", f"{plan:.0f} kWh", f"≈ {plan * rate:,.0f} บาท", delta_color="off")
        st.caption("เป็นค่าประมาณจากพฤติกรรมปัจจุบัน ไม่ได้คำนึงถึงอากาศหรือพฤติกรรมที่เปลี่ยนไป")

    # ---------- AI ----------
    with t_ai:
        ai_tab(daily, rate)


# ================================================================ หน้าแอป
st.markdown("# ⚡ Smart Energy Saving")
st.caption("รู้ว่าค่าไฟมาจากอะไร และลดได้เท่าไร")

src = st.radio("ข้อมูล", ["กรอกเอง", "อัปโหลด CSV", "ข้อมูลตัวอย่าง"],
               horizontal=True, label_visibility="collapsed")
rate = st.number_input("ค่าไฟ (บาท/หน่วย)", 0.0, 20.0, 4.5, 0.1)

result = None
if src == "กรอกเอง":
    st.markdown("##### 1 · แอร์")
    a, b, c = st.columns(3)
    btu = a.selectbox("ขนาด (BTU)", list(AC_WATTS))
    typ = b.selectbox("ชนิด", list(AC_DUTY))
    ac_h = c.number_input("ชั่วโมง/วัน", 0.0, 24.0, 8.0, 0.5)
    with st.expander("ปรับค่าแอร์เอง"):
        ac_qty = st.number_input("จำนวน (เครื่อง)", 0, 5, 1)
        ac_w = st.number_input("กำลังไฟตอนทำงาน (วัตต์)", 0, 5000, AC_WATTS[btu], 50,
                               key=f"acw_{btu}")
        ac_duty = st.number_input("ตัวคูณการทำงาน (0-1)", 0.0, 1.0, AC_DUTY[typ], 0.05,
                                  key=f"acd_{typ}")

    st.markdown("##### 2 · เครื่องใช้ไฟฟ้าอื่น ๆ")
    adv = st.toggle("แสดงกำลังไฟ (วัตต์)")
    order = [C_NAME, C_QTY, C_H] + ([C_W, C_DUTY] if adv else [])
    others = st.data_editor(
        APPLIANCE_DEFAULTS, num_rows="dynamic", hide_index=True, key="others",
        column_order=order,
        column_config={
            C_QTY: st.column_config.NumberColumn(C_QTY, min_value=0, step=1),
            C_H: st.column_config.NumberColumn(C_H, min_value=0, max_value=24, step=0.5),
            C_W: st.column_config.NumberColumn(C_W, min_value=0, step=1),
            C_DUTY: st.column_config.NumberColumn(C_DUTY, min_value=0.0, max_value=1.0, step=0.05),
        })

    st.markdown("##### 3 · บิลค่าไฟย้อนหลัง")
    st.caption("ใส่ตั้งแต่ 2 เดือนขึ้นไป ระบบจะเรียนรู้กำลังไฟแอร์ของห้องคุณ (ตารางนี้เป็นตัวอย่าง แก้เป็นของคุณได้)")
    bills = st.data_editor(
        MONTH_DEFAULTS, num_rows="dynamic", hide_index=True, key="bills",
        column_config={
            M_KWH: st.column_config.NumberColumn(M_KWH, min_value=0),
            M_DAYS: st.column_config.NumberColumn(M_DAYS, min_value=1, max_value=31, step=1),
            M_AC: st.column_config.NumberColumn(M_AC, min_value=0, max_value=24, step=0.5),
        })
    result = manual_profile(ac_w * ac_qty * ac_duty / 1000, ac_h, others, bills)
    if result is None:
        st.info("ใส่จำนวนและชั่วโมงใช้งานอย่างน้อย 1 อุปกรณ์เพื่อเริ่มคำนวณ")

elif src == "อัปโหลด CSV":
    st.caption("ไฟล์ต้องมีคอลัมน์ datetime, appliance, kwh (temperature ไม่บังคับ) "
               "เป็นข้อมูลรายชั่วโมงหรือรายวันก็ได้")
    st.download_button("ดาวน์โหลดไฟล์ตัวอย่าง", sample_template().to_csv(index=False),
                       "template.csv", "text/csv")
    up = st.file_uploader("เลือกไฟล์ CSV", type="csv", label_visibility="collapsed")
    if up is not None:
        try:
            result = csv_profile(pd.read_csv(up))
        except Exception as e:
            st.error(f"อ่านไฟล์ไม่ได้: {e}")
else:
    try:
        result = csv_profile(load_sample())
        st.caption("ข้อมูลจำลองรายชั่วโมง ส.ค.–พ.ย. 2026 ของห้องหอพัก 1 ห้อง (15 อุปกรณ์) "
                   "ก.ย./ต.ค. ปรับให้ตรงกับบิลจริง 151 และ 127 kWh ส่วน ส.ค./พ.ย. เป็นข้อมูลจำลอง")
    except Exception as e:
        st.error(f"โหลดข้อมูลตัวอย่างไม่ได้: {e}")

if result is not None:
    st.divider()
    render(result, rate)

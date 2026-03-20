import streamlit as st
import pandas as pd
import joblib

# 1. ตั้งค่าหน้าเว็บและ Theme
st.set_page_config(page_title="Bank Insight AI", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #fff5f8 !important; }
    .stButton>button {
        width: 100% !important;
        border-radius: 20px !important;
        height: 3.5em !important;
        background-color: #ff85a2 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(255, 133, 162, 0.3) !important;
    }
    .stButton>button:hover { background-color: #ffb3c1 !important; color: white !important; }
    .result-card {
        padding: 25px; border-radius: 20px; background-color: white !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align: center; border: 2px solid #ffe4e1;
    }
    h1, h2, h3 { color: #db7093 !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. โหลดโมเดล
@st.cache_resource
def load_model():
    return joblib.load('bank_model.pkl')

try:
    model = load_model()
except:
    st.error("❌ ไม่พบไฟล์โมเดล 'bank_model.pkl' กรุณาตรวจสอบบน GitHub")
    st.stop()

# 3. ส่วนหัวข้อ
st.title("🏦 Bank Insight AI")
st.subheader("ระบบวิเคราะห์โอกาสการสมัครเงินฝากออมทรัพย์พิเศษ ✨")
st.markdown("กรุณากรอกข้อมูลลูกค้าเพื่อรับการพยากรณ์จากระบบ AI")
st.divider()

# 4. ส่วนรับข้อมูล
with st.container():
    st.markdown("### 💌 ข้อมูลลูกค้า")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("อายุ (Age)", 18, 100, 30)
        
        # --- เปลี่ยนเป็น Dropdown (Selectbox) แบบที่เว็บทางการใช้กัน ---
        income_options = {
            "น้อยกว่า 30,000 บาท": 20000,
            "30,000 - 60,000 บาท": 45000,
            "60,001 - 90,000 บาท": 75000,
            "90,001 - 120,000 บาท": 105000,
            "มากกว่า 120,000 บาท": 150000
        }
        selected_income_label = st.selectbox("ช่วงรายได้ต่อปีประมาณ (Annual Income)", list(income_options.keys()))
        income = income_options[selected_income_label]
        
        balance = st.number_input("เงินในบัญชีปัจจุบัน (Account Balance)", value=1000)

    with col2:
        job_options = {
            'Management': 'Management (นักบริหาร/เจ้าหน้าที่ระดับสูง)',
            'Technician': 'Technician (ช่างเทคนิค/ผู้เชี่ยวชาญ)',
            'Services': 'Services (งานบริการ/พนักงานขาย)',
            'Retired': 'Retired (เกษียณอายุ)',
            'Admin.': 'Admin. (ธุรการ/เสมียน)',
            'Blue-collar': 'Blue-collar (พนักงานฝ่ายผลิต/แรงงาน)',
            'Entrepreneur': 'Entrepreneur (เจ้าของกิจการ)',
            'Self-employed': 'Self-employed (ธุรกิจส่วนตัว)',
            'Housemaid': 'Housemaid (งานบ้าน)',
            'Student': 'Student (นักเรียน/นักศึกษา)',
            'Unemployed': 'Unemployed (ว่างงาน)',
            'Other': 'Other (อาชีพอื่นๆ)'
        }
        selected_job_label = st.selectbox("อาชีพ", list(job_options.values()))
        job_name = [k for k, v in job_options.items() if v == selected_job_label][0]

        marital_options = {
            'Single': 'Single (โสด)',
            'Married': 'Married (แต่งงานแล้ว)',
            'Divorced': 'Divorced (หย่าร้าง/ม่าย)'
        }
        selected_marital_label = st.selectbox("สถานภาพ", list(marital_options.values()))
        marital_name = [k for k, v in marital_options.items() if v == selected_marital_label][0]

        has_card = st.radio("มีบัตรเครดิตหรือไม่?", ["Yes", "No"], horizontal=True)

# 5. ปุ่มวิเคราะห์
if st.button("🚀 เริ่มการวิเคราะห์เชิงลึก"):
    try:
        input_data = pd.DataFrame([[
            age, income, 600, balance, 1, 
            job_name, marital_name, has_card
        ]], columns=['Age', 'AnnualIncome', 'CreditScore', 'AccountBalance', 'NumContactsInCampaign', 
                     'JobTitle', 'MaritalStatus', 'HasCreditCard'])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        if prediction == 1:
            st.balloons()
            st.markdown(f'<div class="result-card"><h2 style="color: #FF1493;">💖 ผลการวิเคราะห์: สมัคร</h2><p>ลูกค้ามีโอกาสสมัครสูงมาก (ความมั่นใจ {prob[1]:.2%})</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card"><h2 style="color: #A9A9A9;">❌ ผลการวิเคราะห์: ไม่สมัคร</h2><p>ลูกค้าอาจยังไม่สนใจ (ความมั่นใจ {prob[0]:.2%})</p></div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ เกิดข้อผิดพลาด: {e}")

st.markdown("<br><hr><center><small style='color: #db7093;'>Powered by Gemini 3 Flash & Streamlit Cloud</small></center>", unsafe_allow_html=True)

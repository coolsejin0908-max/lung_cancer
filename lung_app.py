# lung_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

# ---------- 한글 폰트 설정 ----------
def set_korean_font():
    font_list = [f.name for f in font_manager.fontManager.ttflist]
    if 'NanumGothic' in font_list:
        rc('font', family='NanumGothic')
    elif 'Malgun Gothic' in font_list:
        rc('font', family='Malgun Gothic')
    else:
        rc('font', family='DejaVu Sans')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

st.set_page_config(page_title="폐암 예측 앱", layout="wide")
st.title("🫁 폐암 위험 예측 시스템")

# ---------- 1. 모델 & 스케일러 로드 ----------
@st.cache_resource
def load_model():
    model = joblib.load("lung_model.pkl")
    scaler = joblib.load("lung_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    st.success("✅ 모델과 스케일러를 불러왔습니다.")
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

# ---------- 2. 데이터 로드 (원본 CSV, 시각화용) ----------
@st.cache_data
def load_data():
    df = pd.read_csv("lung.csv")
    return df

try:
    df_raw = load_data()
    st.subheader("📂 원본 데이터 미리보기")
    st.dataframe(df_raw.head())
except:
    st.warning("lung.csv 파일이 없습니다. 예측 기능만 사용 가능합니다.")
    df_raw = None

# ---------- 3. 사이드바에서 사용자 입력 ----------
st.sidebar.header("🧑‍⚕️ 환자 정보 입력")

# 💡 여기서는 lung.csv의 컬럼명에 맞게 실제 특성들을 나열해야 합니다.
# 예시: (실제 컬럼명으로 교체 필수!)
features = {
    "나이": st.sidebar.slider("나이 (AGE)", 20, 100, 60),
    "흡연 여부": st.sidebar.selectbox("흡연 (SMOKING)", [0, 1], format_func=lambda x: "비흡연" if x==0 else "흡연"),
    "노란 손가락": st.sidebar.selectbox("노란 손가락 (YELLOW_FINGERS)", [0, 1]),
    "불안감": st.sidebar.selectbox("불안감 (ANXIETY)", [0, 1]),
    "동료 압박": st.sidebar.selectbox("동료 압박 (PEER_PRESSURE)", [0, 1]),
    "만성 질환": st.sidebar.selectbox("만성 질환 (CHRONIC_DISEASE)", [0, 1]),
    "피로": st.sidebar.selectbox("피로 (FATIGUE)", [0, 1]),
    "알러지": st.sidebar.selectbox("알러지 (ALLERGY)", [0, 1]),
    "쌕쌕거림": st.sidebar.selectbox("쌕쌕거림 (WHEEZING)", [0, 1]),
    "음주": st.sidebar.selectbox("음주 (ALCOHOL_CONSUMING)", [0, 1]),
    "기침": st.sidebar.selectbox("기침 (COUGHING)", [0, 1]),
    "숨가쁨": st.sidebar.selectbox("숨가쁨 (SHORTNESS_OF_BREATH)", [0, 1]),
    "삼킴 곤란": st.sidebar.selectbox("삼킴 곤란 (SWALLOWING_DIFFICULTY)", [0, 1]),
    "가슴 통증": st.sidebar.selectbox("가슴 통증 (CHEST_PAIN)", [0, 1])
}

# 입력된 값들을 모델 입력 형식에 맞게 배열로 변환
input_array = np.array(list(features.values())).reshape(1, -1)

# 스케일링
input_scaled = scaler.transform(input_array)

# ---------- 4. 예측 ----------
if st.sidebar.button("🔍 예측 실행"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error("🚨 **폐암 위험 높음** (Positive)")
    else:
        st.success("✅ **폐암 위험 낮음** (Negative)")

    if proba is not None:
        st.write(f"📊 확률: Positive = {proba[1]*100:.1f}% , Negative = {proba[0]*100:.1f}%")

# ---------- 5. 시각화 (원본 데이터와 비교) ----------
if df_raw is not None:
    st.subheader("📊 데이터 시각화")

    # 예: 'AGE' 컬럼과 'LUNG_CANCER' 타겟 분포
    target_col = 'LUNG_CANCER'  # 실제 타겟 컬럼명으로 변경
    if target_col in df_raw.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        df_raw[target_col].value_counts().plot(kind='bar', ax=ax, color=['green','red'])
        ax.set_title("폐암 발생 빈도")
        ax.set_xticklabels(["정상", "폐암"], rotation=0)
        st.pyplot(fig)

    # 특성 중요도 (모델이 tree 기반일 경우)
    if hasattr(model, "feature_importances_"):
        st.subheader("🧠 특성 중요도")
        importance = model.feature_importances_
        feature_names = list(features.keys())
        fig2, ax2 = plt.subplots(figsize=(8,5))
        sns.barplot(x=importance, y=feature_names, ax=ax2)
        ax2.set_title("모델이 중요하게 본 특성")
        st.pyplot(fig2)

# ---------- 6. CSV / 이미지 다운로드 (한글 파일명) ----------
if df_raw is not None:
    csv = df_raw.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 원본 데이터 CSV 다운로드", data=csv, file_name="폐암_데이터.csv", mime="text/csv")

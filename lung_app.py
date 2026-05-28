import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import io
import os
import requests

# ---------- 강력한 한글 폰트 설정 (폰트 파일 직접 등록) ----------
def register_korean_font():
    font_path = "NanumGothic.ttf"  # 사용할 폰트 파일명 (같은 디렉토리에 있어야 함)
    
    # 파일이 없으면 자동 다운로드 (인터넷 연결 필요)
    if not os.path.exists(font_path):
        st.info("한글 폰트 파일이 없습니다. 다운로드 중...")
        url = "https://github.com/marigold91/Handong_ProbStat_2019/raw/master/fonts/NanumGothic.ttf"
        try:
            r = requests.get(url)
            with open(font_path, "wb") as f:
                f.write(r.content)
            st.success("폰트 다운로드 완료!")
        except:
            st.warning("폰트 다운로드 실패. 시스템 기본 폰트를 사용합니다.")
            font_path = None
    
    if font_path and os.path.exists(font_path):
        # 폰트 등록
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
    else:
        # 시스템에서 한글 폰트 찾기
        font_list = [f.name for f in font_manager.fontManager.ttflist]
        for font in ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Gulim', 'Dotum']:
            if font in font_list:
                rc('font', family=font)
                break
        else:
            rc('font', family='DejaVu Sans')
    
    plt.rcParams['axes.unicode_minus'] = False

register_korean_font()

# ---------- 나머지 코드 (이전과 동일, UI 개선 포함) ----------
st.set_page_config(page_title="🧬 AI 환자 군집 분석", layout="wide", initial_sidebar_state="expanded")

# 커스텀 CSS (생략 가능하지만 유지)
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton > button { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 12px; }
    .stButton > button:hover { background-color: #45a049; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 AI 기반 환자 군집 분석 시스템")
st.markdown("#### 흡연·음주·나이 정보로 환자 유형을 예측하고 맞춤 조언을 받으세요")

# 모델 로드
@st.cache_resource
def load_model_scaler():
    try:
        model = joblib.load("lung_model.pkl")
        scaler = joblib.load("lung_scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"모델/스케일러 로드 실패: {e}")
        st.stop()

model, scaler = load_model_scaler()
st.success("✅ AI 모델이 준비되었습니다", icon="🤖")

# 데이터 로드
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("lung.csv")
        required = ['나이', '흡연 정도', '음주 정도', 'cluster']
        if all(col in df.columns for col in required):
            return df[required].dropna()
        else:
            st.warning("CSV에 필요한 컬럼이 없습니다. 더미 데이터를 생성합니다.")
            return create_dummy_data()
    except FileNotFoundError:
        st.info("lung.csv 파일이 없어 예제 데이터를 생성합니다.")
        return create_dummy_data()

def create_dummy_data():
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        '나이': np.random.randint(20, 80, n),
        '흡연 정도': np.random.randint(0, 10, n),
        '음주 정도': np.random.randint(0, 10, n),
        'cluster': np.random.randint(0, 3, n)
    })

df = load_data()

# 사이드바 입력
st.sidebar.header("📝 환자 정보 입력")
smoking = st.sidebar.slider("🚬 흡연 정도", 0.0, 10.0, 3.0, 0.5)
alcohol = st.sidebar.slider("🍺 음주 정도", 0.0, 10.0, 2.0, 0.5)
age = st.sidebar.slider("🎂 나이", 18, 100, 45, 1)

input_df = pd.DataFrame([[smoking, alcohol, age]], columns=['흡연 정도', '음주 정도', '나이'])
input_scaled = scaler.transform(input_df)

# 예측 버튼
if st.sidebar.button("🔮 환자 군집 예측", use_container_width=True):
    pred_cluster = model.predict(input_scaled)[0]
    
    # 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📊 흡연 정도", f"{smoking} / 10")
    with col2: st.metric("🍷 음주 정도", f"{alcohol} / 10")
    with col3: st.metric("🎂 나이", f"{age}세")
    with col4:
        cluster_name = {0: "🟢 저위험군", 1: "🟡 중간위험군", 2: "🔴 고위험군"}
        st.metric("🏷️ 예측 군집", cluster_name.get(pred_cluster, f"{pred_cluster}번"))
    
    # AI 조언
    st.subheader("💡 AI 건강 조언")
    if pred_cluster == 0:
        st.success("✅ **저위험군**\n- 건강한 생활습관 유지 중입니다.\n- 정기 검진과 꾸준한 운동을 권장합니다.")
    elif pred_cluster == 1:
        st.warning("⚠️ **중간위험군**\n- 흡연/음주 습관 개선이 필요합니다.\n- 금연 상담 및 하루 30분 걷기를 시작하세요.")
    else:
        st.error("🚨 **고위험군**\n- 폐암 위험이 높습니다.\n- 지금 즉시 금연하고 저선량 CT 검진을 받으세요.")
    
    # 시각화
    st.subheader("📊 군집 시각화 (기존 환자 데이터)")
    viz_option = st.radio("표시할 축 선택", ["흡연 정도 vs 음주 정도", "나이 vs 흡연 정도", "나이 vs 음주 정도"], horizontal=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    if viz_option == "흡연 정도 vs 음주 정도":
        x, y = '흡연 정도', '음주 정도'
        new_x, new_y = smoking, alcohol
    elif viz_option == "나이 vs 흡연 정도":
        x, y = '나이', '흡연 정도'
        new_x, new_y = age, smoking
    else:
        x, y = '나이', '음주 정도'
        new_x, new_y = age, alcohol
    
    scatter = ax.scatter(df[x], df[y], c=df['cluster'], cmap='viridis', alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    ax.scatter(new_x, new_y, c='red', s=300, marker='X', edgecolors='white', linewidth=2, label='🔴 새로운 환자', zorder=5)
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.set_title(f"환자 군집 분포 ({x} vs {y})", fontsize=14)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='군집')
    st.pyplot(fig)
    
    # 그래프 이미지 다운로드
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button("📸 그래프 이미지 다운로드 (PNG)", data=buf, file_name="군집_분석_그래프.png", mime="image/png")

else:
    st.info("👈 왼쪽 사이드바에서 환자 정보를 입력하고 예측 버튼을 눌러주세요.")

# 데이터 미리보기
st.markdown("---")
with st.expander("📂 기존 환자 데이터 미리보기 (군집 포함)"):
    st.dataframe(df.head(10), use_container_width=True)
    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 데이터 CSV 다운로드", data=csv_data, file_name="환자_군집_데이터.csv", mime="text/csv")

st.caption("© AI 환자 군집 분석 | 의학적 진단이 아닌 참고용입니다.")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import io

# ---------- 한글 폰트 설정 (Streamlit Cloud + packages.txt) ----------
def set_korean_font():
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    try:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
    except:
        rc('font', family='DejaVu Sans')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# ---------- 페이지 설정 ----------
st.set_page_config(page_title="🧬 AI 환자 군집 분석", layout="wide")
st.title("🧬 AI 기반 환자 군집 분석 시스템")
st.markdown("#### 흡연·음주·나이 정보로 환자 유형을 예측하고 맞춤 조언을 받으세요")

# ---------- 모델 & 스케일러 로드 ----------
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
st.success("✅ AI 모델이 준비되었습니다")

# ---------- 데이터 로드 (시각화용) ----------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("lung.csv")
        required = ['나이', '흡연 정도', '음주 정도', 'cluster']
        if all(col in df.columns for col in required):
            return df[required].dropna()
    except:
        pass
    # 더미 데이터 생성
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        '나이': np.random.randint(20, 80, n),
        '흡연 정도': np.random.randint(0, 10, n),
        '음주 정도': np.random.randint(0, 10, n),
        'cluster': np.random.randint(0, 3, n)
    })

df = load_data()

# ---------- 사이드바 입력 (항상 표시) ----------
st.sidebar.header("📝 환자 정보 입력")
smoking = st.sidebar.slider("🚬 흡연 정도", 0.0, 10.0, 3.0, 0.5)
alcohol = st.sidebar.slider("🍺 음주 정도", 0.0, 10.0, 2.0, 0.5)
age = st.sidebar.slider("🎂 나이", 18, 100, 45, 1)

# ---------- 라디오 버튼 (항상 표시) ----------
st.subheader("📊 군집 시각화 (축 선택)")
viz_option = st.radio(
    "비교할 축을 선택하세요",
    ["흡연 정도 vs 음주 정도", "나이 vs 흡연 정도", "나이 vs 음주 정도"],
    horizontal=True,
    key="viz_radio"
)

# 선택된 축에 따라 그래프 그리기 (새 환자 미표시, 기존 군집만)
fig, ax = plt.subplots(figsize=(8, 6))

if viz_option == "흡연 정도 vs 음주 정도":
    x, y = '흡연 정도', '음주 정도'
elif viz_option == "나이 vs 흡연 정도":
    x, y = '나이', '흡연 정도'
else:
    x, y = '나이', '음주 정도'

# 기존 환자 산점도 (군집별 색상)
scatter = ax.scatter(df[x], df[y], c=df['cluster'], cmap='viridis', alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
ax.set_xlabel(x, fontsize=12)
ax.set_ylabel(y, fontsize=12)
ax.set_title(f"기존 환자 군집 분포 ({x} vs {y})", fontsize=14)
plt.colorbar(scatter, ax=ax, label='군집')

st.pyplot(fig)

# ---------- 예측 버튼 (새 환자 표시) ----------
st.markdown("---")
if st.sidebar.button("🔮 환자 군집 예측", use_container_width=True):
    input_df = pd.DataFrame([[smoking, alcohol, age]], columns=['흡연 정도', '음주 정도', '나이'])
    input_scaled = scaler.transform(input_df)
    pred_cluster = model.predict(input_scaled)[0]
    
    # 결과 메트릭
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📊 흡연 정도", f"{smoking} / 10")
    with col2: st.metric("🍷 음주 정도", f"{alcohol} / 10")
    with col3: st.metric("🎂 나이", f"{age}세")
    with col4:
        cluster_map = {0: "🟢 저위험군", 1: "🟡 중간위험군", 2: "🔴 고위험군"}
        st.metric("🏷️ 예측 군집", cluster_map.get(pred_cluster, f"{pred_cluster}번"))
    
    # AI 조언
    st.subheader("💡 AI 건강 조언")
    if pred_cluster == 0:
        st.success("✅ **저위험군**\n- 건강한 생활습관 유지 중입니다.\n- 정기 검진과 꾸준한 운동을 권장합니다.")
    elif pred_cluster == 1:
        st.warning("⚠️ **중간위험군**\n- 흡연/음주 습관 개선이 필요합니다.\n- 금연 상담 및 하루 30분 걷기를 시작하세요.")
    else:
        st.error("🚨 **고위험군**\n- 폐암 위험이 높습니다.\n- 지금 즉시 금연하고 저선량 CT 검진을 받으세요.")
    
    # 현재 선택된 축에 새 환자 추가하여 그래프 다시 그리기
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    if viz_option == "흡연 정도 vs 음주 정도":
        x2, y2 = '흡연 정도', '음주 정도'
        new_x, new_y = smoking, alcohol
    elif viz_option == "나이 vs 흡연 정도":
        x2, y2 = '나이', '흡연 정도'
        new_x, new_y = age, smoking
    else:
        x2, y2 = '나이', '음주 정도'
        new_x, new_y = age, alcohol
    
    ax2.scatter(df[x2], df[y2], c=df['cluster'], cmap='viridis', alpha=0.6, s=60, edgecolors='black')
    ax2.scatter(new_x, new_y, c='red', s=300, marker='X', edgecolors='white', linewidth=2, label='새 환자')
    ax2.set_xlabel(x2)
    ax2.set_ylabel(y2)
    ax2.set_title(f"새 환자 포함 군집 분포 ({x2} vs {y2})")
    ax2.legend()
    plt.colorbar(ax2.collections[0], ax=ax2, label='군집')
    st.pyplot(fig2)
    
    # 그래프 이미지 다운로드
    buf = io.BytesIO()
    fig2.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button("📸 예측 결과 그래프 다운로드", data=buf, file_name="군집_예측_결과.png", mime="image/png")

# ---------- 데이터 미리보기 및 다운로드 ----------
with st.expander("📂 기존 환자 데이터 미리보기 (군집 포함)"):
    st.dataframe(df.head(10), use_container_width=True)
    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 CSV 다운로드", data=csv_data, file_name="환자_군집_데이터.csv", mime="text/csv")

st.caption("의학적 진단이 아닌 AI 참고용입니다.")

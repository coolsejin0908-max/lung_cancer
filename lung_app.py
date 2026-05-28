# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from matplotlib import font_manager, rc

# ------------------ 한글 폰트 설정 ------------------
def set_korean_font():
    """Streamlit Cloud 환경에서 한글 폰트 설정"""
    # 시스템에 존재하는 한글 폰트 찾기 (NanumGothic, Malgun Gothic 등)
    font_list = [f.name for f in font_manager.fontManager.ttflist]
    if 'NanumGothic' in font_list:
        rc('font', family='NanumGothic')
    elif 'Malgun Gothic' in font_list:
        rc('font', family='Malgun Gothic')
    else:
        # 기본 폰트로 대체 (일부 한자 깨짐 감수)
        rc('font', family='DejaVu Sans')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# ------------------ 페이지 설정 ------------------
st.set_page_config(page_title="환자 군집 시각화", layout="wide")
st.title("📊 흡연 정도 vs 나이 군집 시각화")

# ------------------ 데이터 로드 ------------------
uploaded_file = st.file_uploader("CSV 파일 업로드 (흡연 정도, 나이, cluster 열 필요)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_cols = ['흡연 정도', '나이', 'cluster']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV 파일에 {required_cols} 열이 모두 있어야 합니다.")
        st.stop()
else:
    st.info("예제 데이터를 생성합니다. '흡연 정도', '나이', 'cluster' 열이 포함된 CSV를 업로드하세요.")
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        '흡연 정도': np.random.randint(0, 4, n),
        '나이': np.random.randint(20, 70, n),
        'cluster': np.random.randint(0, 3, n)
    })

st.subheader("📂 데이터 미리보기")
st.dataframe(df.head())

# ------------------ 데이터 다운로드 (한글 파일명) ------------------
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')  # 한글 깨짐 방지

csv_data = convert_df_to_csv(df)
st.download_button(
    label="📥 데이터 CSV 다운로드 (한글 파일명)",
    data=csv_data,
    file_name="환자_데이터.csv",   # 한글 파일명
    mime="text/csv"
)

# ------------------ 새 환자 입력 ------------------
st.sidebar.header("➕ 새 환자 정보 입력")
new_smoking = st.sidebar.slider("흡연 정도", min_value=0, max_value=3, value=1, step=1)
new_age = st.sidebar.slider("나이", min_value=18, max_value=100, value=50, step=1)

# ------------------ 시각화 ------------------
fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(df['흡연 정도'], df['나이'], c=df['cluster'], cmap='viridis', alpha=0.5)

# 새 환자 표시
ax.scatter(new_smoking, new_age, c='black', s=300, marker='X', label='새 환자')

ax.set_xlabel('흡연 정도')
ax.set_ylabel('나이')
ax.set_title('흡연 정도 vs 나이 (군집 색상)')
ax.legend()
plt.colorbar(scatter, ax=ax, label='cluster')

st.pyplot(fig)

# ------------------ 그래프 이미지 다운로드 (한글 파일명) ------------------
buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
buf.seek(0)
st.download_button(
    label="📸 그래프 이미지 다운로드 (PNG, 한글 파일명)",
    data=buf,
    file_name="흡연_나이_군집_그래프.png",
    mime="image/png"
)

st.caption(f"⭐ 현재 표시된 새 환자: 흡연 정도 = {new_smoking}, 나이 = {new_age}")

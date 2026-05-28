# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 페이지 설정
st.set_page_config(page_title="환자 군집 시각화", layout="wide")
st.title("📊 흡연 정도 vs 나이 군집 시각화")

# 1. 데이터 로드 (CSV 업로드 또는 예제 데이터)
uploaded_file = st.file_uploader("CSV 파일 업로드 (흡연 정도, 나이, cluster 열 필요)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_cols = ['흡연 정도', '나이', 'cluster']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV 파일에 {required_cols} 열이 모두 있어야 합니다.")
        st.stop()
else:
    # 예제 데이터 생성 (실제 사용 시 삭제 또는 대체 가능)
    st.info("예제 데이터를 생성하여 보여줍니다. '흡연 정도', '나이', 'cluster' 열이 포함된 CSV를 업로드하세요.")
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        '흡연 정도': np.random.randint(0, 4, n),
        '나이': np.random.randint(20, 70, n),
        'cluster': np.random.randint(0, 3, n)
    })

st.subheader("📂 데이터 미리보기")
st.dataframe(df.head())

# 2. 새 환자 입력 받기
st.sidebar.header("➕ 새 환자 정보 입력")
new_smoking = st.sidebar.slider("흡연 정도", min_value=0, max_value=3, value=1, step=1)
new_age = st.sidebar.slider("나이", min_value=18, max_value=100, value=50, step=1)

# 3. 시각화 (Matplotlib + Streamlit)
fig, ax = plt.subplots(figsize=(8, 6))

# 기존 환자 산점도 (군집별 색상)
scatter = ax.scatter(df['흡연 정도'], df['나이'], c=df['cluster'], cmap='viridis', alpha=0.5)

# 새 환자 표시 (검은색 X)
ax.scatter(new_smoking, new_age, c='black', s=300, marker='X', label='새 환자')

ax.set_xlabel('흡연 정도')
ax.set_ylabel('나이')
ax.set_title('흡연 정도 vs 나이 (군집 색상)')
ax.legend()
plt.colorbar(scatter, ax=ax, label='cluster')

st.pyplot(fig)

# 4. 추가 정보 (선택 사항)
st.caption(f"⭐ 현재 표시된 새 환자: 흡연 정도={new_smoking}, 나이={new_age}")

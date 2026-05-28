import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import io

# ------------------ 한글 폰트 설정 (Streamlit Cloud 대응) ------------------
def set_korean_font():
    font_list = [f.name for f in font_manager.fontManager.ttflist]
    # 우선순위 한글 폰트 목록
    korean_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Gulim', 'Dotum']
    for font in korean_fonts:
        if font in font_list:
            rc('font', family=font)
            break
    else:
        rc('font', family='DejaVu Sans')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# ------------------ 페이지 설정 ------------------
st.set_page_config(page_title="환자 군집 예측 시스템", layout="wide")
st.title("🧬 환자 군집 예측 시스템")
st.markdown("#### 흡연 정도, 음주 정도, 나이를 입력하면 환자가 속할 군집을 알려드립니다.")

# ------------------ 1. 모델 & 스케일러 로드 ------------------
@st.cache_resource
def load_model_scaler():
    try:
        model = joblib.load("lung_model.pkl")
        scaler = joblib.load("lung_scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"모델 또는 스케일러 파일을 불러올 수 없습니다: {e}")
        st.stop()

model, scaler = load_model_scaler()
st.success("✅ 모델과 스케일러가 준비되었습니다.")

# ------------------ 2. 예제 데이터 로드 (시각화용) ------------------
@st.cache_data
def load_sample_data():
    # 예제 데이터 (실제 lung.csv가 없으면 더미 생성)
    try:
        df = pd.read_csv("lung.csv")
        # 필요한 컬럼만 선택 (나이, 흡연 정도, 음주 정도, cluster)
        if {'나이', '흡연 정도', '음주 정도', 'cluster'}.issubset(df.columns):
            return df[['나이', '흡연 정도', '음주 정도', 'cluster']].dropna()
        else:
            st.warning("CSV 파일에 '나이', '흡연 정도', '음주 정도', 'cluster' 컬럼이 없습니다. 더미 데이터를 생성합니다.")
            return create_dummy_data()
    except FileNotFoundError:
        st.info("lung.csv 파일이 없어 예제 데이터를 생성합니다.")
        return create_dummy_data()

def create_dummy_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        '나이': np.random.randint(20, 80, n),
        '흡연 정도': np.random.randint(0, 10, n),
        '음주 정도': np.random.randint(0, 10, n),
        'cluster': np.random.randint(0, 3, n)
    })
    return df

df_sample = load_sample_data()

# ------------------ 3. 사이드바 - 새 환자 입력 ------------------
st.sidebar.header("📝 새 환자 정보 입력")
smoking = st.sidebar.number_input("흡연 정도 (0~10)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
alcohol = st.sidebar.number_input("음주 정도 (0~10)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
age = st.sidebar.number_input("나이 (세)", min_value=0.0, max_value=120.0, value=45.0, step=1.0)

# 입력 데이터를 모델 형식에 맞게 DataFrame으로 변환
input_df = pd.DataFrame([[smoking, alcohol, age]], columns=['흡연 정도', '음주 정도', '나이'])

# 스케일링 (모델이 학습할 때 사용한 스케일러 적용)
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"스케일링 오류: {e}\n입력 특성 개수가 모델과 일치하는지 확인하세요.")
    st.stop()

# ------------------ 4. 예측 및 결과 표시 ------------------
if st.sidebar.button("🔮 군집 예측하기", type="primary"):
    pred_cluster = model.predict(input_scaled)[0]
    
    # 결과를 예쁘게 표시
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("흡연 정도", f"{smoking}")
    with col2:
        st.metric("음주 정도", f"{alcohol}")
    with col3:
        st.metric("나이", f"{age}세")
    
    st.markdown(f"## 🎯 예측 결과: **{pred_cluster}번 군집**")
    
    # 군집별 특성 정보 (선택 사항)
    cluster_desc = {0: "저위험군", 1: "중간위험군", 2: "고위험군"}
    st.info(f"이 환자는 **{cluster_desc.get(pred_cluster, '알 수 없음')}** 으로 분류되었습니다.")
    
    # 시각화: 기존 데이터 위에 새 환자 표시
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 산점도 1: 나이 vs 흡연 정도
    ax1 = axes[0]
    scatter = ax1.scatter(df_sample['나이'], df_sample['흡연 정도'], 
                          c=df_sample['cluster'], cmap='viridis', alpha=0.6, s=50)
    ax1.scatter(age, smoking, c='red', s=200, marker='X', edgecolors='black', linewidth=2, label='새 환자')
    ax1.set_xlabel('나이')
    ax1.set_ylabel('흡연 정도')
    ax1.set_title('나이 vs 흡연 정도 (군집 색상)')
    ax1.legend()
    
    # 산점도 2: 나이 vs 음주 정도
    ax2 = axes[1]
    scatter2 = ax2.scatter(df_sample['나이'], df_sample['음주 정도'], 
                           c=df_sample['cluster'], cmap='viridis', alpha=0.6, s=50)
    ax2.scatter(age, alcohol, c='red', s=200, marker='X', edgecolors='black', linewidth=2, label='새 환자')
    ax2.set_xlabel('나이')
    ax2.set_ylabel('음주 정도')
    ax2.set_title('나이 vs 음주 정도 (군집 색상)')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 그래프 이미지 다운로드 버튼
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button("📸 그래프 이미지 다운로드 (PNG)", data=buf, 
                       file_name="군집_예측_그래프.png", mime="image/png")

# ------------------ 5. 데이터 미리보기 및 다운로드 ------------------
st.markdown("---")
st.subheader("📊 기존 환자 데이터 미리보기 (군집 정보 포함)")
st.dataframe(df_sample.head(10))

# CSV 다운로드 (한글 파일명)
csv_data = df_sample.to_csv(index=False).encode('utf-8-sig')
st.download_button("📥 데이터 CSV 다운로드", data=csv_data, 
                   file_name="환자_군집_데이터.csv", mime="text/csv")

# ------------------ 6. 추가 설명 ------------------
st.markdown("---")
st.caption("💡 모델은 '흡연 정도', '음주 정도', '나이'를 입력받아 미리 학습된 군집을 예측합니다.")

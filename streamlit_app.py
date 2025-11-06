import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans # 클러스터링을 위해 import

# --- 1. 앱 설정 ---
st.set_page_config(
    page_title="HUFS Data Dashboard",
    page_icon="🎓",
    layout="wide"
)

# --- 2. 다국어 지원 텍스트 (i18n) ---
# 모든 UI 텍스트를 이곳에서 관리합니다.
TEXTS = {
    'ko': {
        'lang_select': '언어 선택',
        'prof': '담당교수: 이동현',
        'school': 'Social Science & AI융합학부',
        'course': '산업데이터시각화',
        'filter_header': '데이터 필터',
        'hour_slider': '시간 선택:',
        'k_slider_label': '클러스터 개수 (K):',
        'k_slider_help': 'K=1은 클러스터링을 사용하지 않습니다. 2 이상을 선택하면 K-Means 클러스터링을 실행합니다.',
        'show_data_label': '필터링된 원본 데이터 보기',
        'main_title': '🚕 뉴욕시 Uber 픽업 데이터 실시간 분석',
        'main_desc': "이 앱은 '산업데이터시각화' 수업을 위한 Streamlit 대시보드 예제입니다. (다국어 및 클러스터링 지원)",
        'loading_text': '데이터 로딩 중... (약 10만 건)',
        'cluster_loading_text': '픽업 위치 클러스터링 중...',
        'map_subheader_suffix': '시간대의 Uber 픽업 맵',
        'pickup_count': '총 픽업 건수',
        'no_data_warn': '해당 시간에 데이터가 없습니다.',
        'hist_subheader': '시간대별 전체 픽업 횟수',
        'raw_data_subheader': '원본 데이터 (필터됨)',
        'data_load_error': '데이터 로딩 중 오류 발생'
    },
    'en': {
        'lang_select': 'Language',
        'prof': 'Professor: Donghyun Lee',
        'school': 'Division of Social Science & AI',
        'course': 'Industrial Data Visualization',
        'filter_header': 'Data Filters',
        'hour_slider': 'Select Hour:',
        'k_slider_label': 'Number of Clusters (K):',
        'k_slider_help': 'K=1 means no clustering. Select 2 or more to run K-Means clustering.',
        'show_data_label': 'Show filtered raw data',
        'main_title': '🚕 NYC Uber Pickups Real-time Analysis',
        'main_desc': 'This app is a Streamlit dashboard example for the "Industrial Data Visualization" class. (Multilingual & Clustering supported)',
        'loading_text': 'Loading data... (approx. 100k rows)',
        'cluster_loading_text': 'Clustering pickup locations...',
        'map_subheader_suffix': 'Uber Pickups Map',
        'pickup_count': 'Total Pickups',
        'no_data_warn': 'No data available for this hour.',
        'hist_subheader': 'Total Pickups by Hour',
        'raw_data_subheader': 'Raw Data (Filtered)',
        'data_load_error': 'Error loading data'
    }
}

# --- 3. 세션 상태 초기화 (언어 설정) ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'ko' # 기본값은 한국어

# --- 4. 헬퍼 함수 (데이터 로딩) ---
@st.cache_data
def load_data(nrows):
    DATA_URL = "https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
    try:
        data = pd.read_csv(DATA_URL, nrows=nrows)
        data.rename(lambda x: str(x).lower(), axis='columns', inplace=True)
        data['date/time'] = pd.to_datetime(data['date/time'])
        data['hour'] = data['date/time'].dt.hour
        # st.map은 'lat', 'lon' 컬럼명이 필요합니다.
        data = data.rename(columns={'lat': 'lat', 'lon': 'lon'})
        return data
    except Exception as e:
        st.error(f"{TEXTS[st.session_state.lang]['data_load_error']}: {e}")
        return pd.DataFrame()

# 클러스터링을 위한 색상 리스트 (최대 10개)
CLUSTER_COLORS = [
    "#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#00FFFF",
    "#FF00FF", "#C0C0C0", "#800000", "#008000", "#000080"
]

# --- 5. 사이드바 UI ---
with st.sidebar:
    # 언어 선택
    lang_options = {'한국어': 'ko', 'English': 'en'}
    selected_lang_str = st.radio(
        label=TEXTS['ko']['lang_select'], # 라벨은 고정
        options=lang_options.keys(),
        horizontal=True,
    )
    st.session_state.lang = lang_options[selected_lang_str]
    lang = st.session_state.lang # 편의를 위해 변수 할당

    # 로고 및 수업 정보
    LOGO_URL = "https://www.hufs.ac.kr/sites/hufs/images/sub/simbol_list3.png"
    st.image(LOGO_URL)
    
    st.title("수업 정보")
    st.markdown(
        f"""
        - **대학교:** 한국외국어대학교 (HUFS)
        - **학부:** {TEXTS[lang]['school']}
        - **수업:** {TEXTS[lang]['course']}
        - **{TEXTS[lang]['prof']}** """
    )
    
    st.divider() 
    
    st.header(TEXTS[lang]['filter_header'])
    
    # 시간 필터
    hour_to_filter = st.slider(
        TEXTS[lang]['hour_slider'], 
        0, 23, 17
    )
    
    # 클러스터 개수(K) 필터
    k_clusters = st.slider(
        TEXTS[lang]['k_slider_label'],
        min_value=1,
        max_value=10,
        value=1, # 기본값 1 (클러스터링 없음)
        help=TEXTS[lang]['k_slider_help']
    )
    
    # 원본 데이터 보기
    show_raw_data = st.checkbox(TEXTS[lang]['show_data_label'])

# --- 6. 메인 화면 ---

# 현재 언어 설정(lang)에 따라 텍스트를 가져옵니다.
lang = st.session_state.lang

st.title(TEXTS[lang]['main_title'])
st.markdown(TEXTS[lang]['main_desc'])

# 데이터 로딩
with st.spinner(TEXTS[lang]['loading_text']):
    data = load_data(100000)

if not data.empty:
    # 시간 필터링
    filtered_data = data[data['hour'] == hour_to_filter].copy() # .copy()로 Warning 방지

    # 맵 제목
    st.subheader(f"{hour_to_filter}:00 {TEXTS[lang]['map_subheader_suffix']}")
    st.write(f"{TEXTS[lang]['pickup_count']}: **{len(filtered_data)}**")

    # 맵 시각화 (클러스터링 포함)
    if not filtered_data.empty:
        if k_clusters > 1:
            # K=2 이상이면 K-Means 클러스터링 실행
            with st.spinner(TEXTS[lang]['cluster_loading_text']):
                # 위도(lat)와 경도(lon)를 기반으로 클러스터링
                kmeans = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
                filtered_data['cluster'] = kmeans.fit_predict(filtered_data[['lat', 'lon']])
                
                # 클러스터 번호에 따라 색상 매핑
                # (10개가 넘는 클러스터는 색상이 반복됩니다)
                filtered_data['color'] = filtered_data['cluster'].apply(
                    lambda x: CLUSTER_COLORS[x % len(CLUSTER_COLORS)]
                )
                
                # 'color' 컬럼을 사용하여 지도에 색상 표시
                st.map(filtered_data, color='color')
                
        else:
            # K=1이면 (기본값) 클러스터링 없이 표시
            st.map(filtered_data)
            
    else:
        st.warning(TEXTS[lang]['no_data_warn'])

    # 시간대별 픽업 통계 (막대 차트)
    st.subheader(TEXTS[lang]['hist_subheader'])
    hist_values = np.histogram(data['hour'], bins=24, range=(0, 24))[0]
    hist_df = pd.DataFrame({'hour': range(24), 'pickups': hist_values})
    st.bar_chart(hist_df.set_index('hour'))

    # 원본 데이터 표시
    if show_raw_data:
        st.subheader(TEXTS[lang]['raw_data_subheader'])
        st.dataframe(filtered_data, use_container_width=True)
else:
    st.error(TEXTS[lang]['data_load_error'])

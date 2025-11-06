import os, glob, pathlib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

import plotly.express as px

st.set_page_config(page_title="신용카드 고객 이탈 대시보드(간단版)", page_icon="💳", layout="wide")
st.title("💳 신용카드 고객 이탈(Churn) 대시보드 — 간단版")
st.caption("목적: 고객 이탈 예측 → 취약 세그먼트 식별 → 서비스 개선 포인트 제시")

# ----------------------------
# 데이터 로드
# ----------------------------
st.sidebar.header("데이터")
mode = st.sidebar.radio("데이터 소스", ["KaggleHub 자동 다운로드", "CSV 업로드"], horizontal=True)

def load_from_kagglehub():
    """kagglehub: gonieahn/zero-base-project-creditcard-analysis"""
    try:
        import kagglehub
        path = kagglehub.dataset_download("gonieahn/zero-base-project-creditcard-analysis")
        candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        if not candidates:
            return None, "CSV 파일이 없습니다."
        ranked = sorted(
            candidates,
            key=lambda p: (("churn" not in p.lower()) and ("attrition" not in p.lower()), len(p))
        )
        df = pd.read_csv(ranked[0])
        return df, f"Loaded: {pathlib.Path(ranked[0]).name}"
    except Exception as e:
        return None, f"오류: {e}"

if mode == "CSV 업로드":
    up = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
    if up:
        df = pd.read_csv(up)
        src = f"Uploaded: {up.name}"
    else:
        df, src = None, "CSV 업로드 필요"
else:
    df, src = load_from_kagglehub()

st.sidebar.caption(src)
if df is None or df.empty:
    st.stop()

df.columns = [c.strip().lower() for c in df.columns]

# ----------------------------
# 컬럼 매핑(데이터셋 변형 대응, 없으면 건너뜀)
# ----------------------------
CAND = {
    "target": ["attrition_flag", "churn", "is_churn", "customer_status"],
    "age": ["age", "customer_age"],
    "gender": ["gender", "sex"],
    "marital": ["marital_status", "marital", "maritalstatus"],
    "income_cat": ["income_category", "income_cat", "income_bracket"],
    "card_type": ["card_category", "card_type", "card"],
    "tenure": ["months_on_book", "tenure_months", "tenure"],
    "inactive_m": ["months_inactive_12_mon", "inactive_months", "months_inactive"],
    "contacts_m": ["contacts_count_12_mon", "contacts_12m", "contacts"],
    "credit_limit": ["credit_limit", "clv", "creditlimit"],
    "total_bal": ["total_trans_amt", "total_balance", "total_amt"],
    "total_cnt": ["total_trans_ct", "txn_count", "trans_count"]
}
def pick(name_list): 
    for c in name_list:
        if c in df.columns: return c
    return None

COL = {k: pick(v) for k,v in CAND.items()}
target_col = COL["target"]
if target_col is None:
    st.error("이탈 타깃 컬럼을 찾지 못함. 후보: " + ", ".join(CAND["target"]))
    st.stop()

# 타깃 표준화(1=이탈, 0=유지)
y_raw = df[target_col].astype(str).str.lower()
if set(np.unique(y_raw)) - {"0","1"}:
    y = y_raw.isin(["1","true","yes","y","attrited customer","churned","attrited","exited"]).astype(int)
else:
    y = y_raw.astype(int)

# 피처 구성(있으면 사용)
feature_candidates = [
    COL["age"], COL["gender"], COL["marital"], COL["income_cat"], COL["card_type"],
    COL["tenure"], COL["inactive_m"], COL["contacts_m"],
    COL["credit_limit"], COL["total_bal"], COL["total_cnt"]
]
features = [c for c in feature_candidates if c is not None]
X = df[features].copy()

num_cols = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in features if c not in num_cols]

# ----------------------------
# 사이드바 필터(핵심만)
# ----------------------------
st.sidebar.header("필터")
if COL["age"] in X.columns:
    a_min, a_max = int(X[COL["age"]].min()), int(X[COL["age"]].max())
    age_range = st.sidebar.slider("연령 범위", a_min, a_max, (a_min, a_max))
else:
    age_range = None

if COL["inactive_m"] in X.columns:
    i_max = int(X[COL["inactive_m"]].max())
    inact = st.sidebar.slider("최근 12개월 비활성 개월", 0, i_max, (0, i_max))
else:
    inact = None

mask = pd.Series(True, index=X.index)
if age_range and COL["age"] in X.columns:
    mask &= (X[COL["age"]].between(age_range[0], age_range[1]))
if inact and COL["inactive_m"] in X.columns:
    mask &= (X[COL["inactive_m"]].between(inact[0], inact[1]))

Xf, yf = X[mask].copy(), y[mask].copy()

# ----------------------------
# 탭: ①개요 ②모델 ③세그먼트
# ----------------------------
tab1, tab2, tab3 = st.tabs(["① 개요", "② 이탈 예측(간단)", "③ 취약 세그먼트"])

# ① 개요
with tab1:
    st.subheader("데이터 개요")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("표본 수", f"{len(Xf):,}")
    c2.metric("이탈 비율", f"{(yf.mean()*100):.1f}%")
    if COL["inactive_m"] in Xf.columns:
        c3.metric("평균 비활성(개월)", f"{Xf[COL['inactive_m']].mean():.2f}")
    if COL["tenure"] in Xf.columns:
        c4.metric("평균 가입기간(개월)", f"{Xf[COL['tenure']].mean():.2f}")

    st.markdown("**이탈/유지 분포**")
    lab = yf.replace({1:"Churned", 0:"Active"})
    st.plotly_chart(px.histogram(lab, color=lab), use_container_width=True)

    # 연령/비활성 간단 분포
    plots = [COL["age"], COL["inactive_m"], COL["total_cnt"], COL["credit_limit"]]
    plots = [c for c in plots if c in Xf.columns]
    if plots:
        st.markdown("**주요 변수 분포(이탈 여부별)**")
        for c in plots:
            st.plotly_chart(
                px.box(pd.DataFrame({c: Xf[c], "churn": yf}), x="churn", y=c, points="suspectedoutliers", color="churn"),
                use_container_width=True
            )

# ② 이탈 예측(간단)
with tab2:
    st.subheader("로지스틱 회귀 — 간단 지표")
    test_size = st.slider("검증 비율", 0.1, 0.4, 0.2, step=0.05)
    rnd = st.number_input("random_state", 1, 9999, 42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(Xf, yf, test_size=test_size, random_state=rnd, stratify=yf)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )
    clf = LogisticRegression(max_iter=300, class_weight="balanced")
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
    cols[1].metric("Precision", f"{precision_score(y_test, pred):.3f}")
    cols[2].metric("Recall", f"{recall_score(y_test, pred):.3f}")
    cols[3].metric("F1", f"{f1_score(y_test, pred):.3f}")

    # 변수 중요도(절대 계수 Top 12)
    try:
        ohe = pipe.named_steps["prep"].named_transformers_["cat"]
        num_names = num_cols
        cat_names = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols)>0 else []
        feat_names = num_names + cat_names
        coefs = np.abs(pipe.named_steps["clf"].coef_[0])
        if len(coefs) == len(feat_names):
            imp = pd.DataFrame({"feature": feat_names, "importance": coefs}).sort_values("importance", ascending=False).head(12)
            st.markdown("**변수 중요도(Top 12)**")
            st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h"), use_container_width=True)
    except Exception:
        st.info("변수 중요도를 계산할 수 없음(피처명/계수 불일치).")

# ③ 취약 세그먼트
with tab3:
    st.subheader("취약 세그먼트 인사이트")
    # 연령대별 이탈률
    if COL["age"] in Xf.columns:
        bins = [0,30,40,50,60,70,200]
        labels = ["<30","30s","40s","50s","60s","70+"]
        age_bin = pd.cut(Xf[COL["age"]], bins=bins, labels=labels, right=False)
        ag = pd.DataFrame({"age_bin": age_bin, "churn": yf})
        ag = ag.groupby("age_bin")["churn"].mean().reset_index()
        st.markdown("**연령대별 이탈률**")
        st.plotly_chart(px.bar(ag, x="age_bin", y="churn", text="churn", range_y=[0,1]), use_container_width=True)

    # 비활성 개월 vs 이탈률
    if COL["inactive_m"] in Xf.columns:
        tmp = pd.DataFrame({COL["inactive_m"]: Xf[COL["inactive_m"]].astype(int), "churn": yf})
        gr = tmp.groupby(COL["inactive_m"])["churn"].mean().reset_index()
        st.markdown("**최근 12개월 비활성 개월 수 vs 이탈률**")
        st.plotly_chart(px.line(gr, x=COL["inactive_m"], y="churn", markers=True), use_container_width=True)

    # 거래건수/한도 힌트(간단)
    if COL["total_cnt"] in Xf.columns:
        st.markdown("**거래건수 분위별 이탈률**")
        q = pd.qcut(Xf[COL["total_cnt"]], q=5, duplicates="drop")
        g = pd.DataFrame({"bin": q, "churn": yf}).groupby("bin")["churn"].mean().reset_index()
        st.plotly_chart(px.bar(g, x="bin", y="churn"), use_container_width=True)

st.divider()
st.caption("해석: 비활성↑·거래건수↓·(필요 시) 고연령대에서 이탈률 상승 → 간편인증/리마인드/상담연결 등 재활성 전략 우선 적용")

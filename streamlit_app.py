# app.py
import os, glob, pathlib
import numpy as np
import pandas as pd
import streamlit as st

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="신용카드 고객 이탈 대시보드", page_icon="💳", layout="wide")
st.title("💳 신용카드 고객 이탈(Churn) 대시보드")
st.caption("목적: 고객 이탈 예측 및 취약 세그먼트(고령층·디지털 비활성) 진단 → 서비스 개선/유지 전략")

# ----------------------------
# 0) 데이터 로드
# ----------------------------
st.sidebar.header("데이터")
mode = st.sidebar.radio("데이터 소스", ["KaggleHub 자동 다운로드", "CSV 업로드"], horizontal=True)

def load_from_kagglehub():
    """
    캐글: gonieahn/zero-base-project-creditcard-analysis
    안에 포함된 CSV를 탐색해서 가장 관련성 높은 파일을 읽는다.
    """
    try:
        import kagglehub
        path = kagglehub.dataset_download("gonieahn/zero-base-project-creditcard-analysis")
        # 대개 한 폴더 내 csv가 여러 개 있음 → 'churn' 또는 'attrition' 단어가 있으면 우선 사용
        candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        if not candidates:
            return None, "CSV 파일을 찾지 못했습니다."

        # 우선순위: churn/attrition 포함 → 그 외는 첫 번째
        ranked = sorted(
            candidates,
            key=lambda p: (("churn" not in p.lower()) and ("attrition" not in p.lower()), len(p))
        )
        df = pd.read_csv(ranked[0])
        return df, f"Loaded: {pathlib.Path(ranked[0]).name}"
    except Exception as e:
        return None, f"오류: {e}"

uploaded = None
if mode == "CSV 업로드":
    uploaded = st.sidebar.file_uploader("CSV 파일 선택", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        source_msg = f"Uploaded: {uploaded.name}"
    else:
        df = None
        source_msg = "CSV를 업로드하세요."
else:
    df, source_msg = load_from_kagglehub()

st.sidebar.caption(source_msg)

if df is None or df.empty:
    st.warning("데이터를 불러오지 못했습니다. CSV 업로드 또는 KaggleHub 다시 시도하세요.")
    st.stop()

# 컬럼명 소문자 통일
df.columns = [c.strip().lower() for c in df.columns]

# ----------------------------
# 1) 컬럼 매핑 (데이터셋 변형 대응)
# ----------------------------
# 흔한 컬럼 후보들(다 있지 않아도 작동)
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
    "total_bal": ["total_trans_amt", "total_amt_chng_q4_q1", "total_balance", "total_amt"],
    "total_cnt": ["total_trans_ct", "total_ct_chng_q4_q1", "txn_count", "trans_count"]
}

def pick(colnames):
    for c in colnames:
        if c in df.columns:
            return c
    return None

COL = {k: pick(v) for k, v in CAND.items()}
target_col = COL["target"]

# 타깃이 없으면 추정 불가 → 종료
if target_col is None:
    st.error("타깃(이탈 여부) 컬럼을 찾지 못했습니다. 후보: " + ", ".join(CAND["target"]))
    st.stop()

# 이탈 플래그 표준화 (1=이탈, 0=유지)
y_raw = df[target_col].astype(str).str.lower()
if set(np.unique(y_raw)) - {"0", "1"}:
    # 문자열 범주형을 0/1로 매핑(가장 흔한 규칙)
    # 'attrited' / 'churned' / 'yes' → 1, 나머지 → 0
    y = y_raw.isin(["1", "true", "yes", "y", "attrited customer", "churned", "attrited", "exited"]).astype(int)
else:
    y = y_raw.astype(int)

# 사용 후보 피처 목록
feature_candidates = [
    COL["age"], COL["gender"], COL["marital"], COL["income_cat"], COL["card_type"],
    COL["tenure"], COL["inactive_m"], COL["contacts_m"],
    COL["credit_limit"], COL["total_bal"], COL["total_cnt"]
]
features = [c for c in feature_candidates if c is not None and c in df.columns]
X = df[features].copy()

# 타입 추론
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in features if c not in num_cols]

# ----------------------------
# 사이드바 필터 (연령/비활성/카테고리)
# ----------------------------
st.sidebar.header("필터")
if COL["age"] in X.columns:
    age_min, age_max = int(X[COL["age"]].min()), int(X[COL["age"]].max())
    age_range = st.sidebar.slider("연령 범위", min_value=age_min, max_value=age_max,
                                  value=(age_min, age_max))
else:
    age_range = None

inactive_filter = None
if COL["inactive_m"] in X.columns:
    imax = int(X[COL["inactive_m"]].max())
    inactive_filter = st.sidebar.slider("최근 12개월 비활성 개월 수", 0, imax, (0, imax))

cat_filters = {}
for c in cat_cols:
    opts = sorted(X[c].dropna().astype(str).unique().tolist())
    sel = st.sidebar.multiselect(f"{c} 선택", opts)
    if sel:
        cat_filters[c] = sel

# 필터 적용
mask = pd.Series(True, index=X.index)
if age_range and COL["age"] in X.columns:
    mask &= (X[COL["age"]] >= age_range[0]) & (X[COL["age"]] <= age_range[1])
if inactive_filter and COL["inactive_m"] in X.columns:
    mask &= (X[COL["inactive_m"]] >= inactive_filter[0]) & (X[COL["inactive_m"]] <= inactive_filter[1])
for c, vs in cat_filters.items():
    mask &= X[c].astype(str).isin(vs)

Xf, yf = X[mask].copy(), y[mask].copy()

# ----------------------------
# 탭 구성: ①개요 ②모델 ③세그먼트
# ----------------------------
tab1, tab2, tab3 = st.tabs(["① 개요", "② 이탈 예측 모델", "③ 취약 세그먼트 인사이트"])

# ----------------------------
# 탭 1) 개요
# ----------------------------
with tab1:
    st.subheader("데이터 개요")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("표본 수", f"{len(Xf):,}")
    c2.metric("이탈 비율(필터 적용)", f"{(yf.mean()*100):.1f}%")
    if COL["inactive_m"] in Xf.columns:
        c3.metric("평균 비활성 개월", f"{Xf[COL['inactive_m']].mean():.2f}")
    if COL["tenure"] in Xf.columns:
        c4.metric("평균 가입기간(개월)", f"{Xf[COL['tenure']].mean():.2f}")

    st.markdown("**이탈/유지 분포**")
    fig = px.histogram(yf.replace({1: "Churned", 0: "Active"}), color= yf.replace({1: "Churned", 0: "Active"}))
    st.plotly_chart(fig, use_container_width=True)

    # 연령/비활성/거래수 등 분포
    grid_num_cols = [c for c in [COL["age"], COL["inactive_m"], COL["total_cnt"], COL["credit_limit"]] if c in Xf.columns]
    if grid_num_cols:
        st.markdown("**주요 수치형 변수 분포(이탈 여부별)**")
        for c in grid_num_cols:
            st.plotly_chart(px.box(pd.DataFrame({c: Xf[c], "churn": yf}),
                                   x="churn", y=c, points="all", color="churn"), use_container_width=True)

# ----------------------------
# 탭 2) 예측 모델
# ----------------------------
with tab2:
    st.subheader("이탈 예측 모델")
    st.caption("수치형: 표준화 / 범주형: 원-핫 인코딩 → 분류기(로지스틱 또는 그래디언트부스팅)")

    test_size = st.slider("검증 데이터 비율", 0.1, 0.4, 0.2, step=0.05)
    rnd = st.number_input("random_state", 1, 9999, 42, step=1)
    model_name = st.selectbox("모델", ["LogisticRegression", "GradientBoostingClassifier"])

    X_train, X_test, y_train, y_test = train_test_split(Xf, yf, test_size=test_size, random_state=rnd, stratify=yf)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    if model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=300, class_weight="balanced")
    else:
        clf = GradientBoostingClassifier()

    pipe = Pipeline([("prep", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    # 성능지표
    colm = st.columns(5)
    colm[0].metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
    colm[1].metric("Precision", f"{precision_score(y_test, pred):.3f}")
    colm[2].metric("Recall", f"{recall_score(y_test, pred):.3f}")
    colm[3].metric("F1", f"{f1_score(y_test, pred):.3f}")
    if proba is not None:
        colm[4].metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.3f}")

    # ROC Curve
    if proba is not None:
        fpr, tpr = RocCurveDisplay.from_predictions(y_test, proba).fpr, RocCurveDisplay.from_predictions(y_test, proba).tpr
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Baseline", line=dict(dash="dash")))
        roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=400)
        st.plotly_chart(roc_fig, use_container_width=True)

    # 중요 변수(로지스틱: 절대 계수 / GBoost: feature_importances_)
    st.markdown("**변수 중요도(참고용)**")
    # 피처명 복구
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    num_names = num_cols
    cat_names = []
    if len(cat_cols) > 0:
        try:
            cat_names = list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            # 구버전 호환
            cat_names = list(ohe.get_feature_names(cat_cols))
    all_feat_names = num_names + cat_names

    importances = None
    if model_name == "GradientBoostingClassifier" and hasattr(pipe.named_steps["clf"], "feature_importances_"):
        importances = pipe.named_steps["clf"].feature_importances_
    elif model_name == "LogisticRegression" and hasattr(pipe.named_steps["clf"], "coef_"):
        # 계수의 절대값
        importances = np.abs(pipe.named_steps["clf"].coef_[0])

    if importances is not None and len(all_feat_names) == len(importances):
        imp_df = pd.DataFrame({"feature": all_feat_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        st.plotly_chart(px.bar(imp_df, x="importance", y="feature", orientation="h"), use_container_width=True)
    else:
        st.info("변수 중요도를 계산할 수 없습니다(피처명/계수 불일치 또는 모델 한계).")

# ----------------------------
# 탭 3) 취약 세그먼트 인사이트
# ----------------------------
with tab3:
    st.subheader("취약 세그먼트 (고령층·비활성·한도·거래)")
    # 고령 기준(조정 가능)
    senior_cut = st.slider("고령 기준 연령", 55, 80, 60, step=5) if COL["age"] in Xf.columns else None

    vis_cols = []
    if COL["age"] in Xf.columns: vis_cols.append(COL["age"])
    if COL["inactive_m"] in Xf.columns: vis_cols.append(COL["inactive_m"])
    if COL["total_cnt"] in Xf.columns: vis_cols.append(COL["total_cnt"])
    if COL["credit_limit"] in Xf.columns: vis_cols.append(COL["credit_limit"])

    # (A) 연령대별 이탈률
    if COL["age"] in Xf.columns:
        bins = [0, 30, 40, 50, 60, 70, 120]
        labels = ["<30", "30s", "40s", "50s", "60s", "70+"]
        age_bin = pd.cut(Xf[COL["age"]], bins=bins, labels=labels, right=False)
        st.markdown("**연령대별 이탈률**")
        ag = pd.DataFrame({"age_bin": age_bin, "churn": yf[mask]})
        ag = ag.groupby("age_bin")["churn"].mean().reset_index()
        st.plotly_chart(px.bar(ag, x="age_bin", y="churn", text="churn", range_y=[0, 1]), use_container_width=True)

    # (B) 비활성 대비 이탈률
    if COL["inactive_m"] in Xf.columns:
        st.markdown("**최근 12개월 비활성 개월 수 vs 이탈률**")
        im = pd.DataFrame({COL["inactive_m"]: Xf[COL["inactive_m"]], "churn": yf[mask]})
        im[COL["inactive_m"]] = im[COL["inactive_m"]].astype(int)
        gr = im.groupby(COL["inactive_m"])["churn"].mean().reset_index()
        st.plotly_chart(px.line(gr, x=COL["inactive_m"], y="churn", markers=True), use_container_width=True)

    # (C) 한도/거래 수에 따른 위험도
    if COL["credit_limit"] in Xf.columns and COL["total_cnt"] in Xf.columns:
        st.markdown("**신용한도 × 거래건수 vs 이탈 비율**")
        tmp = pd.DataFrame({
            "limit": Xf[COL["credit_limit"]],
            "txcnt": Xf[COL["total_cnt"]],
            "churn": yf[mask]
        }).dropna()
        ql = pd.qcut(tmp["limit"], q=5, duplicates="drop")
        qc = pd.qcut(tmp["txcnt"], q=5, duplicates="drop")
        heat = tmp.groupby([ql, qc])["churn"].mean().reset_index()
        heat["limit_q"] = heat[ql.name].astype(str)
        heat["tx_q"] = heat[qc.name].astype(str)
        fig = px.density_heatmap(heat, x="tx_q", y="limit_q", z="churn",
                                 color_continuous_scale="Reds", histfunc="avg")
        st.plotly_chart(fig, use_container_width=True)

    # (D) 고령층 세부(선택)
    if senior_cut and COL["age"] in Xf.columns:
        st.markdown(f"**고령층(≥{senior_cut}) 세그먼트 요약**")
        senior_mask = Xf[COL["age"]] >= senior_cut
        s_df = Xf[senior_mask]
        s_y = yf[mask][senior_mask] if hasattr(yf[mask], "__len__") else yf[senior_mask]

        c1, c2, c3 = st.columns(3)
        c1.metric("고령층 표본", f"{len(s_df):,}")
        c2.metric("고령층 이탈률", f"{(s_y.mean()*100):.1f}%")
        if COL["inactive_m"] in s_df.columns:
            c3.metric("평균 비활성 개월", f"{s_df[COL['inactive_m']].mean():.2f}")

        # 고령층 내 주요 불리 요인(단변량) – 예시: 비활성/거래건수
        charts = []
        if COL["inactive_m"] in s_df.columns:
            charts.append((COL["inactive_m"], "비활성 개월"))
        if COL["total_cnt"] in s_df.columns:
            charts.append((COL["total_cnt"], "거래 건수"))

        for col, label in charts:
            df_plot = pd.DataFrame({label: s_df[col], "churn": s_y})
            df_plot[label] = pd.qcut(df_plot[label], q=5, duplicates="drop")
            g = df_plot.groupby(label)["churn"].mean().reset_index()
            st.plotly_chart(px.bar(g, x=label, y="churn", text="churn"), use_container_width=True)

st.divider()
st.caption("해석 가이드: 비활성개월↑, 거래건수↓, (필요 시) 고령층에서 이탈률이 높게 나타나면 '간편 인증·리마인드·대체채널' 우선 적용 타깃.")

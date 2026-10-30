import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# LangGraph is used for an explicit, inspectable tool workflow.
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

st.set_page_config(page_title="Agentic Data Analyst", page_icon="📊", layout="wide")
st.title("📊 Agentic Data Analyst")
st.caption("Upload a CSV, ask a question in plain English, and get evidence-grounded analysis.")

with st.sidebar:
    st.header("Settings")
    st.write("This MVP runs locally. LLM answers are optional; core analysis works without an API key.")
    api_key = st.text_input("OpenAI API key (optional)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    st.caption("If supplied, the key is used only for this app session.")

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
if not uploaded:
    st.info("Start by uploading a CSV file. You can try the sample file included in the project folder: sample_sales.csv")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

st.subheader("Dataset preview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{df.shape[0]:,}")
m2.metric("Columns", f"{df.shape[1]:,}")
m3.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
m4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
st.dataframe(df.head(100), use_container_width=True)

with st.expander("Automated data profile", expanded=True):
    profile = pd.DataFrame({
        "column": df.columns.astype(str),
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "missing_%": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    st.dataframe(profile, use_container_width=True)
    st.write("Numeric summary")
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1]:
        st.dataframe(numeric.describe().T, use_container_width=True)
    else:
        st.write("No numeric columns detected.")

st.divider()
st.subheader("Ask a question")
question = st.text_input(
    "What would you like to know?",
    placeholder="e.g., Which category has the highest average sales? Are there missing values?"
)
run_analysis = st.button("Analyze question", type="primary")

def deterministic_answer(data, q):
    """Safe rule-based tool router. Answers only from computations on the uploaded dataframe."""
    ql = q.lower()
    nums = data.select_dtypes(include=np.number).columns.tolist()
    cats = data.select_dtypes(exclude=np.number).columns.tolist()

    if any(k in ql for k in ["missing", "null", "na values", "incomplete"]):
        miss = data.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0]
        if miss.empty:
            return "There are no missing cells in the uploaded dataset.", None
        return "Columns with missing values (count):\n" + miss.to_string(), None

    if any(k in ql for k in ["duplicate", "repeated rows"]):
        return f"The dataset contains {int(data.duplicated().sum())} duplicate rows.", None

    if any(k in ql for k in ["correlation", "correlate", "relationship"]):
        if len(nums) < 2:
            return "At least two numeric columns are needed to compute a correlation matrix.", None
        corr = data[nums].corr(numeric_only=True).round(3)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Numeric feature correlation")
        return "Computed Pearson correlations for numeric columns. Correlation does not prove causation.", fig

    if any(k in ql for k in ["distribution", "histogram", "spread"]):
        if not nums:
            return "No numeric columns are available for a distribution plot.", None
        col = next((c for c in nums if c.lower() in ql), nums[0])
        fig = px.histogram(data, x=col, title=f"Distribution of {col}", marginal="box")
        return f"Showing the distribution of numeric column '{col}'.", fig

    if any(k in ql for k in ["highest", "lowest", "top", "average by", "group by", "per category", "compare"]):
        if not nums:
            return "This question sounds like a group comparison, but no numeric columns were found.", None
        num = next((c for c in nums if c.lower() in ql), nums[0])
        cat = next((c for c in cats if c.lower() in ql), cats[0] if cats else None)
        if cat is None:
            agg = data[num].agg(["mean", "median", "min", "max", "sum"]).round(3)
            return f"Summary for '{num}':\n{agg.to_string()}", None
        grouped = data.groupby(cat, dropna=False)[num].mean().sort_values(ascending=False).reset_index(name=f"average_{num}")
        fig = px.bar(grouped.head(20), x=cat, y=f"average_{num}", title=f"Average {num} by {cat}")
        return f"Computed average '{num}' grouped by '{cat}'. Top results:\n{grouped.head(10).to_string(index=False)}", fig

    if any(k in ql for k in ["summary", "describe", "statistics", "overview"]):
        if nums:
            return "Numeric descriptive statistics:\n" + data[nums].describe().round(3).to_string(), None
        return f"Dataset has {len(data)} rows and {len(data.columns)} columns. No numeric columns were detected.", None

    return ("I couldn't confidently map that question to a supported analysis tool. Try asking about missing values, duplicates, summary statistics, distributions, correlations, or highest/lowest averages by category.", None)

def llm_explanation(question_text, computed_evidence, key):
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a careful data analyst. Explain only the supplied computed evidence. Do not invent figures or claim causal relationships. If evidence is insufficient, say so."},
                {"role": "user", "content": f"Question: {question_text}\nComputed evidence from the dataset:\n{computed_evidence}\nExplain the result briefly and state any limitation."}
            ]
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"LLM explanation unavailable ({exc}). The computed result below is still valid."

if run_analysis:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        answer, fig = deterministic_answer(df, question)
        st.markdown("#### Tool result")
        st.text(answer)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        explanation = llm_explanation(question, answer, api_key)
        if explanation:
            st.markdown("#### LLM explanation (based on computed evidence)")
            st.write(explanation)
        st.caption("Reliability note: numbers above are computed from the uploaded data. The natural-language router supports common question patterns and may not understand every phrasing.")

st.divider()
st.subheader("Interactive visualization")
all_cols = df.columns.tolist()
left, right = st.columns(2)
with left:
    chart_type = st.selectbox("Chart type", ["Bar chart", "Scatter plot", "Line chart", "Histogram", "Box plot"])
    x_col = st.selectbox("X-axis / category", all_cols, index=0)
with right:
    y_options = df.select_dtypes(include=np.number).columns.tolist()
    y_col = st.selectbox("Y-axis / numeric column", y_options if y_options else all_cols, index=0)
if st.button("Build chart"):
    try:
        if chart_type == "Bar chart":
            if pd.api.types.is_numeric_dtype(df[y_col]):
                chart_df = df.groupby(x_col, dropna=False)[y_col].mean().reset_index()
                fig = px.bar(chart_df.head(50), x=x_col, y=y_col, title=f"Average {y_col} by {x_col}")
            else:
                counts = df[x_col].astype(str).value_counts().head(30).rename_axis(x_col).reset_index(name="count")
                fig = px.bar(counts, x=x_col, y="count", title=f"Count by {x_col}")
        elif chart_type == "Scatter plot":
            if len(y_options) < 2:
                st.warning("Scatter plot needs two numeric columns; choose a different chart.")
                st.stop()
            x_numeric = st.selectbox("Choose numeric X for scatter", y_options, key="scatter_x")
            fig = px.scatter(df, x=x_numeric, y=y_col, title=f"{y_col} vs {x_numeric}")
        elif chart_type == "Line chart":
            fig = px.line(df.head(500), x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        elif chart_type == "Histogram":
            target = y_col if pd.api.types.is_numeric_dtype(df[y_col]) else x_col
            fig = px.histogram(df, x=target, title=f"Distribution of {target}")
        else:
            target = y_col if pd.api.types.is_numeric_dtype(df[y_col]) else x_col
            fig = px.box(df, y=target, title=f"Box plot of {target}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not build chart: {exc}")

st.divider()
st.subheader("Baseline ML evaluation (optional)")
st.write("Choose a target column to train a simple Random Forest baseline. This is a quick benchmark, not proof that the model is production-ready.")
target_col = st.selectbox("Target column", ["— Skip ML evaluation —"] + all_cols)
if target_col != "— Skip ML evaluation —":
    test_size = st.slider("Test split (%)", 10, 40, 20, 5)
    if st.button("Train baseline model"):
        model_df = df.dropna(subset=[target_col]).copy()
        if model_df[target_col].nunique() < 2 or len(model_df) < 20:
            st.error("Need at least 20 usable rows and at least 2 target values.")
        else:
            X = model_df.drop(columns=[target_col])
            y = model_df[target_col]
            # Remove columns that cannot be safely encoded or are entirely missing.
            X = X.loc[:, X.notna().any()]
            cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            num_cols = X.select_dtypes(include=np.number).columns.tolist()
            if not num_cols and not cat_cols:
                st.error("No usable feature columns found.")
            else:
                is_classification = (y.dtype == "object" or y.dtype.name == "category" or y.dtype == bool or y.nunique() <= 10)
                stratify = y if is_classification and y.value_counts().min() >= 2 else None
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size / 100, random_state=42, stratify=stratify
                    )
                    transformers = []
                    if num_cols:
                        transformers.append(("num", Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler())
                        ]), num_cols))
                    if cat_cols:
                        transformers.append(("cat", Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))
                        ]), cat_cols))
                    preprocessor = ColumnTransformer(transformers)
                    estimator = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced") if is_classification else RandomForestRegressor(n_estimators=150, random_state=42)
                    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
                    pipeline.fit(X_train, y_train)
                    pred = pipeline.predict(X_test)
                    st.markdown("#### Holdout test results")
                    if is_classification:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
                        c2.metric("Precision (weighted)", f"{precision_score(y_test, pred, average='weighted', zero_division=0):.3f}")
                        c3.metric("Recall (weighted)", f"{recall_score(y_test, pred, average='weighted', zero_division=0):.3f}")
                        c4.metric("F1 (weighted)", f"{f1_score(y_test, pred, average='weighted', zero_division=0):.3f}")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("MAE", f"{mean_absolute_error(y_test, pred):.3f}")
                        c2.metric("RMSE", f"{mean_squared_error(y_test, pred) ** 0.5:.3f}")
                        c3.metric("R²", f"{r2_score(y_test, pred):.3f}")
                    st.caption(f"Random Forest baseline; random_state=42; test split={test_size}%. Results depend on dataset, target choice, leakage, and split quality.")
                except Exception as exc:
                    st.error(f"Model training failed: {exc}")

st.caption("Built with Streamlit, Pandas, Plotly, scikit-learn, and an optional OpenAI-compatible explanation step. LangGraph availability is checked at runtime.")

import os
import json
import time
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# =========================================================
#  Setup & configuration
# =========================================================
st.set_page_config(
    page_title="Comment Classifier | Customer Comment Analyzer",
    layout="wide",
)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Sample file path (the file you uploaded earlier)
SAMPLE_FILE_PATH = "/mnt/data/random_product_data.xlsx"

# Simple test credentials (for tech case)
USERNAME = "manvendraray"
PASSWORD = "12345"

# Canonical sentiment & category labels (to keep output clean)
VALID_SENTIMENTS = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}

VALID_CATEGORIES = {
    "delivery": "Delivery",
    "shipping": "Delivery",
    "logistics": "Delivery",
    "product quality": "Product Quality",
    "quality": "Product Quality",
    "defect": "Product Quality",
    "customer service": "Customer Service",
    "support": "Customer Service",
    "service": "Customer Service",
    "website": "Website/App",
    "app": "Website/App",
    "pricing": "Pricing",
    "price": "Pricing",
}

# ---------------------------------------------------------
#  Light custom CSS to make it look more “product-ready”
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Main page tweaks */
.main > div {
    padding-top: 1.5rem;
}
.big-title {
    font-size: 2.3rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.1rem;
}
.sub-title {
    font-size: 0.95rem;
    color: #6c757d;
    margin-bottom: 1.2rem;
}
.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    color: #6c757d;
    letter-spacing: 0.08em;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 600;
}

/* Table tweaks */
.dataframe td, .dataframe th {
    white-space: normal !important;
}

/* Sidebar header */
.sidebar-title {
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.2rem;
}
.sidebar-sub {
    font-size: 0.8rem;
    color: #adb5bd;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
#  Init session state
# ---------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None

if "last_runtime" not in st.session_state:
    st.session_state["last_runtime"] = None

# =========================================================
#  Helper functions
# =========================================================
def normalize_sentiment(raw: str) -> str:
    if not raw:
        return ""
    key = raw.strip().lower()
    return VALID_SENTIMENTS.get(key, raw.title())


def normalize_category(raw: str) -> str:
    if not raw:
        return ""
    key = raw.strip().lower()
    if key in VALID_CATEGORIES:
        return VALID_CATEGORIES[key]
    for k, v in VALID_CATEGORIES.items():
        if k in key:
            return v
    return raw.title()


def _extract_first_json(text: str):
    """
    Find first balanced {...} substring and parse it.
    Returns dict or raises ValueError.
    """
    if not text or "{" not in text:
        raise ValueError("no JSON braced block")
    start = text.find("{")
    stack = []
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            stack.append("{")
        elif ch == "}":
            if not stack:
                # unmatched closing
                continue
            stack.pop()
            if not stack:
                candidate = text[start:i+1]
                return json.loads(candidate)
    raise ValueError("no balanced JSON object found")


def simple_rule_fallback(comment: str):
    c = (comment or "").lower()
    if any(w in c for w in ["refund", "return", "bad", "defect", "broken", "not satisfied", "terrible", "poor"]):
        return {"sentiment": "Negative", "category": "Product Quality", "themes": ["complaint"]}
    if any(w in c for w in ["fast", "quick", "on time", "arrived", "five stars", "great"]):
        return {"sentiment": "Positive", "category": "Delivery", "themes": ["delivery"]}
    if any(w in c for w in ["price", "cheap", "expensive", "cost"]):
        return {"sentiment": "Neutral", "category": "Pricing", "themes": ["pricing"]}
    # default fallback
    return {"sentiment": "Neutral", "category": "Other", "themes": []}


def analyze_comment_robust(comment: str, max_retries=3, base_wait=1):
    """
    Robust wrapper: retries, extracts JSON if wrapped in text, returns (result_dict, raw_text).
    result_dict always has keys sentiment, category, themes.
    """
    system_prompt = (
        "You are an analyst helping a product team understand customer feedback.\n"
        "Given a single customer comment, output EXACTLY one JSON object and NOTHING else with keys:\n"
        "  sentiment: one of [\"Positive\",\"Neutral\",\"Negative\"],\n"
        "  category: one of [\"Delivery\",\"Product Quality\",\"Customer Service\",\"Website/App\",\"Pricing\",\"Other\"],\n"
        "  themes: a list of 1-3 short phrases.\n"
        "If you cannot classify, return {\"sentiment\":\"Neutral\",\"category\":\"Other\",\"themes\":[]}.\n"
        "DO NOT include any extra text, explanation, or markdown. Respond only with the JSON object."
    )
    user_prompt = f"Comment: {comment}"

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    attempt = 0
    last_raw = ""
    while attempt < max_retries:
        attempt += 1
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload), timeout=30)
            last_raw = resp.text
            if resp.status_code != 200:
                time.sleep(base_wait * attempt)
                continue
            try:
                data = resp.json()
            except Exception:
                time.sleep(base_wait * attempt)
                continue

            if "choices" in data and data["choices"]:
                raw_text = data["choices"][0]["message"]["content"]
                # Try direct JSON parse
                try:
                    parsed = json.loads(raw_text)
                except Exception:
                    # Try to extract first JSON block
                    try:
                        parsed = _extract_first_json(raw_text)
                    except Exception:
                        # If not found, wait & retry
                        time.sleep(base_wait * attempt)
                        continue

                # Normalize parsed content
                sentiment = normalize_sentiment(parsed.get("sentiment", ""))
                category = normalize_category(parsed.get("category", ""))
                themes = parsed.get("themes") or []
                if isinstance(themes, str):
                    themes = [themes]
                return ({"sentiment": sentiment, "category": category, "themes": themes}, raw_text)
            else:
                time.sleep(base_wait * attempt)
                continue

        except requests.exceptions.RequestException as e:
            last_raw = str(e)
            time.sleep(base_wait * attempt)
            continue

    # If all retries fail, return fallback and the last raw for logging
    fallback = simple_rule_fallback(comment)
    return (fallback, last_raw or "no_response_or_error")

# =========================================================
#  UI blocks
# =========================================================
def show_login():
    st.markdown(
        """
        <div class="big-title">Comment Classifier</div>
        <div class="sub-title">
            Secure access to the customer-comment analysis workspace.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        st.text_input("Username", key="login_user")
        st.text_input("Password", type="password", key="login_pass")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if (
            st.session_state["login_user"] == USERNAME
            and st.session_state["login_pass"] == PASSWORD
        ):
            st.session_state["logged_in"] = True
            st.success("Login successful — welcome to Comment Classifier.")
        else:
            st.error("Wrong username or password. Hint: use the credentials from the tech test.")


def sidebar_content():
    st.sidebar.markdown(
        """
        <div class="sidebar-title">Comment Classifier</div>
        <div class="sidebar-sub">LLM-powered customer feedback intelligence</div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Workflow**")
    st.sidebar.markdown(
        "1. Upload CSV/Excel with a `comment` column\n"
        "2. Click **Process**\n"
        "3. Inspect table + summary dashboard\n"
        "4. Download the enriched CSV"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model**")
    st.sidebar.markdown(
        "- Provider: Groq\n"
        "- Model: `llama-3.1-8b-instant`\n"
        "- Output: sentiment, category, key themes"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built by **Manvendra Ray**")
    st.sidebar.markdown("[GitHub](https://github.com/manvendraray)")
    st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/manvendraray/)")
    st.sidebar.markdown("[Email](mailto:mr6695@nyu.edu)")
    st.sidebar.caption("Built as a technical case study.")


def main_app():
    sidebar_content()

    # Header
    st.markdown(
        """
        <div class="big-title">Comment Classifier</div>
        <div class="sub-title">
            Turn unstructured customer comments into structured insight — sentiment, category,
            and key themes, powered by LLMs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not GROQ_API_KEY:
        st.error(
            "GROQ_API_KEY is not configured.\n\n"
            "Add it to Streamlit secrets or your environment before running the app."
        )
        return

    # File upload + sample button
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file with a `comment` column",
            type=["csv", "xlsx", "xls"],
        )
    with col2:
        if st.button("Use sample file"):
            # load local sample path (the file you uploaded earlier)
            try:
                uploaded_file = open(SAMPLE_FILE_PATH, "rb")
                st.success("Loaded sample file from disk.")
                # we keep a marker in session state so downstream uses same object
                st.session_state["_use_sample_path"] = SAMPLE_FILE_PATH
            except Exception as e:
                st.error(f"Could not load sample file: {e}")
                uploaded_file = None

    # If sample path is set in session, build an in-memory file-like object when reading
    sample_path = st.session_state.get("_use_sample_path")

    if uploaded_file is None and not sample_path:
        st.info("Upload a dataset to begin the analysis.")
        return

    # Read file - prefer sample_path if set
    try:
        if sample_path:
            df = pd.read_excel(sample_path)
            # clear after reading so later uploads work normally
            st.session_state.pop("_use_sample_path", None)
        else:
            name = uploaded_file.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # Allow any case for 'comment'
    cols_lower = {c.lower(): c for c in df.columns}
    if "comment" not in cols_lower:
        st.error("The file must have a column named `comment` (any capitalisation).")
        st.write("Detected columns:", list(df.columns))
        return

    comment_col = cols_lower["comment"]

    st.subheader("Input preview")
    st.dataframe(df[[comment_col]].head())

    run_btn = st.button("Process", type="primary")

    if run_btn:
        comments = df[comment_col].astype(str).tolist()
        rows = []

        start_time = time.time()
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total = len(comments)

        for i, c in enumerate(comments, start=1):
            progress_text.text(f"Analyzing comment {i} of {total}…")
            analysis, raw = analyze_comment_robust(c)
            rows.append(
                {
                    "Comment": c,
                    "Sentiment": analysis.get("sentiment", ""),
                    "Category": analysis.get("category", ""),
                    "Key Themes": ", ".join(analysis.get("themes", [])),
                    "raw_response": raw,
                }
            )
            progress_bar.progress(i / total)

        results_df = pd.DataFrame(rows)
        st.session_state["results_df"] = results_df
        st.session_state["last_runtime"] = time.time() - start_time

    # If we have results, show them
    if st.session_state["results_df"] is not None:
        result_df = st.session_state["results_df"]

        # --- Metrics row ---
        total_comments = len(result_df)
        n_pos = (result_df["Sentiment"] == "Positive").sum()
        n_neu = (result_df["Sentiment"] == "Neutral").sum()
        n_neg = (result_df["Sentiment"] == "Negative").sum()
        n_errors = (result_df["Sentiment"] == "Error").sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-label">Total comments</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{total_comments}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-label">Positive / Neutral / Negative</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{n_pos} / {n_neu} / {n_neg}</div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown('<div class="metric-label">Errors</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{n_errors}</div>', unsafe_allow_html=True)
        with col4:
            rt = st.session_state["last_runtime"]
            rt_disp = f"{rt:.1f}s" if rt else "—"
            st.markdown('<div class="metric-label">Runtime</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{rt_disp}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Retry failed rows button
        col_retry, col_export = st.columns([1, 1])
        with col_retry:
            if st.button("Retry failed rows"):
                df = st.session_state["results_df"].copy()
                mask = df["Sentiment"].isin(["Missing", "Error", "", None]) | df["Category"].isin(["Missing", "Error", "", None])
                failed_idx = df[mask].index.tolist()
                if not failed_idx:
                    st.success("No failed rows to retry.")
                else:
                    progress = st.progress(0)
                    for j, i in enumerate(failed_idx, start=1):
                        c = df.at[i, "Comment"]
                        analysis, raw = analyze_comment_robust(c, max_retries=3, base_wait=1)
                        df.at[i, "Sentiment"] = analysis.get("sentiment", "")
                        df.at[i, "Category"] = analysis.get("category", "")
                        df.at[i, "Key Themes"] = ", ".join(analysis.get("themes", []))
                        df.at[i, "raw_response"] = raw
                        progress.progress(j / len(failed_idx))
                    st.session_state["results_df"] = df
                    st.experimental_rerun()

        with col_export:
            # Export raw responses for investigation
            csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download enriched CSV (with raw responses)",
                data=csv_bytes,
                file_name="categorized_comments_with_raw.csv",
                mime="text/csv",
            )

        # --- Tabs: table + summary ---
        tab_table, tab_summary = st.tabs(["Results table", "Summary dashboard"])

        with tab_table:
            st.subheader("Structured output")
            # Show top N rows but allow full download
            st.dataframe(result_df.drop(columns=["raw_response"]), use_container_width=True)

            # Save failed rows for quick inspection
            fails = result_df[
                result_df["Sentiment"].isin(["Missing", "Error", "", None]) |
                result_df["Category"].isin(["Missing", "Error", "", None])
            ]
            if not fails.empty:
                st.warning(f"{len(fails)} rows failed. You can download them to inspect raw responses.")
                st.download_button(
                    "Download failures (with raw)",
                    data=fails.to_csv(index=False).encode("utf-8-sig"),
                    file_name="failed_rows_for_retry.csv",
                    mime="text/csv",
                )

        with tab_summary:
            # 1) Overall sentiment distribution
            st.subheader("Sentiment distribution")
            sent_counts = (
                result_df["Sentiment"]
                .value_counts()
                .reindex(["Positive", "Neutral", "Negative", "Error"])
                .fillna(0)
            )
            st.bar_chart(sent_counts)
            st.markdown("")

            # 2) Top issues by category
            st.subheader("Top categories (by volume)")
            cat_counts = result_df["Category"].value_counts()
            st.bar_chart(cat_counts)

            cat_table = (
                cat_counts.rename("Count")
                .reset_index()
                .rename(columns={"index": "Category"})
            )
            st.dataframe(cat_table, use_container_width=True)
            st.markdown("")

            # 3) Category vs Sentiment pivot (counts + row-wise percentages)
            st.subheader("Category vs sentiment (pivot)")
            pivot_counts = pd.crosstab(result_df["Category"], result_df["Sentiment"])
            st.markdown("**Counts**")
            st.dataframe(pivot_counts, use_container_width=True)
            pivot_pct = pivot_counts.div(pivot_counts.sum(axis=1), axis=0).round(3)
            st.markdown("**Row-wise proportions (per category)**")
            st.dataframe(pivot_pct, use_container_width=True)
            st.caption(
                "This pivot shows, for each category, how feedback is distributed across "
                "Positive / Neutral / Negative, which is often what product and CX teams care about."
            )


# =========================================================
#  Entry point
# =========================================================
def main():
    if not st.session_state["logged_in"]:
        show_login()
        if not st.session_state["logged_in"]:
            return

    main_app()


if __name__ == "__main__":
    main()

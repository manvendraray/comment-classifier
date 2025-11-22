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
    """Map messy sentiment strings to clean labels."""
    if not raw:
        return ""
    key = raw.strip().lower()
    return VALID_SENTIMENTS.get(key, raw.title())


def normalize_category(raw: str) -> str:
    """Map messy category strings to clean, limited set."""
    if not raw:
        return ""
    key = raw.strip().lower()
    # Try exact match
    if key in VALID_CATEGORIES:
        return VALID_CATEGORIES[key]

    # Try fuzzy contains
    for k, v in VALID_CATEGORIES.items():
        if k in key:
            return v

    # Fallback: title case original
    return raw.title()


def analyze_comment(comment: str) -> dict:
    """
    Call Groq LLM to analyse a single comment.

    We explicitly constrain the model to our sentiment + category schema,
    and ask for pure JSON so downstream processing is robust.
    """
    system_prompt = (
        "You are an analyst helping a product team understand customer feedback.\n"
        "Given a single customer comment (English or Chinese), output a JSON object "
        "with the following keys:\n\n"
        "  sentiment: one of [\"Positive\", \"Neutral\", \"Negative\"].\n"
        "  category: a short label describing which aspect the comment is mainly about. "
        "Prefer one of: \"Delivery\", \"Product Quality\", \"Customer Service\", "
        "\"Website/App\", \"Pricing\". If none fit, use \"Other\".\n"
        "  themes: a list of 1–3 short key phrases capturing the main ideas in the comment.\n\n"
        "Important:\n"
        "- Return ONLY JSON, no extra text, markdown, or explanations.\n"
        "- sentiment MUST be one of the three options above.\n"
    )

    user_prompt = f"Comment: {comment}"

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
        data = resp.json()

        if "choices" in data and data["choices"]:
            raw_text = data["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(raw_text)
            except Exception:
                # Model replied but not as JSON → surface raw text in themes
                return {
                    "sentiment": "Error",
                    "category": "Error",
                    "themes": [raw_text],
                }
        else:
            return {
                "sentiment": "Missing",
                "category": "Missing",
                "themes": ["Unexpected response format"],
            }

    except Exception as e:
        return {
            "sentiment": "Error",
            "category": "Error",
            "themes": [str(e)],
        }

    # Normalise & guard against weird shapes
    sentiment = normalize_sentiment(parsed.get("sentiment", ""))
    category = normalize_category(parsed.get("category", ""))
    themes = parsed.get("themes") or []
    if isinstance(themes, str):
        themes = [themes]

    return {"sentiment": sentiment, "category": category, "themes": themes}


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
            "Add it to a `.env` file or your environment before running the app."
        )
        return

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file with a `comment` column",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is None:
        st.info("Upload a dataset to begin the analysis.")
        return

    # Read file
    try:
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
            analysis = analyze_comment(c)
            rows.append(
                {
                    "Comment": c,
                    "Sentiment": analysis.get("sentiment", ""),
                    "Category": analysis.get("category", ""),
                    "Key Themes": ", ".join(analysis.get("themes", [])),
                }
            )
            progress_bar.progress(i / total)

        st.session_state["results_df"] = pd.DataFrame(rows)
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

        # --- Tabs: table + summary ---
        tab_table, tab_summary = st.tabs(["Results table", "Summary dashboard"])

        with tab_table:
            st.subheader("Structured output")
            st.dataframe(result_df, use_container_width=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download enriched CSV",
                data=csv_bytes,
                file_name="categorized_comments.csv",
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

    # Also show as a small table
            cat_table = (
                cat_counts.rename("Count")
                .reset_index()
                .rename(columns={"index": "Category"})
            )
            st.dataframe(cat_table, use_container_width=True)
            st.markdown("")

    # 3) Category vs Sentiment pivot (counts + row-wise percentages)
            st.subheader("Category vs sentiment (pivot)")

    # Pivot table: rows = Category, columns = Sentiment, values = counts
            pivot_counts = pd.crosstab(result_df["Category"], result_df["Sentiment"])

            st.markdown("**Counts**")
            st.dataframe(pivot_counts, use_container_width=True)

    # Row-wise percentages
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
            return  # stop rendering here if not authenticated

    main_app()


main()
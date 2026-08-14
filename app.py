from __future__ import annotations

from datetime import datetime, timedelta
from html import escape
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from modules.retriver import get_answer, reset_cache as reset_retriever_cache
except Exception as exc:
    get_answer = None
    reset_retriever_cache = None
    RETRIEVER_IMPORT_ERROR = exc
else:
    RETRIEVER_IMPORT_ERROR = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "college_faq.csv"
FAQ_FILENAME = "college_faq.csv"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
SAMPLE_QUESTIONS_PATH = PROJECT_ROOT / "data" / "sample_questions.csv"
NOTIFICATIONS_PATH = PROJECT_ROOT / "data" / "admin_notifications.json"
EMBEDDING_DIMENSION = 384

st.set_page_config(page_title="CampusAI | Dashboard Overview", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
    :root { --navy:#0c1930; --ink:#17233b; --muted:#6d7890; --line:#e7ebf2; --blue:#3b6ff5; --green:#36bd61; --orange:#f6a915; --pink:#e94f88; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink); }
    .stApp { background:#f8fafc; }
    [data-testid="stAppViewContainer"] > .main .block-container { max-width:1600px; padding:1.4rem 2rem 2rem; }
    [data-testid="stSidebar"] { background:var(--navy); border-right:0; }
    [data-testid="stSidebar"] * { color:#eaf0ff !important; }
    [data-testid="stSidebar"] .block-container { padding:1.25rem 1rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:8px; padding:.58rem .65rem; margin:.15rem 0; color:#dbe5fb !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#172c53; }
    .brand { padding:.2rem .45rem 1.45rem; border-bottom:1px solid #253657; margin-bottom:1.05rem; }
    .brand-name { font-family:'Space Grotesk'; font-size:1.35rem; font-weight:700; color:white; }
    .brand-name span { color:#55c9ff; }
    .brand-subtitle { color:#aab9d5; font-size:.75rem; margin-top:.25rem; }
    .sidebar-help { background:#173b7a; border:1px solid #28539e; border-radius:11px; padding:1rem; margin-top:8rem; }
    .sidebar-help strong { color:white; display:block; margin-bottom:.3rem; }
    .sidebar-help p { color:#d6e2ff; font-size:.78rem; line-height:1.5; margin:0 0 .75rem; }
    .admin-box { border:1px solid #2a3c5c; border-radius:11px; padding:.75rem; margin-top:1rem; color:white; }
    .topline { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem; }
    .topline h1 { margin:0; font-size:1.55rem; }
    .topline p { margin:.3rem 0 0; color:var(--muted); font-size:.84rem; }
    .date-chip { background:white; border:1px solid var(--line); padding:.55rem .8rem; border-radius:8px; color:#43516a; font-size:.78rem; }
    .dashboard-card { background:white; border:1px solid var(--line); border-radius:11px; box-shadow:0 2px 9px rgba(28,48,83,.035); padding:1rem; height:100%; }
    .stat-card { min-height:112px; display:flex; align-items:center; gap:.85rem; }
    .stat-icon { width:46px; height:46px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1.3rem; flex:0 0 46px; }
    .stat-label { color:#657088; font-size:.76rem; margin-bottom:.25rem; }
    .stat-value { color:var(--ink); font-family:'Space Grotesk'; font-size:1.55rem; font-weight:700; }
    .stat-note { color:#3aa367; font-size:.7rem; margin-top:.25rem; }
    .muted-note { color:var(--muted); font-size:.72rem; }
    .card-title { font-size:.88rem; font-weight:700; color:var(--ink); margin-bottom:.8rem; }
    .question-row { display:flex; justify-content:space-between; align-items:center; gap:.55rem; padding:.62rem 0; border-bottom:1px solid #f0f2f6; font-size:.77rem; color:#344158; }
    .question-row:last-child { border-bottom:0; }
    .rank-pill { background:#f1efff; color:#6450c7; padding:.25rem .45rem; border-radius:5px; font-size:.7rem; font-weight:700; }
    .status { border-radius:10px; padding:.22rem .42rem; font-size:.65rem; font-weight:600; }
    .status-answered { background:#e7f8ed; color:#2b9a58; }
    .status-unanswered { background:#ffe8e8; color:#d84d4d; }
    .review-card { background:#fff4f4; border:1px solid #ffe2e2; border-radius:9px; padding:.72rem; margin-bottom:.55rem; }
    .review-card p { color:#344158; margin:0 0 .2rem; font-size:.75rem; }
    .review-card small { color:#8c96a9; font-size:.67rem; }
    .section-space { margin-top:1rem; }
    .empty { text-align:center; color:var(--muted); padding:1.5rem .5rem; font-size:.8rem; }
    .answer-card { border-left:4px solid var(--blue); background:#f5f8ff; border-radius:8px; padding:1rem; line-height:1.65; color:#32415b; }
    .result-meta { color:#61708b; font-size:.78rem; margin-top:.7rem; }
    .stButton > button { border-radius:7px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def resolve_faq_path() -> Path:
    """Resolve the FAQ file without relying on a machine-specific absolute path."""
    configured = os.getenv("CAMPUSAI_FAQ_PATH", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        PROJECT_ROOT / "data" / FAQ_FILENAME,
        Path.cwd() / "data" / FAQ_FILENAME,
        Path.cwd() / FAQ_FILENAME,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0] if candidates else PROJECT_ROOT / "data" / FAQ_FILENAME


@st.cache_data(show_spinner=False)
def load_faq_data() -> tuple[pd.DataFrame | None, str | None]:
    faq_path = resolve_faq_path()
    if not faq_path.exists():
        return None, (
            f"FAQ data file was not found. Place `{FAQ_FILENAME}` inside "
            f"`{PROJECT_ROOT / 'data'}` or set the CAMPUSAI_FAQ_PATH environment variable. "
            f"Currently checked: `{faq_path}`"
        )
    try:
        frame = pd.read_csv(faq_path, encoding="utf-8-sig")
    except Exception as exc:
        return None, f"Could not read the FAQ CSV: {exc}"
    if frame.empty:
        return None, "The FAQ CSV is empty."
    normalized = {str(c).strip().lower(): c for c in frame.columns}
    missing = {"question", "answer"} - set(normalized)
    if missing:
        return None, "Missing required CSV column(s): " + ", ".join(sorted(missing))
    rename = {original: name for name, original in normalized.items() if name in {"question", "answer", "category", "source"}}
    frame = frame.rename(columns=rename).copy()
    for column in ("category", "source"):
        if column not in frame.columns:
            frame[column] = "Not specified"
    for column in ("question", "answer", "category", "source"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    # The supplied export contains placeholder rows after the final FAQ.
    # Exclude rows with no question and no answer from dashboard metrics.
    frame = frame[(frame["question"] != "") | (frame["answer"] != "")].reset_index(drop=True)
    return frame, None


@st.cache_data(show_spinner=False)
def load_sample_questions() -> pd.DataFrame:
    if not SAMPLE_QUESTIONS_PATH.exists():
        return pd.DataFrame(columns=["question", "category", "answer", "confidence", "source", "time", "answer"])
    try:
        frame = pd.read_csv(SAMPLE_QUESTIONS_PATH)
        frame["asked_at"] = pd.to_datetime(frame["asked_at"], errors="coerce")
        frame["found"] = frame["found"].astype(str).str.lower().eq("true")
        frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0).clip(0, 1)
        frame["source"] = frame.get("source", "").fillna("").astype(str)
        return frame.rename(columns={"asked_at": "time", "found": "answer"})
    except Exception:
        return pd.DataFrame(columns=["question", "category", "answer", "confidence", "source", "time"])


def init_state() -> None:
    if "query_log" not in st.session_state:
        sample = load_sample_questions()
        st.session_state.query_log = sample[["question", "category", "answer", "time"]].to_dict("records") if not sample.empty else []
        st.session_state.sample_history_loaded = not sample.empty
    if "feedback" not in st.session_state:
        st.session_state.feedback = {"helpful": 0, "not_helpful": 0}
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Overview"


def load_notifications() -> list[dict[str, Any]]:
    if not NOTIFICATIONS_PATH.exists():
        return []
    try:
        with NOTIFICATIONS_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def save_notifications(notifications: list[dict[str, Any]]) -> None:
    NOTIFICATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTIFICATIONS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(notifications[-200:], handle, ensure_ascii=False, indent=2)


def create_admin_notification(question: str, result: dict[str, Any]) -> None:
    notifications = load_notifications()
    duplicate = any(item.get("question", "").strip().lower() == question.strip().lower() and item.get("status") == "pending" for item in notifications)
    if duplicate:
        return
    notifications.append({
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "question": question.strip(),
        "category": str(result.get("category") or "Uncategorized"),
        "confidence": round(answer_confidence(result), 4),
        "status": "pending",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    save_notifications(notifications)


def save_faq_data(frame: pd.DataFrame) -> None:
    path = resolve_faq_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    if "id" not in output.columns and "ID" in output.columns:
        output = output.rename(columns={"ID": "id"})
    rename = {"id": "ID", "question": "Question", "answer": "Answer", "category": "Category", "source": "Source"}
    output = output.rename(columns=rename)
    preferred = [column for column in ["ID", "Question", "Answer", "Category", "Source"] if column in output.columns]
    output[preferred].to_csv(path, index=False, encoding="utf-8-sig")
    load_faq_data.clear()
    if reset_retriever_cache:
        reset_retriever_cache()


def next_faq_id(frame: pd.DataFrame) -> int:
    id_column = "id" if "id" in frame.columns else "ID" if "ID" in frame.columns else None
    if id_column is None:
        return len(frame) + 1
    numeric = pd.to_numeric(frame[id_column], errors="coerce").dropna()
    return int(numeric.max()) + 1 if not numeric.empty else len(frame) + 1


def answer_confidence(result: dict[str, Any]) -> float:
    try:
        value = float(result.get("confidence", 0))
        return max(0, min(1, value if value <= 1 else value / 100))
    except (TypeError, ValueError):
        return 0.0


def log_query(question: str, result: dict[str, Any]) -> None:
    record = {"question": question, "category": result.get("category", "Not specified"), "answer": bool(result.get("found", False)), "time": datetime.now()}
    st.session_state.query_log.insert(0, record)
    st.session_state.query_log = st.session_state.query_log[:100]


def safe_text(value: Any, fallback: str = "Not available") -> str:
    if value is None or str(value).strip() == "":
        return fallback
    return escape(str(value))


def stat_card(icon: str, label: str, value: Any, note: str, color: str) -> None:
    st.markdown(f'<div class="dashboard-card stat-card"><div class="stat-icon" style="background:{color}22">{icon}</div><div><div class="stat-label">{escape(label)}</div><div class="stat-value">{escape(str(value))}</div><div class="stat-note">{escape(note)}</div></div></div>', unsafe_allow_html=True)


def bar_or_empty(frame: pd.DataFrame, column: str, title: str) -> None:
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    if frame.empty:
        st.markdown('<div class="empty">No records available yet.</div>', unsafe_allow_html=True)
        return
    counts = frame[column].value_counts().rename_axis(column).reset_index(name="Count")
    if px:
        fig = px.bar(counts, x=column, y="Count", color=column, color_discrete_sequence=["#4169e1", "#43bd63", "#f2a915", "#7956d8", "#e44e86", "#39abc7" ])
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=5,b=0), showlegend=False, plot_bgcolor="white", paper_bgcolor="white", font=dict(size=10,color="#647086"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.bar_chart(counts.set_index(column))


def render_donut(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.markdown('<div class="empty">No category data available.</div>', unsafe_allow_html=True)
        return
    counts = frame["category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    if px:
        fig = px.pie(counts, values="Count", names="Category", hole=.62, color_discrete_sequence=["#4169e1", "#43bd63", "#f2a915", "#7956d8", "#e44e86", "#39abc7", "#9aa4b5"])
        fig.update_layout(height=260, margin=dict(l=0,r=0,t=0,b=0), showlegend=True, legend=dict(font=dict(size=10)), paper_bgcolor="white", annotations=[dict(text=str(int(counts.Count.sum())), x=.5, y=.5, font_size=18, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.bar_chart(counts.set_index("Category"))


def sidebar() -> str:
    page_labels = ["⌂  Overview", "▣  Student Assistant", "▤  FAQ Management", "?  Unanswered Questions", "▥  Analytics", "⚙  Settings"]
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-name">🎓 Campus<span>AI</span></div><div class="brand-subtitle">College Knowledge Assistant</div></div>', unsafe_allow_html=True)
        pending_notifications = sum(1 for item in load_notifications() if item.get("status") == "pending")
        page_labels[3] = f"?  Unanswered Questions ({pending_notifications})" if pending_notifications else "?  Unanswered Questions"
        chosen = st.radio("Navigation", page_labels, label_visibility="collapsed", key="navigation")
        st.markdown('<div class="sidebar-help"><strong>Need Help?</strong><p>CampusAI is here to help students 24/7.</p></div>', unsafe_allow_html=True)
        st.button("View Documentation", use_container_width=True)
        st.markdown('<div class="admin-box"><strong>◉ &nbsp;Admin</strong><br><small>Administrator</small></div>', unsafe_allow_html=True)
    page_name = chosen.split("  ", 1)[-1]
    return page_name.replace(" (" + str(pending_notifications) + ")", "") if pending_notifications else page_name


def header(title: str, subtitle: str) -> None:
    left, right = st.columns([5, 1])
    with left:
        st.markdown(f'<div class="topline"><div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div></div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="date-chip">▣ &nbsp; {datetime.now().strftime("%d %b %Y")}⌄</div>', unsafe_allow_html=True)


def overview(df: pd.DataFrame | None) -> None:
    header("Dashboard Overview", "Welcome back, Admin")
    logs = st.session_state.query_log
    answered = sum(1 for row in logs if row["answer"])
    unanswered = len(logs) - answered
    answer_rate = (answered / len(logs) * 100) if logs else None
    helpful = st.session_state.feedback["helpful"]
    feedback_total = helpful + st.session_state.feedback["not_helpful"]
    feedback_rate = (helpful / feedback_total * 100) if feedback_total else None
    cards = st.columns(5)
    with cards[0]: stat_card("▣", "Total FAQs", len(df) if df is not None else 0, "From FAQ database", "#7057dc")
    with cards[1]: stat_card("☷", "Total Questions", len(logs), "This session", "#3b8eea")
    with cards[2]: stat_card("↗", "Answer Rate", f"{answer_rate:.1f}%" if answer_rate is not None else "—", "Based on searches" if answer_rate is not None else "No searches yet", "#36bd61")
    with cards[3]: stat_card("♙", "Unanswered", unanswered, "Needs review" if unanswered else "All answered", "#f6a915")
    with cards[4]: stat_card("♡", "Helpful Feedback", f"{feedback_rate:.1f}%" if feedback_rate is not None else "—", "From this session" if feedback_rate is not None else "No feedback yet", "#e94f88")

    st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
    upper = st.columns([1.15, 1.4, 1.1])
    with upper[0]:
        st.markdown('<div class="dashboard-card"><div class="card-title">Questions by Category</div>', unsafe_allow_html=True)
        render_donut(df if df is not None else pd.DataFrame(columns=["category"]))
        st.markdown('</div>', unsafe_allow_html=True)
    with upper[1]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        if logs:
            activity = pd.DataFrame(logs)
            activity["status"] = activity["answer"].map({True: "Answered", False: "Unanswered"})
            st.markdown('<div class="card-title">Answer Rate Over Time</div>', unsafe_allow_html=True)
            counts = activity["status"].value_counts().rename_axis("Status").reset_index(name="Count")
            if px:
                fig = px.bar(counts, x="Status", y="Count", color="Status", color_discrete_map={"Answered": "#36bd61", "Unanswered": "#f6a915"})
                fig.update_layout(height=250, margin=dict(l=0,r=0,t=5,b=0), showlegend=False, plot_bgcolor="white", paper_bgcolor="white", font=dict(size=10,color="#647086"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.bar_chart(counts.set_index("Status"))
        else:
            st.markdown('<div class="card-title">Answer Rate Over Time</div><div class="empty">Ask questions in Student Assistant to build live activity.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with upper[2]:
        st.markdown('<div class="dashboard-card"><div class="card-title">Top Frequently Asked Questions</div>', unsafe_allow_html=True)
        if logs:
            counts = pd.DataFrame(logs)["question"].value_counts().head(5)
            for index, (question, count) in enumerate(counts.items(), 1):
                st.markdown(f'<div class="question-row"><span>{index}. {safe_text(question)}</span><span class="rank-pill">{count}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty">No questions have been asked yet.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    lower = st.columns([1.4, 1, 1])
    with lower[0]:
        st.markdown('<div class="dashboard-card"><div class="card-title">Recent Questions</div>', unsafe_allow_html=True)
        if logs:
            for row in logs[:5]:
                status = "Answered" if row["answer"] else "Unanswered"
                css = "status-answered" if row["answer"] else "status-unanswered"
                st.markdown(f'<div class="question-row"><span>{safe_text(row["question"])}</span><span class="status {css}">{status}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty">Recent questions will appear here after a search.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with lower[1]:
        st.markdown('<div class="dashboard-card"><div class="card-title">Recent Unanswered Questions</div>', unsafe_allow_html=True)
        unresolved = [row for row in logs if not row["answer"]][:3]
        if unresolved:
            for row in unresolved:
                st.markdown(f'<div class="review-card"><p>{safe_text(row["question"])}</p><small>Needs review</small></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty">No unanswered questions in this session.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with lower[2]:
        st.markdown('<div class="dashboard-card"><div class="card-title">Feedback Summary</div>', unsafe_allow_html=True)
        if feedback_total:
            st.metric("Helpful responses", helpful)
            st.progress(helpful / feedback_total, text=f"{feedback_rate:.1f}% helpful")
            st.caption(f"Not helpful: {st.session_state.feedback['not_helpful']}")
        else:
            st.markdown('<div class="empty">Feedback will appear after answering a question.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def assistant_page() -> None:
    header("Student Assistant", "Ask questions about college facilities, fees, hostel, library and examinations.")
    question = st.text_area("Student question", placeholder="Example: How can I apply for hostel admission?", height=120)
    if st.button("🔍 Search CampusAI", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
        elif get_answer is None:
            st.error(f"The existing retrieval module could not be imported: {RETRIEVER_IMPORT_ERROR}")
        else:
            try:
                result = get_answer(question.strip())
                if not isinstance(result, dict):
                    raise ValueError("The retriever must return a dictionary.")
                log_query(question.strip(), result)
                if result.get("found"):
                    st.success("Relevant FAQ found")
                    st.markdown(f'<div class="answer-card">{safe_text(result.get("answer"))}</div>', unsafe_allow_html=True)
                    meta = st.columns(3)
                    meta[0].metric("Category", safe_text(result.get("category")))
                    meta[1].metric("Source", safe_text(result.get("source")))
                    meta[2].metric("Confidence", f"{answer_confidence(result) * 100:.1f}%")
                    st.progress(answer_confidence(result), text="Semantic retrieval confidence")
                    feedback = st.columns(2)
                    with feedback[0]:
                        if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.query_log)}"):
                            st.session_state.feedback["helpful"] += 1
                    with feedback[1]:
                        if st.button("👎 Not helpful", key=f"not_helpful_{len(st.session_state.query_log)}"):
                            st.session_state.feedback["not_helpful"] += 1
                else:
                    create_admin_notification(question.strip(), result)
                    st.warning("No reliable answer found")
                    st.info("The administrator has been notified. Try asking the question using different words, or check Unanswered Questions for review status.")
            except Exception as exc:
                st.error(f"Search failed safely without changing the backend: {exc}")


def faq_management(df: pd.DataFrame | None) -> None:
    header("FAQ Management", "Administrators can add, edit, delete, search and download FAQ records.")
    if df is None:
        st.warning("FAQ management requires a valid FAQ CSV.")
        return

    add_tab, edit_tab, browse_tab = st.tabs(["➕ Add FAQ", "✏️ Edit or Delete", "📚 Browse FAQs"])
    with add_tab:
        with st.form("add_faq_form", clear_on_submit=True):
            question = st.text_input("Question", placeholder="Example: How can I apply for hostel leave?")
            answer = st.text_area("Answer", height=150)
            fields = st.columns(2)
            with fields[0]: category = st.text_input("Category", placeholder="Hostel")
            with fields[1]: source = st.text_input("Source", value="Student Handbook 2025-2026")
            submitted = st.form_submit_button("Save FAQ", type="primary")
        if submitted:
            if not question.strip() or not answer.strip() or not category.strip():
                st.error("Question, answer, and category are required.")
            else:
                id_column = "id" if "id" in df.columns else "ID"
                new_row = {id_column: next_faq_id(df), "question": question.strip(), "answer": answer.strip(), "category": category.strip(), "source": source.strip() or "Admin Added"}
                save_faq_data(pd.concat([df, pd.DataFrame([new_row])], ignore_index=True))
                st.success("FAQ added successfully. The retriever index was refreshed.")
                st.rerun()

    with edit_tab:
        if df.empty:
            st.info("There are no FAQ records to edit.")
        else:
            choices = [f"{row.get('id', index + 1)} — {row['question']}" for index, row in df.iterrows()]
            selected = st.selectbox("Select FAQ record", choices)
            selected_index = choices.index(selected)
            selected_row = df.iloc[selected_index]
            with st.form("edit_faq_form"):
                edited_question = st.text_input("Question", value=str(selected_row.get("question", "")))
                edited_answer = st.text_area("Answer", value=str(selected_row.get("answer", "")), height=150)
                edit_fields = st.columns(2)
                with edit_fields[0]: edited_category = st.text_input("Category", value=str(selected_row.get("category", "")))
                with edit_fields[1]: edited_source = st.text_input("Source", value=str(selected_row.get("source", "")))
                save_changes = st.form_submit_button("Save Changes", type="primary")
            action_columns = st.columns(2)
            with action_columns[0]:
                if save_changes:
                    if not edited_question.strip() or not edited_answer.strip() or not edited_category.strip():
                        st.error("Question, answer, and category are required.")
                    else:
                        updated = df.copy()
                        updated.loc[selected_index, ["question", "answer", "category", "source"]] = [edited_question.strip(), edited_answer.strip(), edited_category.strip(), edited_source.strip() or "Admin Edited"]
                        save_faq_data(updated)
                        st.success("FAQ updated successfully. The retriever index was refreshed.")
                        st.rerun()
            with action_columns[1]:
                if st.button("Delete Selected FAQ", type="secondary"):
                    save_faq_data(df.drop(df.index[selected_index]).reset_index(drop=True))
                    st.success("FAQ deleted successfully. The retriever index was refreshed.")
                    st.rerun()

    with browse_tab:
        filters = st.columns([2, 1, 1])
        with filters[0]: text = st.text_input("Search question or answer", key="browse_search")
        with filters[1]: category = st.selectbox("Category", ["All"] + sorted(df.category.unique().tolist()), key="browse_category")
        with filters[2]: source = st.selectbox("Source", ["All"] + sorted(df.source.unique().tolist()), key="browse_source")
        view = df.copy()
        if text.strip():
            needle = text.lower().strip()
            view = view[view.question.str.lower().str.contains(needle, na=False) | view.answer.str.lower().str.contains(needle, na=False)]
        if category != "All": view = view[view.category == category]
        if source != "All": view = view[view.source == source]
        st.caption(f"Showing {len(view)} of {len(df)} FAQs")
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download filtered FAQ data", view.to_csv(index=False).encode(), "campusai_faq.csv", "text/csv")


def unanswered_page() -> None:
    notifications = load_notifications()
    pending = [item for item in notifications if item.get("status") == "pending"]
    header("Unanswered Questions", f"Admin notifications for questions below the reliable-answer confidence threshold. Pending: {len(pending)}")
    if not pending:
        st.success("No unanswered-question notifications are waiting for review.")
    for item in pending:
        with st.container(border=True):
            st.markdown(f"**{item.get('question', '')}**")
            st.caption(f"Category: {item.get('category', 'Uncategorized')}  •  Confidence: {float(item.get('confidence', 0)) * 100:.1f}%  •  Created: {item.get('created_at', '')}")
            actions = st.columns(3)
            with actions[0]:
                if st.button("Mark reviewed", key=f"notification_review_{item['id']}"):
                    item["status"] = "reviewed"
                    save_notifications(notifications)
                    st.rerun()
            with actions[1]:
                if st.button("Dismiss", key=f"notification_dismiss_{item['id']}"):
                    item["status"] = "dismissed"
                    save_notifications(notifications)
                    st.rerun()
            with actions[2]:
                st.caption("Add the answer in FAQ Management")

    old_unresolved = [row for row in st.session_state.query_log if not row["answer"]]
    if old_unresolved:
        st.markdown("### Current-session unresolved searches")
        st.dataframe(pd.DataFrame(old_unresolved), use_container_width=True, hide_index=True)


def analytics_page(df: pd.DataFrame | None) -> None:
    header("Analytics", "Explore FAQ coverage, source distribution, confidence and question activity with interactive charts.")
    if df is None:
        st.warning("Analytics require a valid FAQ CSV.")
        return

    sample_history = load_sample_questions()
    live_history = pd.DataFrame(st.session_state.query_log)
    if not live_history.empty:
        live_history["time"] = pd.to_datetime(live_history["time"], errors="coerce")
        live_history["answer"] = live_history["answer"].astype(bool)
    history_options = {"Current session": live_history, "Sample question history": sample_history}
    history_choice = st.radio("Question history source", list(history_options), horizontal=True, help="Sample history is provided for demonstration. Current session contains questions asked through Student Assistant.")
    history = history_options[history_choice].copy()

    top_metrics = st.columns(6)
    valid_faqs = int(((df.question != "") & (df.answer != "")).sum())
    answered_count = int(history["answer"].sum()) if not history.empty else 0
    total_questions = len(history)
    answer_rate = answered_count / total_questions * 100 if total_questions else 0
    top_metrics[0].metric("Total FAQs", len(df))
    top_metrics[1].metric("Valid FAQs", valid_faqs)
    top_metrics[2].metric("Categories", df.category.nunique())
    top_metrics[3].metric("Sources", df.source.nunique())
    top_metrics[4].metric("Questions", total_questions)
    top_metrics[5].metric("Answer Rate", f"{answer_rate:.1f}%" if total_questions else "—")

    st.markdown("### FAQ Knowledge Base")
    faq_filters = st.columns([1, 1, 2])
    with faq_filters[0]: selected_category = st.selectbox("Category filter", ["All categories"] + sorted(df.category.unique().tolist()), key="analytics_category")
    with faq_filters[1]: selected_source = st.selectbox("Source filter", ["All sources"] + sorted(df.source.unique().tolist()), key="analytics_source")
    with faq_filters[2]: metric_choice = st.selectbox("Chart metric", ["FAQ count", "Average answer length"], key="analytics_metric")
    filtered = df.copy()
    if selected_category != "All categories": filtered = filtered[filtered.category == selected_category]
    if selected_source != "All sources": filtered = filtered[filtered.source == selected_source]

    chart_row = st.columns(2)
    with chart_row[0]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">FAQ Distribution by Category</div>', unsafe_allow_html=True)
        category_counts = filtered.groupby("category", as_index=False).size().rename(columns={"size": "FAQ count"})
        if px and not category_counts.empty:
            fig = px.pie(category_counts, values="FAQ count", names="category", hole=.58, color_discrete_sequence=["#4169e1", "#43bd63", "#f2a915", "#7956d8", "#e44e86", "#39abc7", "#9aa4b5"])
            fig.update_layout(height=330, margin=dict(l=0,r=0,t=5,b=0), paper_bgcolor="white", legend=dict(font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
        else: st.info("No category data is available for the selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)
    with chart_row[1]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Source Coverage</div>', unsafe_allow_html=True)
        if metric_choice == "FAQ count":
            source_chart = filtered.groupby("source", as_index=False).size().rename(columns={"size": "Value"})
            y_title = "FAQ count"
        else:
            source_chart = filtered.assign(answer_length=filtered.answer.str.len()).groupby("source", as_index=False)["answer_length"].mean().rename(columns={"answer_length": "Value"})
            y_title = "Average answer length"
        if px and not source_chart.empty:
            fig = px.bar(source_chart, x="source", y="Value", color="source", text_auto=True, color_discrete_sequence=["#4169e1", "#43bd63", "#f2a915", "#7956d8", "#e44e86", "#39abc7"])
            fig.update_layout(height=330, margin=dict(l=0,r=0,t=5,b=0), showlegend=False, paper_bgcolor="white", yaxis_title=y_title, xaxis_title="")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
        else: st.info("No source data is available for the selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Question Activity")
    activity_filters = st.columns([1, 1, 1])
    if not history.empty:
        categories = ["All categories"] + sorted(history.category.dropna().astype(str).unique().tolist())
        with activity_filters[0]: history_category = st.selectbox("Question category", categories, key="history_category")
        with activity_filters[1]: status_filter = st.selectbox("Answer status", ["All statuses", "Answered", "Unanswered"], key="history_status")
        with activity_filters[2]: min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05, key="history_confidence")
        view = history.copy()
        if history_category != "All categories": view = view[view.category.astype(str) == history_category]
        if status_filter != "All statuses": view = view[view.answer == (status_filter == "Answered")]
        if "confidence" in view.columns: view = view[view.confidence.fillna(0) >= min_confidence]

        chart_row_2 = st.columns(2)
        with chart_row_2[0]:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Answer Status</div>', unsafe_allow_html=True)
            status_counts = view.assign(status=view.answer.map({True: "Answered", False: "Unanswered"})).groupby("status", as_index=False).size().rename(columns={"size": "Questions"})
            if px and not status_counts.empty:
                fig = px.bar(status_counts, x="status", y="Questions", color="status", text_auto=True, color_discrete_map={"Answered": "#36bd61", "Unanswered": "#f6a915"})
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=5,b=0), showlegend=False, paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
            else: st.info("No question records match the selected filters.")
            st.markdown('</div>', unsafe_allow_html=True)
        with chart_row_2[1]:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Questions Over Time</div>', unsafe_allow_html=True)
            if not view.empty and view["time"].notna().any():
                daily = view.dropna(subset=["time"]).assign(date=lambda x: x["time"].dt.date).groupby("date", as_index=False).size().rename(columns={"size": "Questions"})
                if px:
                    fig = px.line(daily, x="date", y="Questions", markers=True)
                    fig.update_traces(line_color="#4169e1", marker_color="#4169e1")
                    fig.update_layout(height=300, margin=dict(l=0,r=0,t=5,b=0), paper_bgcolor="white", xaxis_title="", yaxis_title="Questions")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
            else: st.info("No dated question activity matches the selected filters.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Question Detail")
        columns = [column for column in ["question", "category", "answer", "confidence", "source", "time"] if column in view.columns]
        st.dataframe(view[columns].sort_values("time", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No question history is available. Use Student Assistant or choose Sample question history.")


def settings_page() -> None:
    header("Settings", "Runtime information for the CampusAI dashboard.")
    st.write(f"**Project root:** `{PROJECT_ROOT}`")
    st.write(f"**FAQ data:** `{resolve_faq_path()}`")
    st.write(f"**Retriever status:** {'Available' if get_answer else 'Unavailable'}")
    st.write(f"**ChromaDB directory:** `{CHROMA_PATH}`")
    if st.button("Clear current-session analytics"):
        st.session_state.query_log = []
        st.session_state.feedback = {"helpful": 0, "not_helpful": 0}
        st.success("Current-session analytics were cleared.")


init_state()
df, data_error = load_faq_data()
if data_error:
    st.error(data_error)
page = sidebar()
if page == "Overview": overview(df)
elif page == "Student Assistant": assistant_page()
elif page == "FAQ Management": faq_management(df)
elif page == "Unanswered Questions": unanswered_page()
elif page == "Analytics": analytics_page(df)
elif page == "Settings": settings_page()

st.markdown('<div style="margin-top:2rem;color:#9aa3b5;font-size:.72rem">© CampusAI. Built for better education.</div>', unsafe_allow_html=True)

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from mlf_assistant import db
from mlf_assistant.config import get_settings
from mlf_assistant.db import DatabaseUnavailable
from mlf_assistant.documents import DocumentProcessingError, ingest_document, reindex_document
from mlf_assistant.reports import excel_report, pdf_report
from mlf_assistant.schemas import AdminUserCreate, DocumentMetadata
from mlf_assistant.security import (
    authenticate_user,
    hash_admin_password,
    session_is_authorized,
)
from mlf_assistant.speech import SpeechServiceError, synthesize_speech, transcribe_audio
from mlf_assistant.workflow import run_assistant


st.set_page_config(page_title="Ministry Information Assistant", page_icon="MLF", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Literata:opsz,wght@7..72,600;7..72,700&display=swap');
    :root { --ink:#17231d; --forest:#164d3a; --leaf:#367456; --paper:#f6f7f3; --white:#fff; --line:#dce2dc; --muted:#69766d; --coral:#bd553f; --gold:#bd9546; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    .stApp { background:var(--paper); }
    header[data-testid="stHeader"] { background:rgba(246,247,243,.92); }
    [data-testid="stSidebar"] { background:#edf1eb; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div { padding-top:1.6rem; }
    .block-container { max-width:1100px; padding-top:2rem; padding-bottom:4rem; }
    .brand { display:flex; align-items:center; gap:13px; padding:4px 0 24px; border-bottom:1px solid var(--line); margin-bottom:28px; }
    .brand-mark { width:42px; height:42px; display:grid; place-items:center; border-radius:50%; background:var(--forest); color:white; font:500 12px 'DM Mono',monospace; letter-spacing:0; }
    .brand-name { font:700 14px 'DM Sans',sans-serif; line-height:1.25; }
    .brand-name span { display:block; font-size:11px; font-weight:400; color:var(--muted); margin-top:3px; }
    .eyebrow { color:var(--leaf); font:500 11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:0; }
    .hero-title { font:700 42px 'Literata',Georgia,serif; letter-spacing:0; line-height:1.12; margin:9px 0 10px; }
    .hero-copy { max-width:620px; color:#5f6d63; font-size:15px; line-height:1.6; margin-bottom:22px; }
    .status-line { display:inline-flex; align-items:center; gap:8px; border:1px solid var(--line); border-radius:4px; background:white; padding:8px 11px; color:#4f5c53; font:11px 'DM Mono',monospace; }
    .status-dot { width:7px; height:7px; background:var(--gold); border-radius:50%; }
    .section-label { border-top:1px solid var(--line); padding-top:19px; margin:24px 0 13px; font:600 12px 'DM Mono',monospace; color:#59675d; text-transform:uppercase; }
    .citation { border-left:2px solid var(--gold); padding:3px 0 3px 12px; margin:8px 0; }
    .citation-title { font-weight:600; color:var(--ink); font-size:13px; }
    .citation-meta { color:var(--muted); font:11px 'DM Mono',monospace; margin-top:3px; }
    .disclaimer { color:#718076; font-size:11px; border-top:1px solid var(--line); padding-top:14px; margin-top:36px; }
    [data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:4px; padding:14px 16px; }
    [data-testid="stMetricLabel"] { color:var(--muted); font:11px 'DM Mono',monospace; }
    [data-testid="stMetricValue"] { color:var(--forest); font:600 26px 'Literata',Georgia,serif; }
    .stButton > button[kind="primary"] { background:var(--forest); border-color:var(--forest); }
    .stButton > button { border-radius:4px; }
    [data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:5px; background:white; }
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] { border-radius:4px; }
    @media(max-width:700px) { .block-container {padding:1.2rem 1rem 3rem;} .hero-title {font-size:32px;} }
    </style>
    """,
    unsafe_allow_html=True,
)


def brand() -> None:
    st.markdown(
        '<div class="brand"><div class="brand-mark">MLF</div><div class="brand-name">Ministry of Livestock and Fisheries<span>United Republic of Tanzania</span></div></div>',
        unsafe_allow_html=True,
    )


def assistant_page() -> None:
    brand()
    st.markdown('<div class="eyebrow">Public information desk</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Ask the Ministry.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Find guidance on livestock, fisheries, laws and government services. Answers are grounded in uploaded Ministry documents, with source references.</div>',
        unsafe_allow_html=True,
    )
    provider_ready = bool(get_settings().llm_provider and get_settings().llm_model)
    db_ready = _database_ready()
    service_label = "Evidence service available" if db_ready and provider_ready else "Evidence service not configured"
    dot_color = "#367456" if db_ready and provider_ready else "#bd9546"
    st.markdown(
        f'<div class="status-line"><span class="status-dot" style="background:{dot_color}"></span>{service_label}</div>',
        unsafe_allow_html=True,
    )
    preferred = st.radio(
        "Response language", ["Auto-detect", "Kiswahili", "English"],
        horizontal=True, label_visibility="collapsed",
    )
    preference = {"Auto-detect": "auto", "Kiswahili": "sw", "English": "en"}[preferred]
    settings = get_settings()
    speech_ready = settings.llm_provider == "openai" and bool(settings.llm_api_key)

    history = st.session_state.setdefault("messages", [])
    for index, message in enumerate(history):
        with st.chat_message("user"):
            st.markdown(message["question"])
        result = message["result"]
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            if result.get("speech_audio"):
                st.audio(result["speech_audio"], format="audio/mp3")
            _render_citations(result)
            if result.get("needs_human"):
                _render_escalation(message, index)
            if result.get("usage_id"):
                rating = result.get("feedback")
                if rating is not None:
                    st.caption("Thank you for your feedback.")
                else:
                    useful, not_useful = st.columns(2)
                    if useful.button("Helpful", key=f"feedback_yes_{result['usage_id']}"):
                        _save_feedback(message, helpful=True)
                    if not_useful.button("Not helpful", key=f"feedback_no_{result['usage_id']}"):
                        _save_feedback(message, helpful=False)

    st.markdown("**Ask by voice**")
    voice_audio = st.audio_input(
        "Record a question",
        key="voice_question_audio",
        help="Record your question, then stop to send it for transcription.",
        disabled=not speech_ready,
    )
    speak_responses = st.checkbox(
        "Include an audio reply",
        value=False,
        disabled=not speech_ready,
        help="Audio replies use the configured OpenAI speech model and can incur API charges.",
    )
    if not speech_ready:
        st.caption("Voice features require the OpenAI provider and an API key. You can still type questions below.")
    else:
        st.caption("Record a question and stop to submit it. Audio is sent to the configured OpenAI speech API for transcription.")

    question = st.chat_input("Ask about livestock, fisheries, laws or a Ministry service")
    if not question and voice_audio is not None:
        audio_bytes = voice_audio.getvalue()
        audio_digest = hashlib.sha256(audio_bytes).hexdigest()
        if audio_digest != st.session_state.get("last_voice_audio_digest"):
            with st.spinner("Transcribing your question"):
                try:
                    question = transcribe_audio(audio_bytes, language=preference)
                    st.session_state["last_voice_audio_digest"] = audio_digest
                except SpeechServiceError as exc:
                    st.session_state["last_voice_audio_digest"] = audio_digest
                    st.error(str(exc))
    if question:
        with st.spinner("Checking uploaded Ministry documents"):
            conversation = [
                {"role": role, "content": content}
                for item in history[-6:]
                for role, content in (("user", item["question"]), ("assistant", item["result"]["answer"]))
            ]
            result = run_assistant(question, conversation, preference)
            if speak_responses:
                try:
                    result["speech_audio"] = synthesize_speech(result["answer"])
                except SpeechServiceError as exc:
                    st.warning(f"Audio reply unavailable: {exc}")
        history.append({"question": question, "result": result, "voice_id": str(time.time_ns())})
        st.rerun()
    st.markdown(
        '<div class="disclaimer">Official information is provided only when supported by current, successfully processed documents. Do not submit confidential or sensitive personal information.</div>',
        unsafe_allow_html=True,
    )


def _render_citations(result: dict) -> None:
    ids = set(result.get("citations", []))
    cited = [
        item for item in result.get("evidence", [])
        if item.chunk_id in ids or f"{item.document_id}:{item.chunk_id}" in ids
    ]
    if cited:
        with st.expander("Source documents"):
            for item in cited:
                st.markdown(
                    f'<div class="citation"><div class="citation-title">{_escape(item.title)}</div><div class="citation-meta">{_escape(item.document_id)} · v{_escape(item.version)} · chunk {item.chunk_id}</div></div>',
                    unsafe_allow_html=True,
                )


def _render_escalation(message: dict, index: int) -> None:
    with st.expander("Request help from a Ministry officer"):
        with st.form(f"escalate_{index}"):
            name = st.text_input("Name", max_chars=120)
            email = st.text_input("Email address", max_chars=200)
            phone = st.text_input("Phone number", max_chars=40)
            consent = st.checkbox("I consent to the Ministry using these contact details to respond to this enquiry.")
            submitted = st.form_submit_button("Submit enquiry", type="primary")
        if submitted:
            if not consent:
                st.warning("Consent is required before contact details can be submitted.")
            elif not email.strip() and not phone.strip():
                st.warning("Provide an email address or phone number so an officer can respond.")
            else:
                result = message["result"]
                try:
                    reference = db.submit_enquiry(
                        category=result.get("intent", "general"),
                        department=result.get("department") or "Ministry information",
                        language=result.get("language", "en"), question=message["question"],
                        reason=result.get("reason", "Evidence was insufficient"), consent=True,
                        name=name, email=email, phone=phone,
                    )
                    st.success(f"Enquiry submitted. Reference: {reference}")
                except (DatabaseUnavailable, ValueError):
                    st.error("The enquiry was not submitted. Please try again later or contact the Ministry directly.")


def _save_feedback(message: dict, *, helpful: bool) -> None:
    try:
        db.record_feedback(message["result"]["usage_id"], helpful)
        message["result"]["feedback"] = 1 if helpful else -1
        st.rerun()
    except (DatabaseUnavailable, ValueError):
        st.error("Feedback could not be saved right now.")


def admin_page() -> None:
    brand()
    st.markdown('<div class="eyebrow">Restricted workspace</div>', unsafe_allow_html=True)
    try:
        authenticated = _admin_authenticated()
    except DatabaseUnavailable:
        st.error("The account database is unavailable. Administration is temporarily disabled.")
        return
    if not authenticated:
        st.markdown('<h1 class="hero-title">Administration</h1>', unsafe_allow_html=True)
        with st.form("admin_login"):
            username = st.text_input("Administrator username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            try:
                role = authenticate_user(username, password)
                if role:
                    st.session_state.admin_user = username
                    st.session_state.admin_role = role
                    st.session_state.admin_auth_at = time.time()
                    st.rerun()
                st.error("Sign-in failed.")
            except DatabaseUnavailable:
                st.error("The account database is unavailable. Sign-in could not be verified.")
        st.caption("Administrator credentials must be provisioned through the deployment environment.")
        return

    st.markdown('<h1 class="hero-title">Operations</h1>', unsafe_allow_html=True)
    if st.button("Sign out"):
        st.session_state.pop("admin_user", None)
        st.session_state.pop("admin_role", None)
        st.session_state.pop("admin_auth_at", None)
        st.rerun()
    if st.session_state.get("admin_role") == "officer":
        _officer_inbox()
        return
    try:
        metrics = db.dashboard_metrics()
        analytics = db.dashboard_analytics()
        documents = db.list_documents(limit=100)
        enquiries = db.list_enquiries(limit=100)
    except DatabaseUnavailable:
        st.error("The Ministry database is unavailable. Administrative changes are disabled.")
        return

    metric_columns = st.columns(6)
    for column, label, key in zip(
        metric_columns,
        ("Enquiries", "Today", "This month", "Unanswered", "Avg latency", "Escalated"),
        ("total", "today", "month", "unanswered", "avg_latency_ms", "escalated"),
        strict=True,
    ):
        column.metric(label, metrics.get(key, 0))
    st.caption(
        f"{metrics.get('kiswahili', 0)} Kiswahili enquiries · {metrics.get('livestock', 0)} livestock · "
        f"{metrics.get('fisheries', 0)} fisheries · {metrics.get('documents', 0)} uploaded documents · "
        f"{metrics.get('tokens', 0)} tokens · ${float(metrics.get('cost_usd', 0)):.4f} configured cost"
    )

    overview, documents_tab, enquiries_tab, users_tab, config_tab = st.tabs(
        ["Reports", "Documents", "Enquiries", "Users", "AI configuration"]
    )
    with overview:
        st.markdown('<div class="section-label">Operational snapshot</div>', unsafe_allow_html=True)
        _render_dashboard_charts(analytics)
        if enquiries:
            st.dataframe([
                {key: item.get(key) for key in ("reference", "category", "department", "language", "status", "created_at")}
                for item in enquiries[:20]
            ], use_container_width=True, hide_index=True)
        else:
            st.info("No enquiries have been recorded.")
        col_pdf, col_xlsx = st.columns(2)
        col_pdf.download_button("Download PDF report", pdf_report(metrics, enquiries), "ministry-enquiries.pdf", "application/pdf")
        col_xlsx.download_button("Download Excel report", excel_report(metrics, enquiries), "ministry-enquiries.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with documents_tab:
        st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)
        completed_document_id = st.session_state.pop("document_processing_complete", None)
        if completed_document_id:
            st.success(
                "Document processing is finished. The document is indexed and ready "
                "for users to ask questions and get answers from it."
            )
        replacement_target_id = st.session_state.get("replacement_target_id")
        document_by_id = {str(item["id"]): item for item in documents}
        if replacement_target_id not in document_by_id:
            replacement_target_id = None
            st.session_state.pop("replacement_target_id", None)
        if documents:
            for item in documents:
                document_id = str(item["id"])
                details, dates_action, replace_action, delete_action = st.columns([5, 1, 1, 1])
                with details:
                    st.write(item["title"])
                    st.caption(
                        f"{item['category']} · v{item['version']} · "
                        f"{item['ingestion_status']} · effective {item['effective_date'] or 'not set'} · "
                        f"review {item['review_date'] or 'not set'}"
                    )
                with dates_action.popover("Dates", use_container_width=True):
                    with st.form(f"document_dates_{document_id}"):
                        effective_date = st.date_input(
                            "Effective date", value=item["effective_date"]
                        )
                        review_date = st.date_input(
                            "Review date", value=item["review_date"]
                        )
                        save_dates = st.form_submit_button("Save dates")
                    if save_dates:
                        try:
                            db.update_document_dates(
                                document_id,
                                effective_date,
                                review_date,
                                st.session_state.admin_user,
                            )
                            st.success("Document dates updated.")
                            st.rerun()
                        except (DatabaseUnavailable, ValueError) as exc:
                            st.error(str(exc))
                can_replace = (
                    item["approval_status"] == "approved"
                    and item["ingestion_status"] == "ready"
                    and not item["superseded"]
                )
                if replace_action.button(
                    "Replace",
                    key=f"replace_document_{document_id}",
                    disabled=not can_replace,
                    help="Replace this active document with a newly uploaded version."
                    if can_replace
                    else "Only active, indexed documents can be replaced.",
                    use_container_width=True,
                ):
                    st.session_state["replacement_target_id"] = document_id
                    st.rerun()
                with delete_action.popover("Delete", use_container_width=True):
                    st.warning(f"Permanently delete {item['title']} and its indexed content?")
                    if st.button(
                        "Delete permanently",
                        key=f"confirm_delete_document_{document_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            db.delete_document(document_id, st.session_state.admin_user)
                            st.success("Document and indexed content deleted.")
                            st.rerun()
                        except (DatabaseUnavailable, ValueError) as exc:
                            st.error(str(exc))
                st.divider()
        else:
            st.info("No documents have been uploaded.")

        replacement_target = document_by_id.get(replacement_target_id)
        upload_label = (
            f"Replace {replacement_target['title']}"
            if replacement_target
            else "Upload a Ministry document"
        )
        with st.expander(upload_label, expanded=bool(replacement_target)):
            if replacement_target and st.button(
                "Cancel replacement", key="cancel_document_replacement"
            ):
                st.session_state.pop("replacement_target_id", None)
                st.rerun()
            with st.form("upload_document"):
                upload = st.file_uploader("Source file", type=["pdf", "docx", "xlsx", "csv", "pptx", "txt", "md", "html", "xml", "json", "png", "jpg", "jpeg", "tif", "tiff"])
                title = st.text_input("Document title")
                department = st.text_input("Department or issuing unit")
                category = st.selectbox("Category", ["Ministry information", "Livestock", "Fisheries", "Legal and regulatory", "Government service"])
                language = st.selectbox("Language", ["en", "sw", "mixed"])
                authority = st.text_input("Issuing authority")
                version = st.text_input("Version")
                published = st.date_input("Publication date", value=None)
                effective = st.date_input("Effective date", value=None)
                review = st.date_input("Review date", value=None)
                uploaded = st.form_submit_button(
                    "Replace document" if replacement_target else "Process document",
                    type="primary",
                )
            if uploaded:
                try:
                    if not upload:
                        raise DocumentProcessingError("Select a file first.")
                    metadata = DocumentMetadata(
                        title=title, department=department, category=category, language=language,
                        issuing_authority=authority, version=version,
                        publication_date=published,
                        effective_date=effective,
                        review_date=review,
                    )
                    document_id = ingest_document(upload.name, upload.getvalue(), metadata, actor=st.session_state.admin_user)
                    if replacement_target:
                        try:
                            db.replace_document(
                                replacement_target_id, document_id, st.session_state.admin_user
                            )
                        except (DatabaseUnavailable, ValueError) as exc:
                            st.error(
                                "The new document was indexed, but the previous version remains active. "
                                f"Replacement could not be completed: {exc}"
                            )
                            st.stop()
                        st.session_state.pop("replacement_target_id", None)
                    st.session_state["document_processing_complete"] = document_id
                    st.rerun()
                except (DocumentProcessingError, ValueError, DatabaseUnavailable) as exc:
                    st.error(str(exc))
        reindexable = {
            str(item["id"]): item
            for item in documents
            if item["ingestion_status"] == "ready"
        }
        if reindexable:
            reindex_id = st.selectbox(
                "Document to re-index",
                list(reindexable),
                format_func=lambda document_id: (
                    f"{reindexable[document_id]['title']} · "
                    f"Version {reindexable[document_id]['version']}"
                ),
            )
            st.caption(f"Document ID: `{reindex_id}`")
            if st.button("Re-index selected document"):
                try:
                    count = reindex_document(reindex_id)
                    st.success(f"Re-indexed {count} chunks.")
                except (DocumentProcessingError, DatabaseUnavailable, ValueError) as exc:
                    st.error(str(exc))
        else:
            st.caption("No successfully indexed documents are available to re-index.")

    with enquiries_tab:
        st.markdown('<div class="section-label">Enquiry follow-up</div>', unsafe_allow_html=True)
        if enquiries:
            selected_index = st.selectbox(
                "Enquiry reference", range(len(enquiries)),
                format_func=lambda index: f"{enquiries[index]['reference']} · {enquiries[index]['category']} · {enquiries[index]['status']}",
            )
            selected = enquiries[selected_index]
            st.write(selected["question"])
            st.caption(f"Contact consent recorded: {selected['consent']} · {selected['contact_email'] or selected['contact_phone'] or 'No contact supplied'}")
            next_status = st.selectbox("Set status", ["in_review", "resolved", "closed"], index=0)
            if st.button("Update enquiry", type="primary"):
                try:
                    db.update_enquiry_status(selected["reference"], next_status, st.session_state.admin_user)
                    st.success("Enquiry status updated.")
                    st.rerun()
                except (ValueError, DatabaseUnavailable) as exc:
                    st.error(str(exc))
        else:
            st.info("No enquiries have been submitted.")

    with users_tab:
        st.markdown('<div class="section-label">Accounts and roles</div>', unsafe_allow_html=True)
        account_db_available = True
        try:
            users = db.list_users()
        except DatabaseUnavailable:
            st.error("The account database is unavailable. Account management is disabled.")
            users = []
            account_db_available = False
        st.dataframe(users, use_container_width=True, hide_index=True)
        with st.form("create_user"):
            new_username = st.text_input("Username", max_chars=80)
            new_password = st.text_input("Temporary password", type="password", help="At least 14 characters.")
            role = st.selectbox("Role", ["officer", "admin"])
            create = st.form_submit_button(
                "Create account", type="primary", disabled=not account_db_available
            )
        if create:
            try:
                request = AdminUserCreate(username=new_username, password=new_password, role=role)
                password_hash = hash_admin_password(request.password.get_secret_value())
                db.create_user(
                    request.username, password_hash, request.role, st.session_state.admin_user
                )
                st.success("Account created.")
                st.rerun()
            except (ValueError, ValidationError, DatabaseUnavailable) as exc:
                st.error(str(exc))
        if users:
            managed = [user for user in users if user["username"] != st.session_state.admin_user]
            if managed:
                user_index = st.selectbox(
                    "Account", range(len(managed)),
                    format_func=lambda index: f"{managed[index]['username']} · {managed[index]['role']} · {'active' if managed[index]['active'] else 'inactive'}",
                )
                selected_user = managed[user_index]
                activate = not selected_user["active"]
                if st.button(
                    "Activate account" if activate else "Deactivate account",
                    disabled=not account_db_available,
                ):
                    try:
                        db.set_user_active(selected_user["username"], activate, st.session_state.admin_user)
                        st.success("Account status updated.")
                        st.rerun()
                    except (ValueError, DatabaseUnavailable) as exc:
                        st.error(str(exc))

    with config_tab:
        settings = get_settings()
        st.markdown('<div class="section-label">Runtime configuration</div>', unsafe_allow_html=True)
        st.write({
            "LLM provider": settings.llm_provider or "Not configured",
            "LLM model": settings.llm_model or "Not configured",
            "Embeddings": settings.embedding_model,
            "Reranker": settings.reranker_model or "Disabled",
            "Allowed website origins": settings.allowed_origins or "None configured",
            "API key": "Configured" if settings.llm_api_key else "Not configured",
        })
        st.caption("Provider credentials and model changes are managed through the deployment environment; secrets are never displayed here.")


def _render_dashboard_charts(analytics: dict) -> None:
    import pandas as pd

    weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    daily = [
        {**row, "day": row["day"].isoformat()}
        for row in analytics["daily"]
    ]
    hours = [
        {**row, "weekday": weekdays[row["day_of_week"]]}
        for row in analytics["hours"]
    ]

    def chart(data: list[dict], spec: dict, *, height: int = 280) -> None:
        if not data:
            st.info("No data available yet.")
            return
        spec["height"] = height
        st.vega_lite_chart(
            pd.DataFrame(data), spec, use_container_width=True
        )

    activity_col, category_col = st.columns(2)
    with activity_col:
        st.markdown("**User Activity Trend**")
        chart(daily, {
            "mark": {"type": "line", "point": True, "color": "#367456"},
            "encoding": {
                "x": {"field": "day", "type": "temporal", "title": None},
                "y": {"field": "activity", "type": "quantitative", "title": "Questions"},
                "tooltip": [
                    {"field": "day", "type": "temporal", "title": "Date"},
                    {"field": "activity", "type": "quantitative", "title": "Questions"},
                ],
            },
        })
    with category_col:
        st.markdown("**Questions by Category**")
        chart(analytics["categories"], {
            "mark": {"type": "bar", "color": "#bd9546"},
            "encoding": {
                "x": {"field": "questions", "type": "quantitative", "title": "Questions"},
                "y": {"field": "category", "type": "nominal", "sort": "-x", "title": None},
                "tooltip": ["category", "questions"],
            },
        })

    st.markdown("**Peak Usage Hours · East Africa Time · Last 30 Days**")
    chart(hours, {
        "mark": "rect",
        "encoding": {
            "x": {"field": "hour", "type": "ordinal", "title": "Hour of day", "axis": {"labelAngle": 0}},
            "y": {"field": "weekday", "type": "ordinal", "sort": weekdays, "title": None},
            "color": {"field": "questions", "type": "quantitative", "title": "Questions", "scale": {"scheme": "greens"}},
            "tooltip": ["weekday", "hour", "questions"],
        },
    }, height=250)

    documents_col, quality_col = st.columns(2)
    with documents_col:
        st.markdown("**Documents by Category**")
        chart(analytics["documents"], {
            "mark": {"type": "arc", "innerRadius": 55},
            "encoding": {
                "theta": {"field": "documents", "type": "quantitative"},
                "color": {"field": "category", "type": "nominal", "title": "Category"},
                "tooltip": ["category", "documents"],
            },
        })
    with quality_col:
        st.markdown("**Answer Quality Trend · Helpful Feedback**")
        chart(daily, {
            "mark": {"type": "line", "point": True, "color": "#bd553f"},
            "encoding": {
                "x": {"field": "day", "type": "temporal", "title": None},
                "y": {"field": "answer_quality", "type": "quantitative", "title": "Helpful feedback (%)", "scale": {"domain": [0, 100]}},
                "tooltip": [
                    {"field": "day", "type": "temporal", "title": "Date"},
                    {"field": "answer_quality", "type": "quantitative", "title": "Helpful feedback (%)"},
                ],
            },
        })

    st.markdown("**Response Time Trend**")
    chart(daily, {
        "mark": {"type": "line", "point": True, "color": "#164d3a"},
        "encoding": {
            "x": {"field": "day", "type": "temporal", "title": None},
            "y": {"field": "response_time_ms", "type": "quantitative", "title": "Average response time (ms)"},
            "tooltip": [
                {"field": "day", "type": "temporal", "title": "Date"},
                {"field": "response_time_ms", "type": "quantitative", "title": "Average response time (ms)"},
            ],
        },
    })

    st.markdown("**Frequently Unanswered Questions · Consent-Based Enquiries**")
    unanswered = [
        {**row, "question_label": row["question"][:72] + ("..." if len(row["question"]) > 72 else "")}
        for row in analytics["unanswered"]
    ]
    chart(unanswered, {
        "mark": {"type": "bar", "color": "#bd553f"},
        "encoding": {
            "x": {"field": "occurrences", "type": "quantitative", "title": "Enquiries"},
            "y": {"field": "question_label", "type": "nominal", "sort": "-x", "title": None, "axis": {"labelLimit": 320}},
            "tooltip": [
                {"field": "question", "type": "nominal", "title": "Question"},
                {"field": "occurrences", "type": "quantitative", "title": "Enquiries"},
            ],
        },
    }, height=320)


def _admin_authenticated() -> bool:
    username = st.session_state.get("admin_user")
    role = st.session_state.get("admin_role")
    authenticated_at = st.session_state.get("admin_auth_at", 0)
    if session_is_authorized(username, role, authenticated_at):
        return True
    st.session_state.pop("admin_user", None)
    st.session_state.pop("admin_role", None)
    st.session_state.pop("admin_auth_at", None)
    return False


def _officer_inbox() -> None:
    st.markdown('<h1 class="hero-title">Enquiry inbox</h1>', unsafe_allow_html=True)
    try:
        enquiries = db.list_enquiries(limit=200)
    except DatabaseUnavailable:
        st.error("The Ministry database is unavailable. The enquiry inbox cannot be loaded.")
        return
    if not enquiries:
        st.info("No enquiries have been submitted.")
        return
    st.dataframe([
        {key: item.get(key) for key in ("reference", "category", "department", "language", "status", "created_at")}
        for item in enquiries
    ], use_container_width=True, hide_index=True)
    selected_index = st.selectbox(
        "Enquiry reference", range(len(enquiries)),
        format_func=lambda index: f"{enquiries[index]['reference']} · {enquiries[index]['department']} · {enquiries[index]['status']}",
    )
    selected = enquiries[selected_index]
    st.write(selected["question"])
    st.caption(f"Consent recorded: {selected['consent']} · {selected['contact_email'] or selected['contact_phone'] or 'No return contact'}")
    status_value = st.selectbox("Set status", ["in_review", "resolved", "closed"])
    if st.button("Update enquiry", type="primary"):
        try:
            db.update_enquiry_status(selected["reference"], status_value, st.session_state.admin_user)
            st.success("Enquiry status updated.")
            st.rerun()
        except (ValueError, DatabaseUnavailable) as exc:
            st.error(str(exc))


def _database_ready() -> bool:
    try:
        with db.connection() as conn:
            conn.execute("SELECT 1")
        return True
    except DatabaseUnavailable:
        return False


def _escape(value: object) -> str:
    from html import escape

    return escape(str(value))


with st.sidebar:
    st.markdown("### Information services")
    page = st.radio("Workspace", ["Ask the Ministry", "Administration"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("United Republic of Tanzania")

if page == "Ask the Ministry":
    assistant_page()
else:
    admin_page()
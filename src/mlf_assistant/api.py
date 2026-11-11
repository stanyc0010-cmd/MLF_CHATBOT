from __future__ import annotations

from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, ValidationError

from mlf_assistant import db
from mlf_assistant.config import get_settings
from mlf_assistant.db import DatabaseUnavailable
from mlf_assistant.documents import DocumentProcessingError, ingest_document
from mlf_assistant.schemas import (
    AdminUserCreate,
    AskRequest,
    AskResponse,
    DocumentMetadata,
    EscalationRequest,
    UserActivation,
)
from mlf_assistant.security import authenticate_user, hash_admin_password, verify_admin
from mlf_assistant.workflow import run_assistant

app = FastAPI(title="Tanzania MLF Information Assistant", version="0.1.0")
basic_auth = HTTPBasic(auto_error=False)
origins = [origin.strip() for origin in get_settings().allowed_origins.split(",") if origin.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


class EnquiryStatusUpdate(BaseModel):
    status: Literal["in_review", "resolved", "closed"]


def authenticate_admin(
    credentials: Annotated[HTTPBasicCredentials | None, Security(basic_auth)],
) -> str:
    try:
        authenticated = bool(
            credentials and verify_admin(credentials.username, credentials.password)
        )
    except DatabaseUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The account database is unavailable.",
        ) from exc
    if not authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Administrator authentication required.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def authenticate_staff(
    credentials: Annotated[HTTPBasicCredentials | None, Security(basic_auth)],
) -> str:
    try:
        role = authenticate_user(credentials.username, credentials.password) if credentials else None
    except DatabaseUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The account database is unavailable.",
        ) from exc
    if role not in {"admin", "officer"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Ministry account authentication required.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.get("/health")
def health() -> dict[str, str]:
    try:
        with db.connection() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok", "database": "connected"}
    except DatabaseUnavailable:
        return {"status": "degraded", "database": "unavailable"}


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = run_assistant(
        request.question,
        [message.model_dump() for message in request.conversation],
        request.preferred_language,
    )
    citation_ids = set(result.get("citations", []))
    citations = [
        item for item in result.get("evidence", [])
        if item.chunk_id in citation_ids or f"{item.document_id}:{item.chunk_id}" in citation_ids
    ]
    return AskResponse(
        answer=result["answer"], language=result.get("language", "en"),
        citations=citations, needs_human=result.get("needs_human", False),
    )


@app.post("/api/escalations", status_code=status.HTTP_201_CREATED)
def create_escalation(request: EscalationRequest) -> dict[str, str]:
    try:
        reference = db.submit_enquiry(
            category=request.category, department=request.department,
            language=request.language, question=request.question, reason=request.reason,
            consent=request.consent, name=request.name, email=str(request.email or ""),
            phone=request.phone,
        )
        return {"reference": reference, "status": "open"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail="Enquiry could not be saved. Please try again later.") from exc


@app.get("/api/admin/summary")
def admin_summary(actor: Annotated[str, Depends(authenticate_admin)]) -> dict:
    try:
        return db.dashboard_metrics()
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail="The Ministry database is unavailable.") from exc


@app.get("/api/admin/documents")
def admin_documents(
    actor: Annotated[str, Depends(authenticate_admin)],
    approval_status: Annotated[str | None, Query(alias="status")] = None,
) -> list[dict]:
    allowed = {None, "pending", "approved", "rejected", "withdrawn"}
    if approval_status not in allowed:
        raise HTTPException(status_code=400, detail="Invalid document status.")
    try:
        return db.list_documents(status=approval_status)
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail="The Ministry database is unavailable.") from exc


@app.post("/api/admin/documents", status_code=status.HTTP_201_CREATED)
def upload_document(
    actor: Annotated[str, Depends(authenticate_admin)],
    upload: UploadFile = File(...),
    metadata: str = Form(..., description="DocumentMetadata as a JSON object"),
) -> dict[str, str]:
    try:
        document_metadata = DocumentMetadata.model_validate_json(metadata)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_context=False)) from exc

    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    content = upload.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="The document exceeds the upload size limit.")
    try:
        document_id = ingest_document(
            upload.filename or "",
            content,
            document_metadata,
            actor=actor,
        )
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail="The document could not be saved.") from exc
    return {
        "document_id": document_id,
        "approval_status": "approved",
        "ingestion_status": "ready",
    }


@app.post("/api/admin/documents/{document_id}/approve")
def approve_document(
    document_id: str,
    actor: Annotated[str, Depends(authenticate_admin)],
    replacement_id: str | None = None,
) -> dict[str, str]:
    try:
        db.approve_document(document_id, actor, replacement_id)
        return {"document_id": document_id, "approval_status": "approved"}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail="The document could not be approved.") from exc


@app.post("/api/admin/documents/{document_id}/reject")
def reject_document(
    document_id: str, actor: Annotated[str, Depends(authenticate_admin)]
) -> dict[str, str]:
    try:
        db.reject_document(document_id, actor)
        return {"document_id": document_id, "approval_status": "rejected"}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail="The document could not be rejected.") from exc


@app.get("/api/admin/enquiries")
def admin_enquiries(
    actor: Annotated[str, Depends(authenticate_staff)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[dict]:
    try:
        return db.list_enquiries(limit=limit)
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail="The Ministry database is unavailable.") from exc


@app.post("/api/admin/enquiries/{reference}/status")
def update_enquiry(
    reference: str,
    update: EnquiryStatusUpdate,
    actor: Annotated[str, Depends(authenticate_staff)],
) -> dict[str, str]:
    try:
        db.update_enquiry_status(reference, update.status, actor)
        return {"reference": reference, "status": update.status}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail="The enquiry status could not be updated.") from exc


@app.get("/api/admin/users")
def admin_users(actor: Annotated[str, Depends(authenticate_admin)]) -> list[dict]:
    try:
        return db.list_users()
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail="The account database is unavailable.") from exc


@app.post("/api/admin/users", status_code=status.HTTP_201_CREATED)
def create_admin_user(
    request: AdminUserCreate,
    actor: Annotated[str, Depends(authenticate_admin)],
) -> dict[str, str]:
    try:
        db.create_user(
            request.username, hash_admin_password(request.password.get_secret_value()), request.role, actor
        )
        return {"username": request.username, "role": request.role, "status": "active"}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail="The account could not be created.") from exc


@app.patch("/api/admin/users/{username}")
def set_account_active(
    username: str,
    request: UserActivation,
    actor: Annotated[str, Depends(authenticate_admin)],
) -> dict[str, str | bool]:
    if username == actor and not request.active:
        raise HTTPException(status_code=409, detail="You cannot deactivate your current account.")
    try:
        db.set_user_active(username, request.active, actor)
        return {"username": username, "active": request.active}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail="The account status could not be updated.") from exc
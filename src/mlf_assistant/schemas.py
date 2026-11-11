from datetime import date
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, SecretStr, model_validator


class Evidence(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    department: str = ""
    category: str = ""
    language: str = ""
    publication_date: str = ""
    effective_date: str = ""
    version: str = ""
    approval_status: Literal["pending", "approved", "rejected", "withdrawn"] = "pending"
    review_date: str = ""
    superseded: bool = False
    score: float = 0.0


class AssistantAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation: list["ConversationMessage"] = Field(default_factory=list, max_length=20)
    preferred_language: Literal["en", "sw", "auto"] = "auto"


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskResponse(BaseModel):
    answer: str
    language: Literal["en", "sw"]
    citations: list[Evidence] = Field(default_factory=list)
    needs_human: bool = False
    reference: str | None = None


class DocumentMetadata(BaseModel):
    title: str = Field(min_length=2, max_length=300)
    department: str = Field(min_length=2, max_length=120)
    category: str = Field(min_length=2, max_length=120)
    language: Literal["en", "sw", "mixed"]
    issuing_authority: str = Field(min_length=2, max_length=200)
    publication_date: date | None = None
    effective_date: date | None = None
    version: str = Field(min_length=1, max_length=60)
    review_date: date | None = None


class EscalationRequest(BaseModel):
    category: str = Field(min_length=2, max_length=120)
    department: str = Field(min_length=2, max_length=120)
    language: Literal["en", "sw"]
    question: str = Field(min_length=1, max_length=4000)
    reason: str = Field(min_length=2, max_length=1000)
    consent: bool
    name: str = Field(default="", max_length=120)
    email: EmailStr | None = None
    phone: str = Field(default="", max_length=40)

    @model_validator(mode="after")
    def require_return_contact_with_consent(self):
        if self.consent and not (self.email or self.phone.strip()):
            raise ValueError("An email address or phone number is required for a follow-up.")
        return self


class AdminUserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")
    password: SecretStr = Field(min_length=14, max_length=256)
    role: Literal["admin", "officer"] = "officer"


class UserActivation(BaseModel):
    active: bool
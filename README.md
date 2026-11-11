# Tanzania Ministry of Livestock and Fisheries AI Assistant

A bilingual, evidence-gated information service for the United Republic of Tanzania's Ministry of Livestock and Fisheries. Public answers are generated only from current, successfully indexed documents. When sources are missing, stale, irrelevant, or not citable, the assistant abstains and can offer a consent-based enquiry escalation.

## Run locally

1. Copy `.env.example` to `.env`. Set `MLF_DB_PASSWORD` (for example, `openssl rand -hex 32`), database URL, administrator username, and provider settings. Generate an administrator password hash with `python -m mlf_assistant.admin` and set the quoted value as `MLF_ADMIN_PASSWORD_HASH`; do not store a plain password in `.env`.
2. Install the project: `pip install -e '.[ai,documents,exports,observability]'`. The `ai` extra includes model clients and sentence-transformers; text-based PDFs use pypdf, while Docling handles scanned PDFs and other complex formats.
3. Start pgvector with `docker compose up -d db` (or use the full stack with `docker compose up --build`).
4. Initialize the schema with `mlf-assistant-init-db`.
5. Start the interface with `streamlit run app.py`; start the API separately with `uvicorn mlf_assistant.api:app --host 0.0.0.0 --port 8000`.

For local development without model credentials or an initialized database, the interface still opens. Greetings work locally; substantive questions abstain and no Ministry facts are fabricated. Document ingestion, retrieval, enquiries, and administrative reporting require PostgreSQL and their configured model/document dependencies.

In the public chat, use **Record a question** to ask by voice and enable **Include an audio reply** to generate a playable spoken answer. The recorder uses Streamlit's browser media capture rather than browser-specific speech recognition, and requires Streamlit 1.41 or newer. Transcription and audio replies require the OpenAI provider, `MLF_LLM_API_KEY`, and access to the configured speech models (`MLF_SPEECH_TRANSCRIPTION_MODEL` and `MLF_SPEECH_SYNTHESIS_MODEL`). Recordings are sent to the configured OpenAI-compatible API for transcription and may incur charges; generated audio is returned to the browser as an MP3. Use HTTPS (or localhost) for microphone access, and do not speak confidential or sensitive personal information. Voice processing is unavailable with the Ollama provider; typed chat remains available.

## Configuration

`MLF_LLM_PROVIDER` supports `openai` and `ollama`; `MLF_LLM_MODEL` selects the model and `MLF_LLM_BASE_URL` optionally selects an OpenAI-compatible or Ollama endpoint. The multilingual embedding default is `BAAI/bge-m3`; the reranker is optional and configured with `MLF_RERANKER_MODEL`. Model timeouts, retries, relevance threshold, upload limit, and optional per-token cost rates are controlled through environment variables. Secrets are not displayed in the dashboard.

Generate the initial administrator hash interactively so the password is not echoed:

```sh
python -m mlf_assistant.admin
```

Set `MLF_ADMIN_USERNAME` and the generated `MLF_ADMIN_PASSWORD_HASH` before initializing the database. Additional admin and officer accounts can then be created in the protected dashboard. Dashboard sessions expire after 30 minutes and are rechecked against the current account role and activation status. Administrative API endpoints use HTTP Basic and must be served behind HTTPS. Configure `MLF_ALLOWED_ORIGINS` with the exact official website origins before browser integration; CORS is disabled when the list is empty.

## Architecture

- `app.py`: Streamlit bilingual public chat, browser voice controls, citations, consent-based escalation, and restricted operations dashboard.
- `mlf_assistant.api`: FastAPI ask/escalation APIs and server-authorized admin routes.
- `mlf_assistant.workflow`: bounded LangGraph planning, language/intent detection, follow-up rewriting, query decomposition, evidence relevance grading, grounded generation, and controlled abstention.
- `mlf_assistant.retrieval`: PostgreSQL full-text and pgvector candidate retrieval, reciprocal-rank fusion, metadata/freshness filtering, and optional multilingual cross-encoder reranking.
- `mlf_assistant.documents`: type/size checks, SHA-256 duplicate detection, Docling extraction for PDF/Office/images, table-friendly formats, chunking, BGE-M3 indexing, and re-indexing.
- `mlf_assistant.db`: PostgreSQL document versions, approval state, enquiries, role accounts, audit events, and aggregate usage metrics.
- `mlf_assistant.grounding`: current approval/effective/review checks and exact citation validation, independent of model output.
- `mlf_assistant.observability`: opt-in Langfuse callbacks and OpenTelemetry spans.

The assistant has no write-capable Ministry tools. Uploaded documents become available to retrieval automatically after successful processing. Re-uploading the same document can refresh a failed ingestion; older versions can be superseded. Only the API or UI operation that successfully commits an escalation returns its reference number. Contact information is collected only with consent. Usage telemetry stores aggregate attributes and token counts, not chat transcripts.

## Document-to-answer flow

Administrators can upload through the dashboard by selecting a file and metadata, then choosing **Process document**, or use `POST /api/admin/documents` with HTTP Basic authentication and multipart form fields `upload` and `metadata` (a JSON `DocumentMetadata` object). The service extracts and chunks the file, creates BGE-M3 embeddings, and stores the text and vectors in PostgreSQL/pgvector. Text-based PDFs use pypdf; scanned PDFs fall back to Docling OCR. After processing succeeds, the dashboard confirms that the document is indexed and available to users immediately—there is no approval step. Retrieval combines vector and full-text search over successfully processed, current documents. It searches both planned and original question wording, includes recent context for follow-up questions, and continues with other query variants if one retrieval attempt fails. Answers use the format appropriate to the request—such as prose, lists, numbered steps, tables, or calculations—while preserving source values and including validated citations. The assistant still abstains if available documents do not support an answer; no system can reliably answer questions that are missing from its indexed sources. Failed uploads can be retried by uploading the same file again after fixing the processing issue.

## Verification and operational limits

Run custom tests with `python -m unittest discover -s tests -v` or install the evaluation extra and use `pytest`. RAGAS dependencies are provided for evaluation workflows, but meaningful bilingual RAGAS scores require a Ministry-approved benchmark corpus and configured model. This repository does not include official Ministry documents, credentials, website origins, a production identity provider, or government service integrations; those must be supplied and approved by the Ministry. Review retention, encryption at rest/in transit, access policies, model data handling, and hosting jurisdiction before production use.

To run the PostgreSQL end-to-end integration test, set `MLF_TEST_DATABASE_URL` to a test database URL and run `pytest -q tests/test_postgres_integration.py`. The test creates and drops its own temporary schema, uses real PostgreSQL/pgvector persistence and retrieval, and stubs only embedding/model inference. Use a disposable test database account with permission to create schemas; never point this test at production.

from typing import TypedDict, Any, Optional
from pathlib import Path
import time

from langgraph.graph import StateGraph, END

from core import (
    detect_document_type,
    extract_structured_json,
    build_resume,
    json_to_kv_dataframe,
    generate_excel,
    get_current_metrics_snapshot,
    diff_metrics_snapshot,
    validate_document_data,
    build_confidence_map,
)

class IDPState(TypedDict, total=False):
    text: str
    template: Optional[bytes]
    filename: str
    progress: Any

    doc_type: str
    data: dict
    result: dict
    error: str
    step_metrics: list
    confidence: dict
    validation: dict
    ocr_used: bool
    extraction_mode: str
    exception_reason: str
    needs_review: bool


def safe_progress(state, percent, message):
    progress = state.get("progress")
    if progress:
        progress(percent, message)


def add_step_metric(state, step, started_at, before_metrics, message=""):
    after_metrics = get_current_metrics_snapshot()
    delta = diff_metrics_snapshot(before_metrics, after_metrics)
    if "step_metrics" not in state or state["step_metrics"] is None:
        state["step_metrics"] = []
    state["step_metrics"].append({
        "step": step,
        "message": message,
        "duration": time.time() - started_at,
        "metrics": delta
    })


def get_resume_filename_from_data(data: dict) -> str:
    if not isinstance(data, dict):
        return "candidate.docx"

    name = (
        data.get("name")
        or data.get("Name")
        or data.get("candidate_name")
        or (
            data.get("personal_details", {}).get("name")
            if isinstance(data.get("personal_details"), dict)
            else None
        )
        or "candidate"
    )

    import re
    safe_name = re.sub(r'[\\/*?:"<>|]', "", str(name)).strip()
    safe_name = safe_name if safe_name else "candidate"
    return f"{safe_name}.docx"


def detect_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 35, "Detecting document type")

    text = (state.get("text") or "").strip()
    if not text:
        state["error"] = "No extracted text available for processing"
        state["doc_type"] = "other"
        add_step_metric(state, "Detect document type", started_at, before, "No text found")
        return state

    try:
        state["doc_type"] = detect_document_type(text)
        add_step_metric(state, "Detect document type", started_at, before, f"Detected {state['doc_type']}")
    except Exception as e:
        state["error"] = f"Document type detection failed: {str(e)}"
        state["doc_type"] = "other"
        add_step_metric(state, "Detect document type", started_at, before, str(e))

    return state


def resume_extract_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 50, "Extracting resume details")

    try:
        state["data"] = extract_structured_json(state["text"], "resume")
        add_step_metric(state, "Extract resume data", started_at, before, "Resume fields extracted")
    except Exception as e:
        state["error"] = f"Resume extraction failed: {str(e)}"
        state["data"] = {}
        add_step_metric(state, "Extract resume data", started_at, before, str(e))

    state["validation"] = validate_document_data(state.get("data") or {}, "resume")
    state["confidence"] = build_confidence_map(state.get("data") or {}, "resume")
    state["needs_review"] = not state["validation"].get("passed", True)
    return state


def resume_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 75, "Building resume")

    data = state.get("data") or {}
    template_bytes = state.get("template")

    if not template_bytes:
        possible_paths = [
            Path("templates/resume_template.docx"),
            Path("templates:resume_template.docx"),
            Path(__file__).parent / "templates" / "resume_template.docx",
            Path(__file__).parent / "templates:resume_template.docx",
        ]
        for template_path in possible_paths:
            if template_path.exists():
                with open(template_path, "rb") as f:
                    template_bytes = f.read()
                break

    if not template_bytes:
        state["error"] = "No resume template provided and default template not found"
        state["result"] = {
            "type": "resume",
            "file": None,
            "data": data,
            "file_name": "candidate.docx",
            "message": "Resume template missing"
        }
        add_step_metric(state, "Build resume", started_at, before, "Template missing")
        return state

    try:
        file_bytes = build_resume(data, template_bytes)
        file_name = get_resume_filename_from_data(data)

        safe_progress(state, 95, "Resume ready")
        state["result"] = {
            "type": "resume",
            "file": file_bytes,
            "data": data,
            "file_name": file_name,
            "message": "Resume generated successfully"
        }
        add_step_metric(state, "Build resume", started_at, before, "Resume file created")
    except Exception as e:
        state["error"] = f"Resume generation failed: {str(e)}"
        state["result"] = {
            "type": "resume",
            "file": None,
            "data": data,
            "file_name": "candidate.docx",
            "message": str(e)
        }
        add_step_metric(state, "Build resume", started_at, before, str(e))

    return state


def extract_json_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 50, "Extracting structured fields")

    doc_type = state.get("doc_type", "other")
    try:
        state["data"] = extract_structured_json(state["text"], doc_type)
        add_step_metric(state, f"Extract {doc_type} data", started_at, before, "Structured fields extracted")
    except Exception as e:
        state["error"] = f"Structured extraction failed: {str(e)}"
        state["data"] = {}
        add_step_metric(state, f"Extract {doc_type} data", started_at, before, str(e))

    state["validation"] = validate_document_data(state.get("data") or {}, doc_type)
    state["confidence"] = build_confidence_map(state.get("data") or {}, doc_type)
    state["needs_review"] = not state["validation"].get("passed", True)
    return state


def invoice_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 75, "Preparing invoice output")

    data = state.get("data") or {}
    try:
        df = json_to_kv_dataframe(data)
        excel = generate_excel(df)

        safe_progress(state, 95, "Invoice ready for review")
        state["result"] = {
            "type": "invoice",
            "table": df,
            "excel": excel,
            "data": data,
            "concur_status": None,
            "concur_mode": None,
            "message": "Invoice extracted. Review and approve to send to Concur."
        }
        add_step_metric(state, "Create invoice output", started_at, before, "Invoice ready")
    except Exception as e:
        state["error"] = f"Invoice output generation failed: {str(e)}"
        state["result"] = {
            "type": "invoice",
            "table": None,
            "excel": None,
            "data": data,
            "message": str(e)
        }
        add_step_metric(state, "Create invoice output", started_at, before, str(e))

    return state


def ticket_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 75, "Preparing ticket output")

    data = state.get("data") or {}
    try:
        safe_progress(state, 95, "Ticket ready for review")
        state["result"] = {
            "type": "ticket",
            "status": "ready",
            "data": data,
            "concur_status": None,
            "concur_mode": None,
            "message": "Ticket extracted. Review and approve to send to Concur."
        }
        add_step_metric(state, "Create ticket output", started_at, before, "Ticket ready")
    except Exception as e:
        state["error"] = f"Ticket processing failed: {str(e)}"
        state["result"] = {
            "type": "ticket",
            "status": "error",
            "data": data,
            "message": str(e)
        }
        add_step_metric(state, "Create ticket output", started_at, before, str(e))

    return state


def other_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 85, "Finalizing output")

    state["data"] = {}
    state["result"] = {
        "type": state.get("doc_type", "other"),
        "message": f"No structured output configured for document type: {state.get('doc_type', 'other')}"
    }
    add_step_metric(state, "Finalize generic output", started_at, before, "No structured processing needed")
    return state


def route_after_detect(state: IDPState) -> str:
    dt = state.get("doc_type", "other")
    if dt == "resume":
        return "resume_extract"
    elif dt in ["invoice", "ticket"]:
        return "extract_json"
    else:
        return "other"


def route_after_extract_json(state: IDPState) -> str:
    dt = state.get("doc_type", "other")
    if dt == "invoice":
        return "invoice"
    elif dt == "ticket":
        return "ticket"
    else:
        return "other"


def build_graph():
    builder = StateGraph(IDPState)

    builder.add_node("detect", detect_node)
    builder.add_node("resume_extract", resume_extract_node)
    builder.add_node("resume", resume_node)
    builder.add_node("extract_json", extract_json_node)
    builder.add_node("invoice", invoice_node)
    builder.add_node("ticket", ticket_node)
    builder.add_node("other", other_node)

    builder.set_entry_point("detect")

    builder.add_conditional_edges(
        "detect",
        route_after_detect,
        {
            "resume_extract": "resume_extract",
            "extract_json": "extract_json",
            "other": "other",
        }
    )

    builder.add_edge("resume_extract", "resume")

    builder.add_conditional_edges(
        "extract_json",
        route_after_extract_json,
        {
            "invoice": "invoice",
            "ticket": "ticket",
            "other": "other",
        }
    )

    builder.add_edge("resume", END)
    builder.add_edge("invoice", END)
    builder.add_edge("ticket", END)
    builder.add_edge("other", END)

    return builder.compile()

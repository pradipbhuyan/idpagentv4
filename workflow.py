from typing import TypedDict
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
    send_to_concur,
)


class IDPState(TypedDict, total=False):
    text: str
    filename: str
    template: bytes
    progress: any
    event_callback: any
    ocr_used: bool
    extraction_mode: str
    exception_reason: str

    doc_type: str
    data: dict
    result: dict
    error: str
    step_metrics: list
    validation: dict
    confidence: dict


def safe_progress(state: IDPState, percent: int, message: str):
    progress = state.get("progress")
    if progress:
        try:
            progress(percent, message)
        except Exception:
            pass


def emit_agent_event(state: IDPState, agent: str, status: str, message: str):
    callback = state.get("event_callback")
    if callback:
        try:
            callback(agent, status, message)
        except Exception:
            pass


def add_step_metric(state: IDPState, step_name: str, started_at: float, before: dict, note: str = ""):
    after = get_current_metrics_snapshot()
    diff = diff_metrics_snapshot(before, after)

    if "step_metrics" not in state or state["step_metrics"] is None:
        state["step_metrics"] = []

    state["step_metrics"].append({
        "step": step_name,
        "duration_sec": round(time.time() - started_at, 2),
        "tokens": diff.get("tokens", 0),
        "input_tokens": diff.get("input_tokens", 0),
        "output_tokens": diff.get("output_tokens", 0),
        "cost": round(diff.get("cost", 0.0), 6),
        "calls": diff.get("calls", 0),
        "note": note,
    })


def detect_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()

    emit_agent_event(state, "Classification Agent", "running", "Classifying document")
    safe_progress(state, 40, "Classification Agent — classifying document")

    state["doc_type"] = detect_document_type(state.get("text", ""))

    emit_agent_event(
        state,
        "Classification Agent",
        "done",
        f"Document identified as {state.get('doc_type', 'other')}"
    )

    add_step_metric(state, "Detect document type", started_at, before, state.get("doc_type", "unknown"))
    return state


def extract_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()

    doc_type = state.get("doc_type", "other")

    if doc_type == "resume":
        emit_agent_event(state, "Structuring Agent", "running", "Extracting candidate profile")
        safe_progress(state, 55, "Structuring Agent — extracting candidate profile")
    elif doc_type == "invoice":
        emit_agent_event(state, "Structuring Agent", "running", "Extracting invoice fields")
        safe_progress(state, 55, "Structuring Agent — extracting invoice fields")
    elif doc_type == "ticket":
        emit_agent_event(state, "Structuring Agent", "running", "Extracting travel fields")
        safe_progress(state, 55, "Structuring Agent — extracting travel fields")

    if doc_type in ["invoice", "ticket", "resume"]:
        state["data"] = extract_structured_json(state.get("text", ""), doc_type)

        if doc_type == "resume":
            emit_agent_event(state, "Structuring Agent", "done", "Candidate profile extracted")
        elif doc_type == "invoice":
            emit_agent_event(state, "Structuring Agent", "done", "Invoice fields extracted")
        elif doc_type == "ticket":
            emit_agent_event(state, "Structuring Agent", "done", "Travel fields extracted")

        add_step_metric(state, "Extract structured data", started_at, before, doc_type)
    else:
        state["data"] = {}
        add_step_metric(state, "Skip structured extraction", started_at, before, doc_type)

    return state


def resume_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()

    emit_agent_event(state, "Output Agent", "running", "Generating resume in template")
    safe_progress(state, 75, "Output Agent — generating resume in template")

    data = state.get("data") or {}
    template_bytes = state.get("template")

    file_bytes = build_resume(data, template_bytes)

    emit_agent_event(state, "Output Agent", "done", "Resume generated")
    safe_progress(state, 95, "Output Agent — resume ready")

    candidate_name = (data.get("name") or "candidate").strip() if isinstance(data, dict) else "candidate"
    safe_name = "".join(ch for ch in candidate_name if ch not in '\\/*?:"<>|').strip() or "candidate"

    state["result"] = {
        "type": "resume",
        "file": file_bytes,
        "file_name": f"{safe_name}.docx",
        "data": data,
    }

    add_step_metric(state, "Build resume", started_at, before, state["result"]["file_name"])
    return state


def invoice_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()

    data = state.get("data") or {}

    try:
        emit_agent_event(state, "Output Agent", "running", "Creating invoice Excel")
        safe_progress(state, 70, "Output Agent — creating invoice Excel")

        df = json_to_kv_dataframe(data)
        excel = generate_excel(df)

        emit_agent_event(state, "Output Agent", "done", "Invoice Excel created")

        emit_agent_event(state, "Concur Agent", "running", "Submitting invoice to Concur")
        safe_progress(state, 88, "Concur Agent — submitting invoice to Concur")

        concur_result = send_to_concur("invoice", data, mode="mock")

        emit_agent_event(state, "Concur Agent", "done", "Invoice submitted to Concur")
        safe_progress(state, 95, "Concur Agent — invoice submitted")

        state["result"] = {
            "type": "invoice",
            "table": df,
            "excel": excel,
            "data": data,
            "payload": concur_result.get("payload"),
            "concur_status": concur_result.get("status"),
            "concur_mode": concur_result.get("mode"),
            "concur_submission_id": concur_result.get("submission_id"),
            "concur_batch_id": concur_result.get("batch_id"),
            "concur_document_id": concur_result.get("document_id"),
            "concur_submitted_at": concur_result.get("submitted_at"),
            "concur_endpoint": concur_result.get("endpoint"),
            "concur_processing_state": concur_result.get("processing_state"),
            "concur_next_status": concur_result.get("next_status"),
            "message": concur_result.get("message", "Invoice processed successfully")
        }

        add_step_metric(state, "Create invoice output + send to Concur", started_at, before, "Invoice submitted")
    except Exception as e:
        emit_agent_event(state, "Concur Agent", "error", str(e))
        state["error"] = f"Invoice processing failed: {str(e)}"
        state["result"] = {
            "type": "invoice",
            "table": None,
            "excel": None,
            "data": data,
            "concur_status": "error",
            "message": str(e)
        }
        add_step_metric(state, "Create invoice output + send to Concur", started_at, before, str(e))

    return state


def ticket_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()

    data = state.get("data") or {}

    try:
        emit_agent_event(state, "Output Agent", "running", "Preparing ticket payload")
        safe_progress(state, 70, "Output Agent — preparing ticket payload")

        emit_agent_event(state, "Output Agent", "done", "Ticket payload ready")

        emit_agent_event(state, "Concur Agent", "running", "Submitting ticket to Concur")
        safe_progress(state, 88, "Concur Agent — submitting ticket to Concur")

        concur_result = send_to_concur("ticket", data, mode="mock")

        emit_agent_event(state, "Concur Agent", "done", "Ticket submitted to Concur")
        safe_progress(state, 95, "Concur Agent — ticket submitted")

        state["result"] = {
            "type": "ticket",
            "status": "sent",
            "data": data,
            "payload": concur_result.get("payload"),
            "concur_status": concur_result.get("status"),
            "concur_mode": concur_result.get("mode"),
            "concur_submission_id": concur_result.get("submission_id"),
            "concur_batch_id": concur_result.get("batch_id"),
            "concur_document_id": concur_result.get("document_id"),
            "concur_submitted_at": concur_result.get("submitted_at"),
            "concur_endpoint": concur_result.get("endpoint"),
            "concur_processing_state": concur_result.get("processing_state"),
            "concur_next_status": concur_result.get("next_status"),
            "message": concur_result.get("message", "Ticket processed successfully")
        }

        add_step_metric(state, "Create ticket output + send to Concur", started_at, before, "Ticket submitted")
    except Exception as e:
        emit_agent_event(state, "Concur Agent", "error", str(e))
        state["error"] = f"Ticket processing failed: {str(e)}"
        state["result"] = {
            "type": "ticket",
            "status": "error",
            "data": data,
            "concur_status": "error",
            "message": str(e)
        }
        add_step_metric(state, "Create ticket output + send to Concur", started_at, before, str(e))

    return state


def other_node(state: IDPState) -> IDPState:
    state["result"] = {
        "type": "other",
        "data": {},
    }
    return state


def route(state: IDPState):
    dt = state.get("doc_type", "other")

    if dt == "resume":
        return "resume"
    if dt == "invoice":
        return "invoice"
    if dt == "ticket":
        return "ticket"
    return "other"


def build_graph():
    builder = StateGraph(IDPState)

    builder.add_node("detect", detect_node)
    builder.add_node("extract", extract_node)
    builder.add_node("resume", resume_node)
    builder.add_node("invoice", invoice_node)
    builder.add_node("ticket", ticket_node)
    builder.add_node("other", other_node)

    builder.set_entry_point("detect")
    builder.add_edge("detect", "extract")

    builder.add_conditional_edges(
        "extract",
        route,
        {
            "resume": "resume",
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

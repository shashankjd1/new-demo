from fastapi import APIRouter, HTTPException

from models.schemas import AnalysisResponse, AnalyzeRequest, ChatRequest, ChatResponse
from services.ai_service import analyze_product, generate_chat_response


router = APIRouter()


def _extract_product_name(query: str) -> str:
    stripped = query.strip()
    if '"' in stripped:
        parts = [part.strip() for part in stripped.split('"') if part.strip()]
        if parts:
            return parts[0][:80]

    lowered = stripped.lower()
    for marker in ("about ", "for ", "is ", "of "):
        if marker in lowered:
            start = lowered.index(marker) + len(marker)
            candidate = stripped[start:].strip(" ?.!,:;")
            if candidate:
                return candidate[:80]

    return stripped[:80] or "Product from chat query"


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalyzeRequest) -> AnalysisResponse:
    try:
        return analyze_product(request)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to analyze product.") from exc


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        context = request.context
        if context is None:
            analyze_request = AnalyzeRequest(
                product_name=_extract_product_name(request.query),
                marketing_text=request.query,
                specs="",
            )
            context = analyze_product(analyze_request)

        answer = generate_chat_response(request.query, context)
        return ChatResponse(answer=answer)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to answer chat query.") from exc

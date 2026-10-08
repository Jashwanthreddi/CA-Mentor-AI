from fastapi import APIRouter, HTTPException

from app.core import logger
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService


router = APIRouter()

chat_service = ChatService()


@router.post(
    "/chat",
    summary="Ask CA Mentor AI",
    description="Retrieve relevant CA study material and generate an evidence-grounded answer.",
)
async def chat(request: ChatRequest):
    """
    Main CA Mentor AI chat endpoint.

    Flow:
        Question
            ↓
        Hybrid Retrieval
            ↓
        Reranking
            ↓
        Gemini Generation
            ↓
        Answer + Sources
    """

    logger.info("Chat request received")

    try:
        result = chat_service.chat(
            question=request.question,
        )

        return {
            "success": True,
            "message": "Answer generated successfully",
            "data": result,
        }

    except ValueError as exc:
        logger.warning("Chat request validation failed: %s", exc)

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Chat request failed")

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer at this time.",
        ) from exc
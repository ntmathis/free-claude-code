"""FastAPI route handlers."""

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from providers.base import BaseProvider
from providers.common import is_failover_retryable
from providers.exceptions import InvalidRequestError, ProviderError
from providers.logging_utils import build_request_summary, log_request_compact

from .dependencies import get_provider, get_provider_for_type, get_settings
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import TokenCountResponse
from .optimization_handlers import try_optimizations
from .request_utils import get_token_count

router = APIRouter()


def _build_sse_error_event(exc: Exception) -> str:
    if isinstance(exc, ProviderError):
        payload = exc.to_anthropic_format()
    else:
        payload = {
            "type": "error",
            "error": {"type": "api_error", "message": str(exc)},
        }
    return f"event: error\ndata: {json.dumps(payload)}\n\n"


# =============================================================================
# Routes
# =============================================================================


@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    raw_request: Request,
    provider: BaseProvider = Depends(get_provider),
    settings: Settings = Depends(get_settings),
):
    """Create a message (always streaming)."""

    try:
        if not request_data.messages:
            raise InvalidRequestError("messages cannot be empty")

        optimized = try_optimizations(request_data, settings)
        if optimized is not None:
            return optimized

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        log_request_compact(logger, request_id, request_data)
        input_tokens = get_token_count(
            request_data.messages, request_data.system, request_data.tools
        )
        candidates = request_data.target_candidates or [
            {
                "provider_type": request_data.target_provider_type
                or settings.provider_type,
                "provider_model": request_data.model,
                "mapped_model": (
                    f"{request_data.target_provider_type or settings.provider_type}/"
                    f"{request_data.model}"
                ),
            }
        ]

        last_exc: Exception | None = None
        for idx, candidate in enumerate(candidates):
            provider_type = candidate["provider_type"]
            provider_model = candidate["provider_model"]
            is_last = idx == len(candidates) - 1

            route_provider = provider
            if provider_type != settings.provider_type:
                route_provider = get_provider_for_type(provider_type)

            candidate_request = request_data.model_copy(deep=True)
            candidate_request.model = provider_model
            candidate_request.target_provider_type = provider_type

            stream = route_provider.stream_response(
                candidate_request,
                input_tokens=input_tokens,
                request_id=request_id,
                raise_errors=True,
            )

            try:
                first_event = await stream.__anext__()
            except StopAsyncIteration:
                continue
            except Exception as exc:
                last_exc = exc
                if not is_last and is_failover_retryable(exc):
                    logger.warning(
                        "MODEL_FAILOVER: request_id=%s attempt=%d/%d failed provider=%s model=%s error=%s",
                        request_id,
                        idx + 1,
                        len(candidates),
                        provider_type,
                        provider_model,
                        type(exc).__name__,
                    )
                    continue
                raise

            async def stream_with_first(
                first: str = first_event,
                source: AsyncIterator[str] = stream,
                active_provider_type: str = provider_type,
                active_provider_model: str = provider_model,
            ) -> AsyncIterator[str]:
                emitted_content = "event: content_block_delta" in first
                yield first
                try:
                    async for event in source:
                        if "event: content_block_delta" in event:
                            emitted_content = True
                        yield event
                except Exception as exc:
                    logger.error(
                        "STREAM_ERROR_AFTER_START: request_id=%s provider=%s model=%s error=%s",
                        request_id,
                        active_provider_type,
                        active_provider_model,
                        type(exc).__name__,
                    )
                    if not emitted_content:
                        yield _build_sse_error_event(exc)

            if idx > 0:
                logger.info(
                    "MODEL_FAILOVER: request_id=%s using fallback attempt=%d/%d provider=%s model=%s",
                    request_id,
                    idx + 1,
                    len(candidates),
                    provider_type,
                    provider_model,
                )

            return StreamingResponse(
                stream_with_first(),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        if last_exc is not None:
            raise last_exc
        raise HTTPException(status_code=500, detail="No model candidates available.")

    except ProviderError:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Error: {e!s}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=getattr(e, "status_code", 500), detail=str(e)
        ) from e


@router.post("/v1/messages/count_tokens")
async def count_tokens(request_data: TokenCountRequest):
    """Count tokens for a request."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    with logger.contextualize(request_id=request_id):
        try:
            tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
            summary = build_request_summary(request_data)
            summary["request_id"] = request_id
            summary["input_tokens"] = tokens
            logger.info("COUNT_TOKENS: %s", json.dumps(summary))
            return TokenCountResponse(input_tokens=tokens)
        except Exception as e:
            import traceback

            logger.error(
                "COUNT_TOKENS_ERROR: request_id=%s error=%s\n%s",
                request_id,
                str(e),
                traceback.format_exc(),
            )
            raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/")
async def root(settings: Settings = Depends(get_settings)):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": settings.provider_type,
        "model": settings.model,
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post("/stop")
async def stop_cli(request: Request):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()
    logger.info("STOP_CLI: source=handler cancelled_count=%d", count)
    return {"status": "stopped", "cancelled_count": count}

import logging
from time import perf_counter

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from pydantic import BaseModel

from graphtool.run_logging import LOGGER_NAME

RUN_LOGGER = logging.getLogger(LOGGER_NAME)


def invoke_model(model, messages: list, *, stage: str):
    RUN_LOGGER.info(
        "Starting %s: prompt approximately %d tokens",
        stage,
        count_tokens_approximately(messages),
    )
    started_at = perf_counter()
    try:
        return model.invoke(messages), perf_counter() - started_at
    except Exception as exc:
        duration = perf_counter() - started_at
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            RUN_LOGGER.error(
                "%s failed after %.2fs: %s",
                stage.capitalize(),
                duration,
                type(exc).__name__,
            )
        else:
            RUN_LOGGER.error(
                "%s failed after %.2fs: %s (status=%s)",
                stage.capitalize(),
                duration,
                type(exc).__name__,
                status_code,
            )
        raise


def validated_output(model_type: type[BaseModel], value):
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(value)


def message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)

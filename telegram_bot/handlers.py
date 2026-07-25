import asyncio
import logging
from contextlib import suppress
from time import perf_counter

from telegram import Chat, Message, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from graphtool.agents import AgentResponse, KnowledgeAgent
from graphtool.retrieval import format_source_reference
from graphtool.run_logging import LOGGER_NAME

MAX_TELEGRAM_MESSAGE_LENGTH = 4_096
TYPING_REFRESH_INTERVAL_SECONDS = 4
START_MESSAGE = (
    "Send me a question or request. I can search the GraphTool knowledge base "
    "when needed and will remember this conversation until the bot restarts. "
    "Send /new to start over."
)
UNAUTHORIZED_MESSAGE = "You are not authorized to use this bot."
NEW_CONVERSATION_MESSAGE = "Started a new conversation."
ERROR_MESSAGE = "I couldn't answer that question. Please try again."


class TelegramHandlers:
    def __init__(
        self,
        agent: KnowledgeAgent,
        allowed_user_ids: frozenset[int],
    ) -> None:
        self._agent = agent
        self._allowed_user_ids = allowed_user_ids
        self._agent_lock = asyncio.Lock()
        self._logger = logging.getLogger(LOGGER_NAME)

    async def start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        del context
        message = update.effective_message
        if message is None:
            return
        if not self._is_authorized(update):
            await message.reply_text(UNAUTHORIZED_MESSAGE)
            return
        await message.reply_text(START_MESSAGE)

    async def new_conversation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        del context
        message = update.effective_message
        if message is None:
            return
        if not self._is_authorized(update):
            await message.reply_text(UNAUTHORIZED_MESSAGE)
            return

        thread_id = telegram_thread_id(update)
        try:
            async with self._agent_lock:
                await asyncio.to_thread(self._agent.reset, thread_id)
        except Exception:
            self._logger.exception(
                "Telegram conversation reset failed for thread %s",
                thread_id,
            )
            await message.reply_text(ERROR_MESSAGE)
            return
        await message.reply_text(NEW_CONVERSATION_MESSAGE)

    async def message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        del context
        message = update.effective_message
        if message is None or message.text is None:
            return
        if not self._is_authorized(update):
            await message.reply_text(UNAUTHORIZED_MESSAGE)
            return

        question = message.text.strip()
        if not question:
            return
        thread_id = telegram_thread_id(update)
        self._logger.info("Received Telegram question")
        queue_started_at = perf_counter()
        typing_task = asyncio.create_task(self._show_typing(message))
        try:
            try:
                async with self._agent_lock:
                    self._logger.info(
                        "Telegram queue wait: %.2fs",
                        perf_counter() - queue_started_at,
                    )
                    response = await asyncio.to_thread(
                        self._agent.ask,
                        question,
                        thread_id=thread_id,
                    )
            except Exception:
                self._logger.exception(
                    "Telegram agent request failed for thread %s",
                    thread_id,
                )
                await message.reply_text(ERROR_MESSAGE)
                return
        finally:
            typing_task.cancel()
            with suppress(asyncio.CancelledError):
                await typing_task

        parts = split_telegram_message(format_agent_response(response))
        send_started_at = perf_counter()
        for part in parts:
            await message.reply_text(part)
        self._logger.info(
            "Sent Telegram response in %.2fs: messages=%d",
            perf_counter() - send_started_at,
            len(parts),
        )

    async def _show_typing(self, message: Message) -> None:
        while True:
            try:
                await message.reply_chat_action(ChatAction.TYPING)
            except TelegramError:
                self._logger.warning(
                    "Telegram typing indicator failed",
                    exc_info=True,
                )
                return
            await asyncio.sleep(TYPING_REFRESH_INTERVAL_SECONDS)

    def _is_authorized(self, update: Update) -> bool:
        user = update.effective_user
        return user is not None and user.id in self._allowed_user_ids


def telegram_thread_id(update: Update) -> str:
    chat = update.effective_chat
    user = update.effective_user
    if chat is None or user is None:
        raise ValueError("Telegram update must include a chat and user.")
    if chat.type == Chat.PRIVATE:
        return f"telegram:chat:{chat.id}"
    return f"telegram:chat:{chat.id}:user:{user.id}"


def format_agent_response(response: AgentResponse) -> str:
    parts = []
    if response.status == "partial":
        parts.append("Partial answer")
    parts.append(response.answer)
    if response.references:
        sources = "\n".join(
            f"- {format_source_reference(reference)}"
            for reference in response.references
        )
        parts.append(f"Sources:\n{sources}")
    return "\n\n".join(parts)


def split_telegram_message(
    text: str,
    *,
    max_length: int = MAX_TELEGRAM_MESSAGE_LENGTH,
) -> list[str]:
    if max_length < 1:
        raise ValueError("Maximum message length must be positive.")

    remaining = text.strip()
    parts = []
    while len(remaining) > max_length:
        split_at = remaining.rfind("\n", 0, max_length + 1)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, max_length + 1)
        if split_at <= 0:
            split_at = max_length
        part = remaining[:split_at].rstrip()
        if not part:
            part = remaining[:max_length]
            split_at = max_length
        parts.append(part)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts

import asyncio
import threading
from types import SimpleNamespace

from telegram import Chat
from telegram.constants import ChatAction

from graphtool.agents import AgentResponse
from graphtool.retrieval import SourceReference
from telegram_bot.handlers import (
    ERROR_MESSAGE,
    NEW_CONVERSATION_MESSAGE,
    START_MESSAGE,
    UNAUTHORIZED_MESSAGE,
    TelegramHandlers,
    format_agent_response,
    split_telegram_message,
    telegram_thread_id,
)


class FakeAgent:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.ask_calls = []
        self.reset_calls = []

    def ask(self, question, *, thread_id):
        self.ask_calls.append((question, thread_id))
        if self.error is not None:
            raise self.error
        return self.response

    def reset(self, thread_id):
        self.reset_calls.append(thread_id)
        if self.error is not None:
            raise self.error


class FakeMessage:
    def __init__(self, text=None):
        self.text = text
        self.replies = []
        self.chat_actions = []

    async def reply_text(self, text):
        self.replies.append(text)

    async def reply_chat_action(self, action):
        self.chat_actions.append(action)


class BlockingAgent(FakeAgent):
    def __init__(self, response):
        super().__init__(response=response)
        self.started = threading.Event()
        self.release = threading.Event()

    def ask(self, question, *, thread_id):
        self.ask_calls.append((question, thread_id))
        self.started.set()
        self.release.wait()
        return self.response


def _update(*, user_id=123, chat_id=456, chat_type=Chat.PRIVATE, text=None):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
        effective_message=FakeMessage(text),
    )


def test_private_and_group_thread_ids_are_isolated():
    private_update = _update()
    group_update = _update(chat_type=Chat.GROUP)

    assert telegram_thread_id(private_update) == "telegram:chat:456"
    assert telegram_thread_id(group_update) == "telegram:chat:456:user:123"


def test_start_explains_conversation_behavior():
    update = _update()
    handlers = TelegramHandlers(FakeAgent(), frozenset({123}))

    asyncio.run(handlers.start(update, None))

    assert update.effective_message.replies == [START_MESSAGE]


def test_unauthorized_user_is_rejected():
    update = _update(user_id=999, text="Question")
    agent = FakeAgent()
    handlers = TelegramHandlers(agent, frozenset({123}))

    asyncio.run(handlers.message(update, None))

    assert agent.ask_calls == []
    assert update.effective_message.replies == [UNAUTHORIZED_MESSAGE]


def test_message_asks_agent_and_formats_sources():
    response = AgentResponse(
        answer="GraphTool builds a graph.",
        status="complete",
        references=[
            SourceReference(
                source="documents/manual.pdf",
                page_start=2,
                page_end=3,
            )
        ],
        search_count=1,
    )
    update = _update(text=" What does GraphTool do? ")
    agent = FakeAgent(response=response)
    handlers = TelegramHandlers(agent, frozenset({123}))

    asyncio.run(handlers.message(update, None))

    assert agent.ask_calls == [
        ("What does GraphTool do?", "telegram:chat:456")
    ]
    assert update.effective_message.replies == [
        "GraphTool builds a graph.\n\n"
        "Sources:\n- documents/manual.pdf (pp. 2-3)"
    ]


def test_message_logs_queue_and_send_timings(monkeypatch):
    async def run_in_place(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("telegram_bot.handlers.asyncio.to_thread", run_in_place)
    response = AgentResponse(
        answer="GraphTool builds a graph.",
        status="complete",
        search_count=1,
    )
    update = _update(text="What does GraphTool do?")
    handlers = TelegramHandlers(FakeAgent(response=response), frozenset({123}))
    calls = []
    handlers._logger = type(
        "FakeLogger",
        (),
        {"info": lambda self, *args: calls.append(args)},
    )()

    asyncio.run(handlers.message(update, None))

    assert calls[0] == ("Received Telegram question",)
    assert calls[1][0] == "Telegram queue wait: %.2fs"
    assert calls[2][0] == "Sent Telegram response in %.2fs: messages=%d"
    assert calls[2][2] == 1


def test_message_shows_typing_until_agent_finishes(monkeypatch):
    monkeypatch.setattr(
        "telegram_bot.handlers.TYPING_REFRESH_INTERVAL_SECONDS",
        0.01,
    )
    response = AgentResponse(
        answer="Finished.",
        status="complete",
        search_count=1,
    )
    update = _update(text="Question")
    agent = BlockingAgent(response)
    handlers = TelegramHandlers(agent, frozenset({123}))

    async def run():
        task = asyncio.create_task(handlers.message(update, None))
        try:
            assert await asyncio.to_thread(agent.started.wait, 1)
            async with asyncio.timeout(1):
                while len(update.effective_message.chat_actions) < 2:
                    await asyncio.sleep(0.001)
        finally:
            agent.release.set()
        await task

        action_count = len(update.effective_message.chat_actions)
        await asyncio.sleep(0.02)
        assert len(update.effective_message.chat_actions) == action_count

    asyncio.run(run())

    assert len(update.effective_message.chat_actions) >= 2
    assert all(
        action == ChatAction.TYPING
        for action in update.effective_message.chat_actions
    )
    assert update.effective_message.replies == ["Finished."]


def test_new_conversation_resets_current_thread():
    update = _update()
    agent = FakeAgent()
    handlers = TelegramHandlers(agent, frozenset({123}))

    asyncio.run(handlers.new_conversation(update, None))

    assert agent.reset_calls == ["telegram:chat:456"]
    assert update.effective_message.replies == [NEW_CONVERSATION_MESSAGE]


def test_new_conversation_error_sends_safe_message():
    update = _update()
    agent = FakeAgent(error=RuntimeError("secret internal error"))
    handlers = TelegramHandlers(agent, frozenset({123}))

    asyncio.run(handlers.new_conversation(update, None))

    assert update.effective_message.replies == [ERROR_MESSAGE]


def test_agent_error_sends_safe_message():
    update = _update(text="Question")
    agent = FakeAgent(error=RuntimeError("secret internal error"))
    handlers = TelegramHandlers(agent, frozenset({123}))

    asyncio.run(handlers.message(update, None))

    assert update.effective_message.replies == [ERROR_MESSAGE]


def test_partial_response_is_labeled():
    response = AgentResponse(
        answer="Best-effort answer.",
        status="partial",
        search_count=5,
    )

    assert format_agent_response(response) == (
        "Partial answer\n\nBest-effort answer."
    )


def test_clarification_response_is_not_labeled_as_partial():
    response = AgentResponse(
        answer="Which plan should I use?",
        status="needs_input",
        search_count=1,
    )

    assert format_agent_response(response) == "Which plan should I use?"


def test_long_messages_are_split_within_limit():
    text = "alpha beta gamma delta"

    parts = split_telegram_message(text, max_length=10)

    assert parts == ["alpha beta", "gamma", "delta"]
    assert all(len(part) <= 10 for part in parts)

# GraphTool

GraphTool builds and searches a knowledge graph from files placed under
`documents/`. Markdown, PDF, PowerPoint, and audio inputs are discovered
recursively. A
recommended layout is:

```text
documents/
├── markdown/
├── pdfs/
├── presentations/
└── recordings/
```

The folders are organizational; supported files can be nested anywhere under
`documents/` and are identified by extension.

Knowledge-folder scopes are configured in `config/knowledge_scopes.json`. Each
catalog name maps a user-friendly folder name to its path:

```json
{
  "work": "documents/work",
  "personal": "documents/personal",
  "finance": "documents/finance"
}
```

When a user explicitly asks the agent to search a catalog folder, such as
"search only my work folder," the selected scope applies to all retrieval for
that question. Direct chunk results and knowledge-graph paths are both limited
to documents under that folder. If no folder restriction is requested, the
agent searches the complete knowledge base. An unrecognized folder request
returns the available catalog names and asks the user to clarify.

PDFs are rendered page by page and converted to Markdown with the configured
`AZURE_OPENAI_FAST_DEPLOYMENT`. Converted pages are cached under
`data/pdf_conversions/`, so unchanged PDFs do not require additional model calls.

PowerPoint `.pptx` files are converted to PDF with LibreOffice, one slide per
page, and then use the same text-and-image PDF conversion pipeline. Generated
PDFs are cached under `data/presentation_conversions/`; source references retain
the original `.pptx` path and display page ranges as slide ranges. Speaker notes,
comments, animations, and embedded media are not ingested.

## PDF and PowerPoint requirements

Install the project dependencies with `uv sync`. PDF rendering also requires
Poppler's `pdftoppm` executable, and PowerPoint conversion requires LibreOffice's
`soffice` executable. Audio transcription requires the `ffmpeg` and `ffprobe`
executables:

```sh
# macOS
brew install poppler
brew install --cask libreoffice
brew install ffmpeg

# Ubuntu or Debian
sudo apt-get install poppler-utils
sudo apt-get install libreoffice
sudo apt-get install ffmpeg
```

The configured fast Azure OpenAI deployment must support image input and
structured output. Password-protected PDFs are not supported.

## Audio transcription

GraphTool accepts `.flac`, `.m4a`, `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.ogg`,
`.wav`, and `.webm` files. Audio is normalized to mono 16 kHz MP3 at 64 kbps,
split into approximately 20-minute chunks with five seconds of overlap, and
transcribed sequentially with `AZURE_OPENAI_TRANSCRIPTION_DEPLOYMENT`. Overlap
is reconciled from bounded transcript windows so minor transcription variation
does not duplicate boundary text.

Raw chunk transcripts and assembled Markdown are cached separately under
`data/audio_transcriptions/`. This directory is generated data; put original
recordings under `documents/`, preferably `documents/recordings/`. Interrupted
conversions resume from the last cached chunk, and assembly changes reuse raw
transcripts. The configured deployment should use `gpt-4o-transcribe`.

Copy `config/transcription_glossary.example.json` to
`config/transcription_glossary.json`, then add names, project names, acronyms,
and other expected proper nouns. The local glossary is gitignored so personal
terms are not committed. If it does not exist, ingestion uses an empty list.
Each entry is a canonical spelling sent to the transcription model as a hint
and to the fast model for a constrained correction pass:

```json
{
  "terms": [
    "HIP-SA",
    "Aishwarya Rao"
  ]
}
```

The raw transcript chunks remain unchanged. Proposed proper-noun substitutions
are stored in `corrections.jsonl` beside each recording's cached chunks. An
`apply` correction is included in the assembled Markdown, while `review` and
`reject` corrections are not. To undo an automatic correction, change its
`decision` to `reject` and set `reviewed` to `true`. Deleting the row also
removes the correction, but a reviewed rejection is retained if correction
proposals are regenerated.

Editing `corrections.jsonl` rebuilds the Markdown from the raw chunks without
rerunning audio transcription. Only exact replacements with terms still in the
glossary are applied. The correction records are local generated data and are
not used to train or update the models.

Changing the glossary invalidates cached audio transcripts so they are
transcribed again with the updated terms.

## Knowledge agent

Set `AZURE_OPENAI_AGENT_DEPLOYMENT` to an Azure OpenAI deployment that supports
structured output. Ingest documents and update the knowledge base with:

```sh
uv run python ingest.py
```

Ingestion and agent conversations run separately. To start a conversation using
the latest completed ingestion, run:

```sh
uv run python main.py
```

Do not start the agent while ingestion is running because the persisted stores
are updated in place. If documents have changed but ingestion has not been run,
the agent continues to use the previous knowledge base.

You can also create the read-only agent through the Python API:

```python
from graphtool.agents import create_knowledge_agent
from graphtool.llm import (
    create_azure_openai_agent_model,
    create_azure_openai_fast_agent_model,
    load_azure_openai_config,
)
from graphtool.runtime import create_runtime

config = load_azure_openai_config()
runtime = create_runtime(config)
runtime.prepare_search()
answer_model = create_azure_openai_agent_model(config)
orchestration_model = create_azure_openai_fast_agent_model(config)
agent = create_knowledge_agent(answer_model, orchestration_model, runtime)

response = agent.ask("What can GraphTool do?", thread_id="demo")
print(response.answer)
print(response.references)
```

Call `runtime.prepare_search()` after each document synchronization. It loads the
current knowledge base, prepares reusable search indexes, and refreshes stale chunk
embeddings before queries are served.

Document graphs, canonical entities, relationships, provenance, aliases, chunks,
and embeddings are stored in `data/graphtool.db`. Graph and embedding updates use
row-level SQLite upserts and targeted deletes. Semantic entity resolution loads
normalized `float32` embedding matrices by compatible entity set and reuses them
for exact cosine-similarity candidate search. Corpus synchronization stages
computed embeddings and commits all SQLite-backed artifacts through one shared
transaction.

Calls using the same thread ID share conversation history while the process is
running. By default, older history is summarized when the conversation reaches
approximately 256,000 tokens, while the most recent 64,000 tokens remain
verbatim. These limits can be changed with `compaction_trigger_tokens` and
`retained_recent_tokens` when creating the agent.

The agent binds four tools. `ask_user` pauses the current graph when a material
ambiguity or missing input can only be resolved by the user. The next message on
the same thread resumes the paused graph with its evidence and research progress
preserved. Pending questions use the in-memory checkpointer and therefore do not
survive a process restart. `find_documents` discovers canonical source IDs from
filenames, titles, headings, and document topics.
`search_knowledge_base` searches document chunks and knowledge-graph paths across
the selected folder or, when canonical source IDs are supplied, only within those
documents. `get_chunk_neighborhood` retrieves the previous, current, and next
chunks around a search result when adjacent context is needed. Source-filtered
search accepts only IDs returned by `find_documents` for the current question, and
neighborhood lookup accepts only chunks returned by an earlier knowledge search
for the same subquestion. The agent decomposes compound questions into at most
five non-overlapping subquestions and makes up to three retrieval tool calls for
each one. It returns `status="partial"` when the available evidence remains
incomplete and `status="needs_input"` when it is waiting for clarification.
Requests that can be completed from the conversation, user-supplied content, or
general knowledge—such as identity questions, translation, and reformatting—can
be answered without knowledge-base retrieval.

## Telegram bot

Create a bot by sending `/newbot` to [BotFather](https://t.me/BotFather), then
add its token and the numeric Telegram user IDs allowed to use it to `.env`:

```env
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_ALLOWED_USER_IDS=123456789
```

To find your numeric user ID, send the new bot a message before starting
GraphTool and inspect the `message.from.id` value returned by Telegram's
[`getUpdates`](https://core.telegram.org/bots/api#getupdates) Bot API method.
Multiple allowed user IDs can be separated with commas. After ingesting the
documents, start the bot with:

```sh
uv run python -m telegram_bot
```

The bot uses long polling, so it does not require a public webhook. Messages in
the same private chat share conversation context while the process is running.
In group chats, each user has separate context. Send `/new` to clear the current
conversation. Restarting the bot clears all conversations because history is
stored in memory.

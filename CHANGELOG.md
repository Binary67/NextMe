# Changelog

This changelog records changes from July 24, 2026 onward. Each entry explains
the intention behind the work and the features or code changes that implemented
it.

## July 25, 2026

### Deterministic research routing and progress tracking

#### Intention

- Prevent useful single-chunk retrievals from being mistaken for no progress.
- Let evidence evaluation recommend focused searches or authorized adjacent
  context while keeping stop decisions under deterministic workflow control.
- Prevent invalid, duplicate, or multiple tool selections from consuming the
  retrieval budget.

#### Changes implemented

- Replaced prose similarity-based early stopping with consecutive empty-result
  tracking and retained the fixed per-subquestion retrieval limit.
- Changed evaluator output to list atomic missing facts and return a structured
  `search` or `expand` recommendation with the required focus or chunk target.
- Forced initial research to use document discovery or knowledge search,
  validated expansion targets against unused returned chunks, and fell back to
  focused search when expansion was unavailable.
- Added corrective retries for duplicate searches, multiple tool calls, and
  tool selections that conflict with the recommended action.
- Counted newly discovered citable graph paths as evidence progress and reset
  document and chunk authorization state between subquestions.
- Added focused tests for single-chunk progress, empty-result stopping,
  recommendation routing, expansion fallback, duplicate-query correction, and
  exactly-one-tool enforcement.

---

### Document discovery and source-filtered retrieval

#### Intention

- Let users identify documents by filename, title, heading, or topic before
  asking questions about their contents.
- Prevent named-document questions from mixing evidence from similarly relevant
  sources.
- Keep document-specific retrieval scalable without caching a complete
  retriever for every source.

#### Changes implemented

- Added a prepared document index over canonical source paths, filenames,
  derived titles, and headings, combined with passage-level topic relevance.
- Added the read-only `find_documents` agent tool and carried its canonical
  source IDs into research context and citable document evidence.
- Extended `search_knowledge_base` with optional source filtering and rejected
  source IDs that were not returned by document discovery for the question.
- Added ephemeral source-filtered hybrid retrieval that filters chunks and
  source-provenance graph evidence without creating unbounded per-document
  caches.
- Included source metadata in lexical and semantic chunk-search inputs.
- Added focused coverage for filename and topic discovery, folder isolation,
  source authorization, source-filtered graph paths, document-list questions,
  and document-specific factual questions.
- Updated the knowledge-agent API example and tool documentation in
  `README.md`.

---

### Citable graph-path evidence

#### Intention

- Preserve ordered multi-hop graph relationships through evidence evaluation
  and final answer generation.
- Expose graph paths only when every supporting chunk has a valid citation.

#### Changes implemented

- Added structured graph-path evidence to knowledge-search artifacts and agent
  state, including supporting chunk and reference identifiers.
- Carried citable graph paths into evaluator and answer prompts while excluding
  paths whose supporting chunks were not all returned.
- Deduplicated graph-path evidence across subquestions and added focused tests
  for propagation, citation coverage, exclusion, and deduplication.

---

### Auditable proper-noun corrections

#### Intention

- Improve proper-noun reliability without requiring users to maintain
  pronunciations or transcription variants.
- Keep automatic corrections auditable and reversible while preserving the raw
  transcript.
- Let human decisions rebuild derived transcripts without another transcription
  request.
- Support only the current generated-data formats while the project remains in
  active development.

#### Changes implemented

- Added a structured fast-model correction pass after audio transcription.
- Restricted corrections to exact canonical terms from
  `config/transcription_glossary.json`, uniquely located source text, and
  non-overlapping substitutions.
- Added a per-recording `corrections.jsonl` ledger with `apply`, `review`, and
  `reject` decisions plus a `reviewed` marker.
- Kept raw transcript chunks immutable and made assembled Markdown a derived
  artifact of the raw chunks and active correction ledger.
- Made ledger edits and row deletion rebuild Markdown without rerunning audio
  transcription or correction generation.
- Preserved reviewed decisions when correction proposals are regenerated.
- Added correction-model and ledger hashes to audio manifests while requiring
  the current manifest schema rather than migrating legacy cache data.
- Wired the runtime fast model into audio ingestion and documented the glossary
  and correction review workflow in `README.md`.
- Added focused tests for automatic application, deferred review, rejection,
  deletion, raw-chunk preservation, cache reuse, reviewed-decision retention,
  and non-glossary replacement rejection.
- Added `AGENTS.md` to prohibit migrations and compatibility paths for generated
  development data unless explicitly requested.

## July 24, 2026

### Intention

- Let users limit a question to a known document folder using natural language,
  without requiring them to know or enter an internal source path.
- Preserve broad knowledge-base search as the default when the user does not
  request a folder.
- Prevent an unknown or ambiguous folder request from silently searching the
  complete knowledge base.
- Keep folder-scoped answers grounded only in document chunks and graph
  relationships supported by the selected folder.

### Changes implemented

- Added `config/knowledge_scopes.json` as the catalog that maps user-facing scope
  names such as `work`, `personal`, and `finance` to folders under `documents/`.
- Added catalog loading and validation, including normalized scope names, safe
  document-folder paths, folder-boundary matching, and a reserved `all` name for
  unrestricted search.
- Extended question decomposition to select one catalog scope only when the user
  explicitly requests a folder. Questions without a folder restriction search
  all documents.
- Added clarification behavior for folder requests that do not match the
  catalog.
- Stored the selected scope once for the question and automatically applied it
  to every knowledge search performed for its subquestions.
- Added scoped runtime retrieval that filters both document chunks and
  source-provenance graph entities and relationships.
- Added per-scope retrieval caching while retaining the existing unrestricted
  search index.
- Documented the scope catalog and natural-language behavior in `README.md`.
- Added focused tests for catalog validation, default unrestricted search,
  selected-scope enforcement, unknown-scope clarification, source filtering, and
  graph-path filtering.

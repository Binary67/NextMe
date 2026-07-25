_CONVERSION_INSTRUCTIONS = (
    "Convert PDF pages into faithful Markdown for downstream knowledge extraction. "
    "Transcribe rather than summarize or rewrite. Preserve the original reading "
    "order, heading hierarchy, paragraphs, lists, tables, code, footnotes, captions, "
    "and visible links. Omit repeated headers, footers, and printed page numbers. "
    "If a page contains only repeated template content, such as headers, footers, "
    "confidentiality labels, logos, copyright notices, or page numbers, set "
    "is_blank to true and return empty Markdown for that page. Continue processing "
    "the remaining pages and still return one page record for every requested page. "
    "Describe meaningful figures only from clearly visible content, and never infer "
    "unreadable values or facts. Mark unreadable content as [Unclear]. Do not wrap "
    "Markdown in a code fence and do not add page markers; the caller adds them."
)
BATCH_SYSTEM_PROMPT = (
    f"{_CONVERSION_INSTRUCTIONS} "
    "Return exactly one page record for every requested page, in request order."
)
SINGLE_PAGE_SYSTEM_PROMPT = (
    f"{_CONVERSION_INSTRUCTIONS} Return conversion content for exactly the requested "
    "page. Do not return a page number; the caller assigns it."
)

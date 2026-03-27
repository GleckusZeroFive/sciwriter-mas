"""Quality gate — two-level check before publishing.

Level 1: Regex/heuristic checks (instant)
Level 2: LLM verification via Qwen (slow, optional)

Designed to catch common Qwen3 8B artifacts:
- Chinese characters leaked into Russian text
- Untranslated English blocks
- Thinking tags (<think>, /no_think)
- Markdown garbage (excessive *** or ---)
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# --- Regex patterns ---

# Chinese characters: CJK Unified Ideographs + Extension A
CHINESE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# Long English blocks: 20+ consecutive Latin words among Cyrillic text
ENGLISH_BLOCK_RE = re.compile(r"(?:(?:[A-Za-z]+['\-]?[A-Za-z]*)\s+){20,}")

# LLM thinking artifacts
THINKING_RE = re.compile(r"</?think>|/no_think|<\|.*?\|>", re.IGNORECASE)

# Markdown garbage: 3+ consecutive *** or --- on separate lines
MD_GARBAGE_RE = re.compile(r"^(\*{3,}|\-{3,})\s*$", re.MULTILINE)

# Repetitive paragraphs (same text within 500 chars)
REPEAT_WINDOW = 500


@dataclass
class QualityReport:
    """Result of quality gate checks."""
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    chinese_count: int = 0
    english_blocks: int = 0
    thinking_tags: int = 0
    md_garbage: int = 0
    char_count: int = 0
    has_headings: bool = False
    has_sources: bool = False
    duplicate_paragraphs: int = 0

    def summary(self) -> str:
        if self.passed:
            return f"PASS ({self.char_count} chars, {len(self.issues)} minor issues)"
        return f"FAIL: {'; '.join(self.issues)}"


def check_level1(
    text: str,
    min_length: int = 4000,
    min_sources: int = 1,
) -> QualityReport:
    """Level 1: fast regex/heuristic checks. Returns QualityReport."""
    report = QualityReport()
    report.char_count = len(text)

    # --- Length check ---
    if len(text) < min_length and min_length > 0:
        report.issues.append(f"Too short: {len(text)} chars (min {min_length})")
        report.passed = False

    # --- Chinese artifacts ---
    chinese_matches = CHINESE_RE.findall(text)
    report.chinese_count = len(chinese_matches)
    if report.chinese_count > 0:
        sample = "".join(chinese_matches[:5])
        report.issues.append(f"Chinese artifacts found ({report.chinese_count}): {sample}")
        report.passed = False

    # --- English blocks ---
    english_blocks = ENGLISH_BLOCK_RE.findall(text)
    report.english_blocks = len(english_blocks)
    if report.english_blocks > 0:
        report.issues.append(f"Untranslated English blocks: {report.english_blocks}")
        # Not auto-fail: some English (names, terms) is expected
        if report.english_blocks > 3:
            report.passed = False

    # --- Thinking tags ---
    thinking = THINKING_RE.findall(text)
    report.thinking_tags = len(thinking)
    if report.thinking_tags > 0:
        report.issues.append(f"LLM thinking tags found: {thinking[:3]}")
        report.passed = False

    # --- Markdown garbage ---
    md_garbage = MD_GARBAGE_RE.findall(text)
    report.md_garbage = len(md_garbage)
    if report.md_garbage > 2:
        report.issues.append(f"Excessive markdown separators: {report.md_garbage}")
        report.passed = False

    # --- Structure: headings ---
    headings = re.findall(r"^#{1,3}\s+.+", text, re.MULTILINE)
    report.has_headings = len(headings) >= 2
    if not report.has_headings:
        report.issues.append("Missing H2/H3 headings (need at least 2)")
        report.passed = False

    # --- Sources ---
    # Check for URLs or reference markers
    urls = re.findall(r"https?://\S+", text)
    refs = re.findall(r"\[\d+\]|\[source\]|\[ссылка\]", text, re.IGNORECASE)
    source_count = len(urls) + len(refs)
    report.has_sources = source_count >= min_sources
    if not report.has_sources:
        report.issues.append(f"Not enough source references: {source_count} (min {min_sources})")
        # Not auto-fail: some articles legitimately have no external links

    # --- Duplicate paragraphs ---
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
    seen = set()
    dupes = 0
    for p in paragraphs:
        # Normalize whitespace for comparison
        normalized = " ".join(p.split())
        if normalized in seen:
            dupes += 1
        seen.add(normalized)
    report.duplicate_paragraphs = dupes
    if dupes > 0:
        report.issues.append(f"Duplicate paragraphs found: {dupes}")
        report.passed = False

    # --- Cyrillic in URLs ---
    urls = re.findall(r"https?://\S+", text)
    cyrillic_url_re = re.compile(r"[\u0400-\u04ff]")
    for url in urls:
        if cyrillic_url_re.search(url):
            report.issues.append(f"Cyrillic character in URL: {url[:80]}")
            report.passed = False

    # --- Repeated facts (same sentence structure appearing 3+ times) ---
    sentences = re.split(r"[.!?]\s", text)
    sentence_starts = {}
    for s in sentences:
        words = s.strip().split()[:4]
        if len(words) >= 4:
            key = " ".join(words).lower()
            sentence_starts[key] = sentence_starts.get(key, 0) + 1
    for key, count in sentence_starts.items():
        if count >= 3:
            report.issues.append(f"Repetitive sentence pattern (x{count}): '{key}...'")

    return report


def clean_artifacts(text: str) -> str:
    """Remove known artifacts from text. Returns cleaned text."""
    # Remove thinking tags
    text = THINKING_RE.sub("", text)

    # Remove Chinese characters (replace with empty string)
    text = CHINESE_RE.sub("", text)

    # Remove excessive markdown separators (keep max 1)
    text = re.sub(r"^(\*{3,}|\-{3,})\s*\n", "", text, flags=re.MULTILINE)

    # Remove SEO spam sections (Meta Description, Keywords)
    text = re.sub(r"##\s*Meta\s*Description\s*\n.*?(?=\n##|\n---|\Z)", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"##\s*Keywords?\s*\n.*?(?=\n##|\n---|\Z)", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Clean [FACT-N] tags: replace [FACT-N](url) with just [source](url)
    text = re.sub(r"\[FACT-\d+\]\(([^)]+)\)", r"[источник](\1)", text)
    # Remove standalone [FACT-N] references without URLs
    text = re.sub(r"\[FACT-\d+\]", "", text)

    # Fix Cyrillic characters leaked into URLs
    cyrillic_in_url = re.compile(r"(https?://\S*?)([\u0400-\u04ff]+)(\S*)")
    def _fix_cyrillic_url(m):
        # Map common Cyrillic→Latin lookalikes
        cyr_to_lat = {"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y",
                       "х": "x", "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C",
                       "л": "l", "н": "n", "т": "t", "и": "i", "к": "k", "м": "m"}
        fixed = ""
        for ch in m.group(2):
            fixed += cyr_to_lat.get(ch, "")
        return m.group(1) + fixed + m.group(3)

    for _ in range(5):  # multiple passes for multiple Cyrillic chars in one URL
        new_text = cyrillic_in_url.sub(_fix_cyrillic_url, text)
        if new_text == text:
            break
        text = new_text

    # Remove Changelog section (editor artifact)
    text = re.sub(r"##\s*Changelog\s*\n.*?(?=\n##|\Z)", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove empty lines caused by cleanup (max 2 consecutive)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Remove trailing --- separators
    text = re.sub(r"\n---\s*$", "", text)

    return text.strip()


# --- Number extraction and validation ---

# Patterns for numbers/specs in text
NUMBER_PATTERNS = [
    re.compile(r"\$[\d,.]+"),                          # prices: $200, $1,500
    re.compile(r"\d+[\s]?(?:ГГц|МГц|GHz|MHz)"),       # frequencies
    re.compile(r"\d+[\s]?(?:ГБ|МБ|TB|GB|MB)"),        # memory/storage
    re.compile(r"\d+[\s]?(?:Вт|кВт|W|kW)"),           # power
    re.compile(r"\d+[\s]?(?:А|мА|A|mA)"),              # current
    re.compile(r"\d+[\s]?(?:В|V)"),                    # voltage
    re.compile(r"\d+[\s]?(?:Гбит/с|Мбит/с|Gbps|Mbps)"),  # bandwidth
    re.compile(r"\d+[\s]?(?:FPS|fps|Гц|Hz)"),          # framerates
    re.compile(r"\d+[\s]?(?:мм|см|м|mm|cm|m)\b"),      # dimensions
    re.compile(r"\d+[\s]?(?:г|кг|g|kg)\b"),            # weight
    re.compile(r"\d+[\s]?%"),                           # percentages
    re.compile(r"\d+[\s]?(?:секунд|минут|часов|дней)"), # time
]


def extract_numbers_from_text(text: str) -> list[dict]:
    """Extract all number/spec mentions from article text.

    Returns list of {value, context, line_num} dicts.
    Context is ~50 chars around the number for LLM validation.
    """
    results = []
    lines = text.split("\n")

    for line_num, line in enumerate(lines, 1):
        for pattern in NUMBER_PATTERNS:
            for match in pattern.finditer(line):
                start = max(0, match.start() - 30)
                end = min(len(line), match.end() + 30)
                context = line[start:end].strip()
                results.append({
                    "value": match.group(),
                    "context": context,
                    "line": line_num,
                })

    # Deduplicate by value+context
    seen = set()
    unique = []
    for r in results:
        key = r["value"] + r["context"]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def validate_facts_deterministic(text: str) -> tuple[str, list[str]]:
    """Deterministic fact validation: any sentence with a number MUST have [FACT-N] nearby.

    If a sentence contains a number/spec but no [FACT-N] tag, the sentence is removed.
    Returns (cleaned_text, list_of_removed_sentences).

    This is the core anti-hallucination mechanism:
    - Research extracts facts as [FACT-N]
    - Writer must tag every claim with [FACT-N]
    - This function enforces it: no tag = hallucination = removed
    """
    # Pattern: any number-like content (digits with units, prices, percentages, specs)
    HAS_NUMBER = re.compile(
        r"\$[\d,.]+"           # prices
        r"|\d+\s*(?:ГГц|МГц|GHz|MHz|ГБ|МБ|GB|MB|TB|Вт|кВт|W|kW"
        r"|А|мА|A|mA|В|V|Гбит/с|Мбит/с|Gbps|Mbps|FPS|fps|Гц|Hz"
        r"|мм|см|м|mm|cm|г|кг|g|kg|%"
        r"|USD|руб|EUR|секунд|минут|часов|дней|млн|млрд|тонн)"
        r"|\d{2,}",           # any number with 2+ digits
        re.IGNORECASE
    )

    # Pattern: [FACT-N] tag nearby
    HAS_FACT_TAG = re.compile(r"\[FACT-\d+\]", re.IGNORECASE)

    lines = text.split("\n")
    cleaned_lines = []
    removed = []

    for line in lines:
        stripped = line.strip()

        # Skip headings, empty lines, list markers without numbers, code blocks
        if (not stripped
                or stripped.startswith("#")
                or stripped.startswith("```")
                or stripped.startswith("![")
                or stripped.startswith("[FACT-")
                or stripped.startswith("- **") and not HAS_NUMBER.search(stripped)):
            cleaned_lines.append(line)
            continue

        # Check: does this line have a number?
        if HAS_NUMBER.search(stripped):
            # Does it also have a [FACT-N] tag?
            if HAS_FACT_TAG.search(stripped):
                cleaned_lines.append(line)  # Tagged = trusted
            else:
                # Untagged number = potential hallucination → remove
                removed.append(stripped)
                # Replace with empty line (don't break structure)
                continue
        else:
            cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    # Clean up excessive empty lines from removals
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result, removed


def build_llm_verification_prompt(text: str) -> str:
    """Build prompt for Level 2 LLM verification."""
    return (
        "Ты — редактор-корректор. Проверь текст статьи на качество русского языка.\n\n"
        "Найди:\n"
        "1. Иностранные слова/фразы без перевода или пояснения\n"
        "2. Незаконченные предложения или оборванные мысли\n"
        "3. Повторяющиеся абзацы или фразы\n"
        "4. Бессмысленные или нелогичные фрагменты\n"
        "5. Грамматические ошибки\n"
        "6. Иероглифы или символы других языков\n\n"
        "Ответь строго в формате JSON:\n"
        '{"ok": true/false, "issues": ["описание проблемы 1", "описание проблемы 2"]}\n\n'
        "Если проблем нет, верни: {\"ok\": true, \"issues\": []}\n\n"
        f"Текст статьи:\n\n{text}"
    )

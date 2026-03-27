"""LangGraph node functions — each node calls a CrewAI agent or runs logic."""

import logging
import yaml
from pathlib import Path

import os

from crewai import Task, Crew, Process, LLM

from app.config import settings
from app.graph.state import ArticleState
from app.agents.researcher import create_researcher
from app.agents.writer import create_writer
from app.agents.fact_checker import create_fact_checker
from app.agents.editor import create_editor
from app.agents.reviewer import create_reviewer

logger = logging.getLogger(__name__)


def _get_llm() -> LLM:
    # CrewAI natively supports Ollama via 'ollama/' prefix
    return LLM(
        model=f"ollama/{settings.llm_model}",
        base_url=settings.llm_base_url.replace("/v1", ""),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=300,  # 5 min timeout per LLM call, prevents hanging on Ollama idle unload
    )


def _load_preset(name: str) -> dict:
    preset_path = settings.presets_dir / f"{name}.yml"
    if not preset_path.exists():
        logger.warning("Preset %s not found, using defaults", name)
        return {}
    with open(preset_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _add_log(state: dict, message: str) -> list[str]:
    log = list(state.get("log") or [])
    log.append(message)
    return log


# --- Node: Research ---

def research_node(state: ArticleState) -> dict:
    """Researcher agent finds sources on the topic.

    Two modes:
    - Factory mode (pre_sources populated): summarize pre-collected data, skip web search
    - Normal mode: search the web and knowledge base
    """
    topic = state["topic"]
    preset = _load_preset(state.get("preset", "habr"))
    pre_sources = state.get("pre_sources", [])
    llm = _get_llm()

    # Factory mode: use pre-collected sources from DB
    if pre_sources:
        logger.info("[RESEARCH] Factory mode: %d pre-collected sources", len(pre_sources))
        source_text = "\n\n---\n\n".join(
            f"Source: [{s.get('source', 'unknown')}] {s.get('title', 'N/A')}\n"
            f"URL: {s.get('url', 'N/A')}\n"
            f"Summary: {s.get('summary', '')}\n"
            f"Content: {(s.get('content') or '')[:2000]}"
            for s in pre_sources
        )

        agent = create_researcher(llm)
        task = Task(
            description=(
                f"You have pre-collected sources about: '{topic}'\n\n"
                f"Extract a numbered list of KEY FACTS from these sources.\n"
                f"Use ONLY information that is EXPLICITLY stated in the source text.\n"
                f"Do NOT infer, guess, or add any numbers/specs not present in sources.\n\n"
                f"For each fact, write one line:\n"
                f"[FACT-N] <specific detail in Russian> (Source: <url>)\n\n"
                f"What counts as a fact:\n"
                f"- Exact numbers from the text: prices, specs, measurements\n"
                f"- Names mentioned: models, chips, tools, people\n"
                f"- Direct quotes (keep in original language)\n"
                f"- Problems described and solutions used\n"
                f"- Concrete steps the author took\n\n"
                f"CRITICAL: If a number or spec is NOT in the source text, do NOT include it.\n"
                f"Better to have 5 real facts than 20 made-up ones.\n\n"
                f"Pre-collected sources:\n{source_text}"
            ),
            expected_output=(
                "A numbered list of facts ONLY from source text.\n"
                "Format: [FACT-1] <detail> (Source: <url>)\n"
                "No invented numbers. No guesses. Only what sources say."
            ),
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()

        return {
            "sources": str(result),
            "source_count": len(pre_sources),
            "status": "writing",
            "log": _add_log(state, f"[RESEARCH] Summarized {len(pre_sources)} pre-collected sources on: {topic}"),
        }

    # Normal mode: web search
    agent = create_researcher(llm)
    task = Task(
        description=(
            f"Research the topic: '{topic}'\n\n"
            f"Find 5-10 reliable sources with specific facts, numbers, and quotes.\n"
            f"Search both the web and the local knowledge base.\n"
            f"Focus on: recent developments, key statistics, expert opinions.\n"
            f"Language: Russian.\n"
            f"Target audience: {preset.get('audience', 'general readers')}"
        ),
        expected_output=(
            "A structured list of sources with:\n"
            "- Source title and URL (if available)\n"
            "- Key facts and quotes from each source\n"
            "- Relevance to the topic\n"
            "At least 5 sources with specific data points."
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = crew.kickoff()

    return {
        "sources": str(result),
        "source_count": str(result).count("---") + 1,
        "status": "writing",
        "log": _add_log(state, f"[RESEARCH] Found sources on: {topic}"),
    }


# --- Node: Write (sectional approach) ---

def write_node(state: ArticleState) -> dict:
    """Sectional writer: Plan → Write each section → Assemble.

    Step 1 (Planner): LLM splits facts into 5-7 sections
    Step 2 (Section Writer): LLM writes each section from its assigned facts (200-400 words)
    Step 3 (Assembler): Python joins sections into one article

    This keeps each LLM call small and grounded — 8B model can't hallucinate
    when writing 200 words from 2-3 specific facts.
    """
    topic = state["topic"]
    sources = state.get("sources", "")
    preset = _load_preset(state.get("preset", "habr"))
    llm = _get_llm()

    # --- Step 1: Planner ---
    logger.info("[WRITE] Step 1: Planning sections...")
    agent = create_writer(llm)
    plan_task = Task(
        description=(
            f"Topic: '{topic}'\n"
            f"Available facts:\n{sources}\n\n"
            f"Create a plan for an article with 5-7 sections.\n"
            f"For each section, write one line:\n"
            f"SECTION: [title] | FACTS: [comma-separated FACT numbers to use]\n\n"
            f"Example:\n"
            f"SECTION: Введение | FACTS: 1, 2\n"
            f"SECTION: Технические характеристики | FACTS: 3, 4, 5\n"
            f"SECTION: Проблемы и решения | FACTS: 6, 7\n"
            f"SECTION: Заключение | FACTS: 1\n\n"
            f"Rules:\n"
            f"- Every FACT must appear in at least one section\n"
            f"- Each section uses 1-3 facts\n"
            f"- Write section titles in Russian\n"
            f"- Do NOT add sections without facts (no 'Future work', no 'Applications')\n"
        ),
        expected_output="List of sections with FACT numbers, one per line.",
        agent=agent,
    )
    plan_crew = Crew(agents=[agent], tasks=[plan_task], process=Process.sequential, verbose=True)
    plan_result = str(plan_crew.kickoff())

    # Parse plan
    import re
    sections = []
    for line in plan_result.split("\n"):
        match = re.match(r"SECTION:\s*(.+?)\s*\|\s*FACTS?:\s*(.+)", line, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            fact_nums = match.group(2).strip()
            sections.append({"title": title, "facts": fact_nums})

    if not sections:
        # Fallback: simple 3-section plan
        logger.warning("[WRITE] Could not parse plan, using fallback")
        sections = [
            {"title": "Введение", "facts": "1, 2"},
            {"title": "Основная часть", "facts": "3, 4, 5"},
            {"title": "Заключение", "facts": "1"},
        ]

    logger.info("[WRITE] Plan: %d sections", len(sections))

    # --- Step 2: Write each section ---
    section_texts = []
    for i, sec in enumerate(sections):
        logger.info("[WRITE] Step 2.%d: Writing '%s' (facts: %s)", i+1, sec['title'], sec['facts'])

        sec_agent = create_writer(llm)
        sec_task = Task(
            description=(
                f"Write ONE section of an article.\n\n"
                f"Section title: {sec['title']}\n"
                f"Use ONLY these facts:\n"
                f"{sources}\n"
                f"(focus on facts: {sec['facts']})\n\n"
                f"Rules:\n"
                f"- Write 200-400 words in Russian\n"
                f"- Use ONLY information from the facts above\n"
                f"- Do NOT invent any numbers, names, or details\n"
                f"- If you don't have enough facts, write less — don't pad\n"
                f"- Do NOT write a title — just the body text\n"
                f"- Reference sources as [источник](url)\n"
            ),
            expected_output="Section body text in Russian, 200-400 words. No title.",
            agent=sec_agent,
        )
        sec_crew = Crew(agents=[sec_agent], tasks=[sec_task], process=Process.sequential, verbose=True)
        sec_result = str(sec_crew.kickoff())
        section_texts.append({"title": sec["title"], "text": sec_result})

    # --- Step 3: Assemble (pure Python) ---
    logger.info("[WRITE] Step 3: Assembling %d sections", len(section_texts))
    article_parts = [f"# {topic}\n"]
    for sec in section_texts:
        article_parts.append(f"\n## {sec['title']}\n")
        article_parts.append(sec["text"].strip())
        article_parts.append("")

    draft = "\n".join(article_parts)
    version = state.get("draft_version", 0) + 1

    return {
        "draft": draft,
        "draft_version": version,
        "status": "rating",
        "log": _add_log(state, f"[WRITE] Sectional draft v{version}: {len(sections)} sections, {len(draft)} chars"),
    }


# --- Node: Enrich (verify entities via web search, no LLM) ---

def enrich_node(state: ArticleState) -> dict:
    """Verify technical entities (chip names, protocols, specs) via DuckDuckGo.

    Pure Python + web search, no LLM call.
    Confirmed entities stay, unverified/contradicted lines are removed.
    """
    draft = state.get("draft", "")
    topic = state.get("topic", "")

    from app.factory.fact_enricher import enrich_article
    cleaned, log = enrich_article(draft, topic)

    confirmed = sum(1 for v in log if v["status"] == "confirmed")
    removed = sum(1 for v in log if v["status"] != "confirmed")

    return {
        "draft": cleaned,
        "status": "rating",
        "log": _add_log(state, f"[ENRICH] {len(log)} entities checked: {confirmed} confirmed, {removed} removed"),
    }


# --- Node: Rate ---

def rate_node(state: ArticleState) -> dict:
    """Rater reads the draft and produces a structured checklist per section.

    One job only: evaluate and produce actionable feedback.
    Does NOT rewrite the article — that's Improver's job.
    """
    draft = state.get("draft", "")
    llm = _get_llm()

    agent = create_reviewer(llm)
    task = Task(
        description=(
            f"Here is an article:\n\n{draft}\n\n"
            f"Rate this article from 0 to 100.\n"
            f"Then for each section (heading), write one line of feedback.\n\n"
            f"Format your response EXACTLY like this:\n"
            f"RATING: [number]\n\n"
            f"CHECKLIST:\n"
            f"- Section [name]: [what could be better — be specific]\n"
            f"- Section [name]: [ok / what could be better]\n"
            f"- Duplicates: [list any repeated information across sections]\n"
            f"- Missing: [what important info is absent]\n"
        ),
        expected_output=(
            "RATING: [number]\n\nCHECKLIST:\n- Section ...: ...\n- Section ...: ..."
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = str(crew.kickoff())

    # Extract rating
    import re
    score = 70.0
    rating_match = re.search(r"RATING:\s*(\d+)", result)
    if rating_match:
        score = float(rating_match.group(1))

    return {
        "fact_check_report": result,  # reuse field for checklist
        "fact_check_score": score / 10.0,
        "status": "improving",
        "log": _add_log(state, f"[RATE] Score: {score}/100"),
    }


# --- Node: Validate Numbers (deterministic, no LLM) ---

def validate_numbers_node(state: ArticleState) -> dict:
    """Deterministic fact validation: remove any sentence with numbers but no [FACT-N] tag.

    This is pure Python, no LLM call. Rules:
    - Any line with a number/spec MUST have a [FACT-N] tag
    - If it doesn't → the line is removed (hallucination)
    - Removed lines are logged and added to Rater's checklist for Improver
    """
    draft = state.get("draft", "")
    sources = state.get("sources", "")  # Research output with [FACT-N] list
    checklist = state.get("fact_check_report", "")

    from app.factory.quality_gate import validate_tagged_claims
    cleaned, removed = validate_tagged_claims(draft, sources)

    if removed:
        logger.info("[VALIDATE] Removed %d untagged lines with numbers", len(removed))
        # Add to checklist so Improver knows what was removed
        addition = (
            "\n\nREMOVED LINES (contained numbers without [FACT-N] source tag):\n"
            + "\n".join(f"- {r[:100]}" for r in removed)
            + "\n\nDo NOT re-add these claims. Only use facts from [FACT-N] tags."
        )
        checklist += addition
    else:
        logger.info("[VALIDATE] All numbered claims are tagged")

    return {
        "draft": cleaned,  # Replace draft with cleaned version
        "fact_check_report": checklist,
        "status": "improving",
        "log": _add_log(state, f"[VALIDATE] {len(removed)} untagged lines removed (deterministic)"),
    }


# --- Node: Improve ---

def improve_node(state: ArticleState) -> dict:
    """Improver takes the draft + checklist and produces an improved version.

    One job only: rewrite based on specific feedback.
    """
    draft = state.get("draft", "")
    checklist = state.get("fact_check_report", "")
    llm = _get_llm()

    agent = create_reviewer(llm)
    task = Task(
        description=(
            f"Here is an article:\n\n{draft}\n\n"
            f"Here is reviewer feedback:\n{checklist}\n\n"
            f"Improve the article based on the feedback above.\n"
            f"- Remove duplicate information across sections\n"
            f"- Integrate quotes into text naturally (no standalone 'quotes' sections)\n"
            f"- Remove sections that have no real content\n"
            f"- Keep all specific facts, numbers, and source links\n"
            f"- Output the full improved article in Markdown\n"
        ),
        expected_output=(
            "The full improved article in Markdown format. Nothing else."
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = str(crew.kickoff())

    return {
        "edited_text": result,
        "revision_count": state.get("revision_count", 0) + 1,
        "status": "final_rating",
        "log": _add_log(state, f"[IMPROVE] Article improved ({len(result)} chars)"),
    }


# --- Node: Final Rate ---

def final_rate_node(state: ArticleState) -> dict:
    """Final rater evaluates the improved article. If below threshold → back to Improver.

    Threshold is configurable. Default 60 (will be calibrated via benchmarks).
    """
    article = state.get("edited_text", state.get("draft", ""))
    llm = _get_llm()

    agent = create_reviewer(llm)
    task = Task(
        description=(
            f"Here is the final version of an article:\n\n{article}\n\n"
            f"Rate it from 0 to 100. Write one sentence explaining the rating."
            f"\n\nFormat:\nRATING: [number]\nEXPLANATION: [one sentence]"
        ),
        expected_output=(
            "RATING: [number]\nEXPLANATION: [one sentence]"
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = str(crew.kickoff())

    import re
    score = 75.0
    rating_match = re.search(r"RATING:\s*(\d+)", result)
    if rating_match:
        score = float(rating_match.group(1))

    revision_count = state.get("revision_count", 0)
    threshold = 60  # TODO: calibrate via benchmarks

    # Decide: accept or loop back
    if score >= threshold or revision_count >= 2:
        verdict = "accept"
    else:
        verdict = "revise"

    return {
        "fact_check_score": score / 10.0,
        "review_verdict": verdict,
        "status": "final_rated",
        "log": _add_log(state, f"[FINAL RATE] Score: {score}/100, verdict: {verdict} (revision {revision_count}/2)"),
    }


# --- Node: Publish ---

def publish_node(state: ArticleState) -> dict:
    """Finalize the article and save to DB if in factory mode."""
    article = state.get("edited_text", state.get("draft", ""))
    article_db_id = state.get("article_db_id")

    # Factory mode: update article in PostgreSQL
    if article_db_id:
        try:
            from app.factory.db import update_article
            from app.factory.quality_gate import check_level1, clean_artifacts, validate_facts_deterministic

            # Clean artifacts before saving
            cleaned = clean_artifacts(article)

            # Final deterministic validation — catch any numbers Improver re-introduced
            sources = state.get("sources", "")
            from app.factory.quality_gate import validate_tagged_claims
            cleaned, removed = validate_tagged_claims(cleaned, sources)
            if removed:
                logger.info("[PUBLISH] Final validator removed %d lines: %s",
                           len(removed), "; ".join(r[:60] for r in removed[:5]))

            # Run quality gate (min 3000 chars for factory articles)
            report = check_level1(cleaned, min_length=3000)
            status = "ready" if report.passed else "quality_check"

            update_article(
                article_db_id,
                content_ru=cleaned,
                title_ru=cleaned.split("\n")[0].lstrip("# ").strip()[:200],
                char_count=len(cleaned),
                fact_check_score=state.get("fact_check_score"),
                revision_count=state.get("revision_count", 0),
                status=status,
                generation_log=state.get("log", []),
            )
            logger.info("[PUBLISH] Article id=%d saved to DB (status=%s, %d chars)",
                        article_db_id, status, len(cleaned))

            if not report.passed:
                logger.warning("[PUBLISH] Quality gate issues: %s", "; ".join(report.issues))

        except Exception as e:
            logger.error("[PUBLISH] Failed to save to DB: %s", e, exc_info=True)

    return {
        "final_article": article,
        "status": "published",
        "log": _add_log(state, f"[PUBLISH] Article finalized ({len(article)} chars)"),
    }

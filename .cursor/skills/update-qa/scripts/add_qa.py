#!/usr/bin/env python3
"""Append Q&A entry to QA.md and refresh section counts."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
QA = ROOT / "QA.md"

SECTIONS = [
    (1, "1-diffusion--text-to-image", "Diffusion & Text-to-Image"),
    (2, "2-swiftedit--pipeline", "SwiftEdit & Pipeline"),
    (3, "3-inversion--noise", "Inversion & Noise"),
    (4, "4-mask--aram", "Mask & ARaM"),
    (5, "5-đánh-giá--piebench", "Đánh giá & PieBench"),
    (6, "6-triển-khai-mac--colab", "Triển khai Mac / Colab"),
    (7, "7-chỉnh-sửa--style", "Chỉnh sửa & Style"),
    (8, "8-chưa-phân-loại", "Chưa phân loại"),
]

SECTION_HEADER = re.compile(r"^## (\d+)\. .+$")
QUESTION_HEADING = re.compile(r"^### Q: ", re.M)
PLACEHOLDER = "*(Chưa có câu hỏi.)*"


def build_entry(question: str, answer: str, tags: str, notes: str, entry_date: str) -> str:
    q = question.strip()
    if not q.endswith("?"):
        q += "?"

    bullets = [ln.strip() for ln in answer.strip().split("\n") if ln.strip()]
    if bullets and not bullets[0].startswith("- "):
        bullets = [f"- {ln}" if not ln.startswith("-") else ln for ln in bullets]

    notes_block = ""
    if notes.strip():
        note_lines = [ln.strip() for ln in notes.strip().split("\n") if ln.strip()]
        notes_block = "\n**Ghi chú thêm / link:**\n" + "\n".join(
            ln if ln.startswith("-") else f"- {ln}" for ln in note_lines
        ) + "\n"

    return (
        f"### Q: {q}\n\n"
        f"**Ngày:** {entry_date}  \n"
        f"**Chủ đề:** {tags.strip()}\n\n"
        f"**Trả lời (tóm tắt):**\n"
        + "\n".join(bullets)
        + f"\n{notes_block}\n"
    )


def split_by_sections(text: str) -> dict[int, tuple[str, str]]:
    """Return section_num -> (header_line, body)."""
    lines = text.splitlines(keepends=True)
    sections: dict[int, tuple[str, str]] = {}
    current_num: int | None = None
    current_header = ""
    current_body: list[str] = []

    for line in lines:
        m = SECTION_HEADER.match(line.strip())
        if m:
            if current_num is not None:
                sections[current_num] = (current_header, "".join(current_body))
            current_num = int(m.group(1))
            current_header = line
            current_body = []
        elif current_num is not None:
            current_body.append(line)

    if current_num is not None:
        sections[current_num] = (current_header, "".join(current_body))

    return sections


def count_questions(body: str) -> int:
    return len(QUESTION_HEADING.findall(body))


def insert_entry(body: str, entry: str) -> str:
    body = body.replace(PLACEHOLDER + "\n", "")
    body = body.replace(PLACEHOLDER, "")

    marker = "<!-- qa:insert -->"
    if marker in body:
        return body.replace(marker, marker + "\n" + entry, 1)

    # After italic description block (first blank line after *...*)
    lines = body.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("*") and line.strip().endswith("*"):
            insert_at = i + 1
            break
    # skip blank lines after description
    while insert_at < len(lines) and lines[insert_at].strip() == "":
        insert_at += 1

    new_lines = lines[:insert_at] + [entry] + lines[insert_at:]
    return "".join(new_lines)


def update_toc(text: str, counts: dict[int, int]) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        m = re.match(r"^\| (\d+) \| \[([^\]]+)\]\((#[^)]+)\) \| (\d+) \|$", line.strip())
        if m:
            num = int(m.group(1))
            title = m.group(2)
            anchor = m.group(3).lstrip("#")
            count = counts.get(num, int(m.group(4)))
            out.append(f"| {num} | [{title}](#{anchor}) | {count} |\n")
        else:
            out.append(line)
    return "".join(out)


def question_exists(text: str, question: str) -> bool:
    q_norm = question.strip().rstrip("?").lower()
    for m in QUESTION_HEADING.finditer(text):
        existing = m.group(0)[6:].strip().rstrip("?").lower()
        if existing == q_norm:
            return True
    return False


def add_qa(
    question: str,
    answer: str,
    section: int = 8,
    tags: str = "#swiftedit",
    notes: str = "",
    entry_date: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> str:
    if not QA.exists():
        raise FileNotFoundError(f"QA.md not found: {QA}")

    if section not in range(1, 9):
        raise ValueError("section must be 1-8")

    text = QA.read_text(encoding="utf-8")
    entry_date = entry_date or date.today().isoformat()

    if question_exists(text, question) and not force:
        return f"Skipped: question already in QA.md — {question.strip()}"

    entry = build_entry(question, answer, tags, notes, entry_date)
    sections = split_by_sections(text)

    if section not in sections:
        raise ValueError(f"Section {section} not found in QA.md")

    header, body = sections[section]
    sections[section] = (header, insert_entry(body, entry))

    new_text = rebuild_file(text, sections)
    counts = {num: count_questions(body) for num, (_, body) in sections.items()}
    new_text = update_toc(new_text, counts)

    if not dry_run:
        QA.write_text(new_text, encoding="utf-8")

    return (
        f"Added to section {section}: {question.strip()}\n"
        f"Counts: {', '.join(f'{n}={counts[n]}' for n in sorted(counts))}\n"
    )


def rebuild_file(text: str, sections: dict[int, tuple[str, str]]) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = SECTION_HEADER.match(line.strip())
        if m:
            num = int(m.group(1))
            out.append(sections[num][0])
            out.append(sections[num][1])
            i += 1
            while i < len(lines):
                if SECTION_HEADER.match(lines[i].strip()):
                    break
                i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Q&A entry to QA.md")
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument("--answer", "-a", required=True, help="Multiline OK; use \\n for bullets")
    parser.add_argument("--section", "-s", type=int, default=8, choices=range(1, 9))
    parser.add_argument("--tags", "-t", default="#swiftedit")
    parser.add_argument("--notes", "-n", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Add even if duplicate question")
    args = parser.parse_args()

    answer = args.answer.replace("\\n", "\n")
    notes = args.notes.replace("\\n", "\n")

    report = add_qa(
        question=args.question,
        answer=answer,
        section=args.section,
        tags=args.tags,
        notes=notes,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(report)
    if args.dry_run:
        print("(dry-run — QA.md not modified)")


if __name__ == "__main__":
    main()

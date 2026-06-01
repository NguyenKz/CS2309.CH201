#!/usr/bin/env python3
"""Sync README.md progress, work journal (NHAT_KY.md), and de-tai report table."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
README = ROOT / "README.md"
JOURNAL = ROOT / "NHAT_KY.md"
DETAI = ROOT / "SwiftEdit_DeTai_CS2309.md"

PHASES = [
    (1, "1. Lý thuyết", re.compile(r"^## Giai đoạn 1\b", re.M)),
    (2, "2. Setup Mac + Colab", re.compile(r"^## Giai đoạn 2\b", re.M)),
    (3, "3. Thực nghiệm cơ bản", re.compile(r"^## Giai đoạn 3\b", re.M)),
    (4, "4. So sánh & mở rộng", re.compile(r"^## Giai đoạn 4\b", re.M)),
    (5, "5. Báo cáo & nộp", re.compile(r"^## Giai đoạn 5\b", re.M)),
]

CHECKBOX = re.compile(r"^- \[([ xX])\] (.+)$", re.M)
LAST_UPDATED = re.compile(r"^> Cập nhật tiến độ lần cuối: .+$", re.M)
PLACEHOLDER_ROW = re.compile(r"^\| \*?\(chưa có entry.*\|.*\|$", re.M)
DETAI_PLACEHOLDER = re.compile(r"^\| \| Giai đoạn \d.*\|$", re.M)

JOURNAL_SUMMARY_HEADER = "## Tóm tắt nhanh"
JOURNAL_DETAIL_HEADER = "## Chi tiết theo phiên làm việc"
DETAI_TABLE_HEADER = "### 8.1. Nhật ký thực nghiệm"


def split_sections(text: str) -> dict[int, str]:
    phase_starts: list[tuple[int, int]] = []
    for phase_id, _, pattern in PHASES:
        match = pattern.search(text)
        if match:
            phase_starts.append((match.start(), phase_id))
    phase_starts.sort()

    h2_markers = [(m.start(), m.group()) for m in re.finditer(r"^## .+$", text, re.M)]

    sections: dict[int, str] = {}
    for i, (start, phase_id) in enumerate(phase_starts):
        if i + 1 < len(phase_starts):
            end = phase_starts[i + 1][0]
        else:
            end = len(text)
            for h2_start, h2_title in h2_markers:
                if h2_start > start and not h2_title.startswith("## Giai đoạn"):
                    end = h2_start
                    break
        sections[phase_id] = text[start:end]
    return sections


def count_checkboxes(section: str) -> tuple[int, int]:
    done = total = 0
    skip = False
    for line in section.splitlines():
        if line.startswith("### "):
            skip = "tùy chọn" in line.lower()
            continue
        match = CHECKBOX.match(line)
        if not match or skip:
            continue
        total += 1
        if match.group(1).lower() == "x":
            done += 1
    return done, total


def phase_status(done: int, total: int) -> str:
    if total == 0 or done == 0:
        return "⬜ Chưa bắt đầu"
    if done >= total:
        return "✅ Hoàn thành"
    return f"🔄 Đang làm ({done}/{total})"


def build_status_table(rows: list[tuple[str, str]]) -> str:
    lines = [
        "### Trạng thái nhanh",
        "",
        "| Giai đoạn | Tiến độ |",
        "|---|---|",
    ]
    for label, status in rows:
        lines.append(f"| {label} | {status} |")
    lines.append("")
    return "\n".join(lines)


def sync_readme(dry_run: bool = False) -> str:
    if not README.exists():
        raise FileNotFoundError(f"README not found: {README}")

    text = README.read_text(encoding="utf-8")
    sections = split_sections(text)

    rows: list[tuple[str, str]] = []
    summary_lines = ["", "## Tiến độ (tự động)", ""]
    total_done = total_all = 0

    phase_defs = [
        (1, "1. Lý thuyết"),
        (2, "2. Setup Mac + Colab"),
        (3, "3. Thực nghiệm cơ bản"),
        (4, "4. So sánh & mở rộng"),
        (5, "5. Báo cáo & nộp"),
    ]
    for phase_id, label in phase_defs:
        section = sections.get(phase_id, "")
        done, total = count_checkboxes(section)
        status = phase_status(done, total)
        rows.append((label, status))
        total_done += done
        total_all += total
        summary_lines.append(f"- **{label}:** {done}/{total} — {status}")

    new_table = build_status_table(rows)
    table_pattern = re.compile(
        r"### Trạng thái nhanh\n\n\| Giai đoạn \| Tiến độ \|\n\|---\|---\|\n(?:\| .+ \| .+ \|\n)+",
        re.M,
    )
    if not table_pattern.search(text):
        raise ValueError("Could not find 'Trạng thái nhanh' table in README.md")

    updated = table_pattern.sub(new_table + "\n", text, count=1)
    stamp = f"> Cập nhật tiến độ lần cuối: {date.today().isoformat()} — {total_done}/{total_all} task bắt buộc"
    if LAST_UPDATED.search(updated):
        updated = LAST_UPDATED.sub(stamp, updated, count=1)
    else:
        updated = updated.replace(
            "> Đánh dấu `[x]` khi hoàn thành. Cập nhật file này trong quá trình làm đề tài.\n",
            f"> Đánh dấu `[x]` khi hoàn thành. Cập nhật file này trong quá trình làm đề tài.\n{stamp}\n",
        )

    report = "\n".join(summary_lines) + f"\n\n**Tổng:** {total_done}/{total_all} task bắt buộc\n"
    if not dry_run:
        README.write_text(updated, encoding="utf-8")
    return report


def mark_tasks(task_labels: list[str], dry_run: bool = False) -> list[str]:
    if not README.exists():
        raise FileNotFoundError(f"README not found: {README}")

    text = README.read_text(encoding="utf-8")
    marked: list[str] = []

    for label in task_labels:
        pattern = re.compile(rf"^- \[ \] (.*{re.escape(label)}.*)$", re.M | re.I)

        def repl(match: re.Match[str]) -> str:
            marked.append(match.group(1).strip())
            return f"- [x] {match.group(1)}"

        text, n = pattern.subn(repl, text, count=1)
        if n == 0:
            marked.append(f"(NOT FOUND: {label})")

    if not dry_run and any(not m.startswith("(NOT FOUND") for m in marked):
        README.write_text(text, encoding="utf-8")
    return marked


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _build_detail_block(
    entry_date: str,
    phase: str,
    title: str,
    env: str,
    work: str,
    result: str,
    tasks_done: list[str],
    issues: str,
    next_steps: str,
) -> str:
    tasks_block = "\n".join(f"- {t}" for t in tasks_done) if tasks_done else "- *(không đánh task README)*"
    issues_block = f"\n**Vấn đề / cách xử lý:**\n{issues}\n" if issues.strip() else ""
    next_block = ""
    if next_steps.strip():
        next_lines = "\n".join(f"- {s.strip()}" for s in next_steps.split("\n") if s.strip())
        next_block = f"\n**Bước tiếp theo:**\n{next_lines}\n"

    work_lines = "\n".join(f"- {line.strip()}" for line in work.split("\n") if line.strip())
    result_lines = "\n".join(f"- {line.strip()}" for line in result.split("\n") if line.strip())

    return (
        f"### {entry_date} — [{phase}] {title}\n\n"
        f"**Môi trường:** {env}\n\n"
        f"**Công việc đã làm:**\n{work_lines}\n\n"
        f"**Kết quả:**\n{result_lines}\n\n"
        f"**Task README đã đánh [x]:**\n{tasks_block}\n"
        f"{issues_block}{next_block}\n---\n\n"
    )


def append_journal(
    phase: str,
    work: str,
    result: str,
    env: str = "Mac M4 (MPS)",
    title: str = "",
    tasks_done: list[str] | None = None,
    issues: str = "",
    next_steps: str = "",
    entry_date: str | None = None,
    dry_run: bool = False,
) -> str:
    if not JOURNAL.exists():
        raise FileNotFoundError(f"Journal not found: {JOURNAL}")

    entry_date = entry_date or date.today().isoformat()
    title = title or work.split("\n")[0][:60]
    tasks_done = tasks_done or []

    summary_row = (
        f"| {entry_date} | {phase} | {_escape_cell(work.split(chr(10))[0][:80])} "
        f"| {_escape_cell(result.split(chr(10))[0][:120])} | {env} |"
    )
    detail = _build_detail_block(
        entry_date, phase, title, env, work, result, tasks_done, issues, next_steps
    )

    text = JOURNAL.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Insert summary row after table header separator
    inserted = False
    new_lines: list[str] = []
    i = 0
    while i < len(lines):
        new_lines.append(lines[i])
        if (
            not inserted
            and lines[i].strip() == "|---|---|---|---|---|"
            and i > 0
            and "Ngày" in lines[i - 1]
            and "Môi trường" in lines[i - 1]
        ):
            new_lines.append(summary_row + "\n")
            inserted = True
            i += 1
            # Skip placeholder row if present
            if i < len(lines) and PLACEHOLDER_ROW.match(lines[i].strip()):
                i += 1
            continue
        i += 1

    if not inserted:
        raise ValueError("Could not find summary table in NHAT_KY.md")

    text = "".join(new_lines)

    # Prepend detail block after detail section intro
    marker = "*(Các entry chi tiết xuất hiện bên dưới, mới nhất ở trên cùng.)*\n\n"
    if marker in text:
        text = text.replace(marker, marker + detail, 1)
    else:
        text = text.rstrip() + "\n\n" + detail

    if not dry_run:
        JOURNAL.write_text(text, encoding="utf-8")
        sync_journal_to_detai(dry_run=False)

    return f"Journal entry added: {entry_date} — {phase}\n{summary_row}\n"


def sync_journal_to_detai(dry_run: bool = False) -> int:
    """Copy summary rows from NHAT_KY.md to SwiftEdit_DeTai_CS2309.md §8.1."""
    if not JOURNAL.exists() or not DETAI.exists():
        return 0

    journal = JOURNAL.read_text(encoding="utf-8")
    detai_lines = DETAI.read_text(encoding="utf-8").splitlines(keepends=True)

    rows: list[str] = []
    in_summary = False
    for line in journal.splitlines():
        if line.strip() == JOURNAL_SUMMARY_HEADER:
            in_summary = True
            continue
        if in_summary and (line.startswith("## ") or line.strip() == "---"):
            if rows:
                break
            continue
        if in_summary and line.startswith("|") and not line.startswith("|---") and "Ngày" not in line:
            if PLACEHOLDER_ROW.match(line.strip()):
                continue
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 5:
                note = parts[3]
                if parts[4]:
                    note = f"{note} ({parts[4]})"
                rows.append(f"| {parts[0]} | {parts[1]} | {parts[2]} | {note} |\n")
            elif len(parts) == 4:
                rows.append(f"| {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} |\n")

    if not rows:
        return 0

    out: list[str] = []
    i = 0
    replaced = False
    while i < len(detai_lines):
        line = detai_lines[i]
        out.append(line)
        if (
            not replaced
            and line.strip() == "### 8.1. Nhật ký thực nghiệm"
        ):
            # skip blank line already appended; expect header + separator next
            i += 1
            if i < len(detai_lines) and detai_lines[i].strip() == "":
                out.append(detai_lines[i])
                i += 1
            if i < len(detai_lines) and "Ngày" in detai_lines[i]:
                out.append(detai_lines[i])
                i += 1
            if i < len(detai_lines) and detai_lines[i].startswith("|---"):
                out.append(detai_lines[i])
                i += 1
            # skip old data rows until next ###
            while i < len(detai_lines) and detai_lines[i].startswith("|"):
                i += 1
            out.extend(rows)
            replaced = True
            continue
        i += 1

    if not replaced:
        raise ValueError("Could not find §8.1 journal table in SwiftEdit_DeTai_CS2309.md")

    if not dry_run:
        DETAI.write_text("".join(out), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync README progress and work journal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mark", nargs="+", metavar="SUBSTRING")
    parser.add_argument("--journal-phase", metavar="PHASE", help='e.g. "2a. Mac"')
    parser.add_argument("--journal-work", metavar="TEXT", help="Công việc đã làm")
    parser.add_argument("--journal-result", metavar="TEXT", help="Kết quả / ghi chú")
    parser.add_argument("--journal-env", default="Mac M4 (MPS)", metavar="ENV")
    parser.add_argument("--journal-title", default="", metavar="TITLE")
    parser.add_argument("--journal-issues", default="", metavar="TEXT")
    parser.add_argument("--journal-next", default="", metavar="TEXT")
    parser.add_argument(
        "--journal-sync-detai-only",
        action="store_true",
        help="Only sync NHAT_KY summary → de-tai §8.1",
    )
    args = parser.parse_args()

    if args.journal_sync_detai_only:
        n = sync_journal_to_detai(dry_run=args.dry_run)
        print(f"Synced {n} rows to SwiftEdit_DeTai_CS2309.md §8.1")
        return

    marked: list[str] = []
    if args.mark:
        marked = mark_tasks(args.mark, dry_run=args.dry_run)
        print("Marked:")
        for item in marked:
            print(f"  - {item}")

    if args.journal_work and args.journal_result:
        tasks_done = [m for m in marked if not m.startswith("(NOT FOUND")]
        print(
            append_journal(
                phase=args.journal_phase or "—",
                work=args.journal_work,
                result=args.journal_result,
                env=args.journal_env,
                title=args.journal_title,
                tasks_done=tasks_done,
                issues=args.journal_issues,
                next_steps=args.journal_next,
                dry_run=args.dry_run,
            )
        )
    elif args.journal_phase or args.journal_work:
        parser.error("--journal-work and --journal-result required with --journal-phase")

    report = sync_readme(dry_run=args.dry_run)
    print(report)
    if args.dry_run:
        print("(dry-run — files not modified)")


if __name__ == "__main__":
    main()

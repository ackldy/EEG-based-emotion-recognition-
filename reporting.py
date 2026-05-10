from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union


HeaderSpec = Union[str, Tuple[str, str]]


def _bar(value: float, width: int = 24) -> str:
    """把 0 到 1 之间的数值渲染成紧凑的 ASCII 条形图。"""
    clipped = max(0.0, min(1.0, float(value)))
    filled = int(round(clipped * width))
    return "#" * filled + "-" * (width - filled)


def format_metric_table(
    rows: Sequence[Dict[str, Any]],
    headers: Sequence[HeaderSpec],
) -> List[str]:
    """按给定表头顺序，把字典列表渲染成 Markdown 表格。"""
    resolved_headers: List[Tuple[str, str]] = []
    for header in headers:
        if isinstance(header, tuple):
            resolved_headers.append((str(header[0]), str(header[1])))
        else:
            resolved_headers.append((str(header), str(header)))

    lines = [
        "| " + " | ".join(display_name for _key, display_name in resolved_headers) + " |",
        "| " + " | ".join(["---"] * len(resolved_headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _display_name in resolved_headers) + " |")
    return lines


def format_metric_bars(title: str, metric_rows: Sequence[Dict[str, Any]], metric_key: str) -> List[str]:
    """为单个指标生成便于快速比较的文本条形图。"""
    lines = [f"### {title}", "", "```text"]
    for row in metric_rows:
        label = str(row["Classifier"]).ljust(8)
        value = float(row[metric_key])
        lines.append(f"{label} {value:0.4f} {_bar(value)}")
    lines.append("```")
    return lines


def write_markdown_report(
    path: Path,
    title: str,
    intro_lines: Iterable[str],
    sections: Sequence[Dict[str, Any]],
) -> Path:
    """把结构化报告内容写入 Markdown 文件，并返回输出路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [f"# {title}", ""]
    lines.extend(intro_lines)
    lines.append("")

    for section in sections:
        section_title = str(section["title"])
        lines.append(f"## {section_title}")
        lines.append("")

        body_lines = section.get("body_lines", [])
        lines.extend(body_lines)
        if body_lines:
            lines.append("")

        table = section.get("table")
        if table:
            lines.extend(table)
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path

"""Dependency-free SVG plots for the sanitized competition leaderboard."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Callable


COLORS = ("#2563eb", "#16a34a", "#dc2626", "#9333ea")


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _category_pass3(category: str) -> Callable[[dict[str, Any]], Any]:
    return lambda row: row["pass_scores"]["by_category"][category]["Pass^3"]


def write_grouped_bar_chart(
    rows: list[dict[str, Any]],
    *,
    series: list[tuple[str, Callable[[dict[str, Any]], Any]]],
    title: str,
    output: Path,
    maximum: float | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    width = max(760, 120 + len(rows) * max(70, len(series) * 34))
    height = 520
    left, top, bottom, right = 75, 70, 125, 30
    chart_width = width - left - right
    chart_height = height - top - bottom
    values = [_number(getter(row)) for _, getter in series for row in rows]
    scale_max = maximum if maximum is not None else max(values, default=1.0)
    scale_max = max(scale_max, 1e-9)
    group_width = chart_width / max(1, len(rows))
    bar_width = min(30, group_width * 0.75 / max(1, len(series)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="600">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        value = scale_max * tick / 5
        y = top + chart_height - chart_height * tick / 5
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.2g}</text>'
        )

    for row_index, row in enumerate(rows):
        center = left + group_width * (row_index + 0.5)
        total_width = bar_width * len(series)
        for series_index, (label, getter) in enumerate(series):
            value = _number(getter(row))
            bar_height = chart_height * value / scale_max
            x = center - total_width / 2 + series_index * bar_width
            y = top + chart_height - bar_height
            color = COLORS[series_index % len(COLORS)]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(2, bar_width-2):.1f}" height="{bar_height:.1f}" fill="{color}"><title>{html.escape(label)}: {value:.6g}</title></rect>'
            )
        team = html.escape(str(row.get("team_name") or row.get("team_id")))
        parts.append(
            f'<text x="{center:.1f}" y="{top+chart_height+18}" text-anchor="end" transform="rotate(-35 {center:.1f} {top+chart_height+18})" font-family="sans-serif" font-size="11">{team}</text>'
        )

    legend_x = left
    legend_y = height - 20
    for index, (label, _) in enumerate(series):
        x = legend_x + index * 180
        parts.append(f'<rect x="{x}" y="{legend_y-11}" width="12" height="12" fill="{COLORS[index % len(COLORS)]}"/>')
        parts.append(
            f'<text x="{x+18}" y="{legend_y}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_standard_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    write_grouped_bar_chart(
        rows,
        series=[("Macro Pass^3", lambda row: row["primary_score"])],
        title="CAR-bench Hidden Evaluation — Macro Pass^3",
        output=output_dir / "macro-pass3.svg",
        maximum=1.0,
    )
    write_grouped_bar_chart(
        rows,
        series=[
            (category.capitalize(), _category_pass3(category))
            for category in ("base", "hallucination", "disambiguation")
        ],
        title="Pass^3 by Task Category",
        output=output_dir / "category-pass3.svg",
        maximum=1.0,
    )
    write_grouped_bar_chart(
        rows,
        series=[
            ("Pass@3 − Pass^3", lambda row: row["pass_scores"]["macro"]["Pass@3"] - row["pass_scores"]["macro"]["Pass^3"])
        ],
        title="Consistency Gap",
        output=output_dir / "consistency-gap.svg",
        maximum=1.0,
    )
    write_grouped_bar_chart(
        rows,
        series=[
            ("Mean A2A latency (s)", lambda row: _number(row["latency"]["a2a_raw_per_task_trial_ms"]["mean"]) / 1000)
        ],
        title="Evaluator-Measured A2A Latency",
        output=output_dir / "latency.svg",
    )
    write_grouped_bar_chart(
        rows,
        series=[
            ("Mean self-reported tokens", lambda row: row["tokens"]["metrics"]["total_tokens"]["mean"]),
            ("Reporting coverage × 500k", lambda row: _number(row["tokens"]["coverage"]) * 500_000),
        ],
        title="Participant-Reported Token Usage and Coverage",
        output=output_dir / "tokens.svg",
    )
    write_grouped_bar_chart(
        rows,
        series=[
            ("Retries", lambda row: row.get("recovery", {}).get("retry_count", 0)),
            ("Timeouts", lambda row: row.get("recovery", {}).get("timeout_count", 0)),
            (
                "Other errors",
                lambda row: row.get("recovery", {}).get(
                    "infrastructure_error_count", 0
                )
                - row.get("recovery", {}).get("timeout_count", 0),
            ),
        ],
        title="Recovery, Timeout, and Error Counts",
        output=output_dir / "retries.svg",
    )

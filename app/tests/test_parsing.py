"""
Оценка качества парсинга PDF-учебника.

Сравниваем:
  - Мой парсер  (файл с уже приведённым к plain-text результатом)
  - PyMuPDF
  - pdfplumber

Метрики: CER (Character Error Rate), WER (Word Error Rate), Coverage
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich import box
from tqdm import tqdm


REFERENCE_TEXT_FILE = Path("data/parser_test_data/reference.txt")   # эталонный plain-text
MY_PARSER_TEXT_FILE = Path("data/parser_test_data/my_parser.txt")   # результат моего парсера (plain-text)
PYMUPDF_TEXT_FILE = Path("data/parser_test_data/pymupdf.txt")  # результат pymupdf
MARKER_TEXT_FILE = Path("data/parser_test_data/marker.txt") # результат маркера
PDF_FILE = Path("data/parser_test_data/textbook.pdf")    # оригинальный PDF для сторонних парсеров

RESULTS_DIR  = Path("data/results")
METRICS_JSON = RESULTS_DIR / "parsing_metrics.json"
METRICS_PDF = RESULTS_DIR / "parsing_metrics.pdf"

console = Console()

# Нормализация
def normalize(text: str) -> str:
    """
    Базовая нормализация: нижний регистр, схлопывание пробелов.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_text(path: Path) -> str:
    return normalize(path.read_text(encoding="utf-8"))


# Метрики
def _edit_distance(a: list, b: list) -> int:
    """
    Levenshtein distance.
    """
    if len(a) < len(b):
        a, b = b, a
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in tqdm(range(1, n + 1)):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate = edit_distance(chars) / len(reference).
    """
    ref = list(reference)
    hyp = list(hypothesis)
    return min(_edit_distance(ref, hyp) / max(len(ref), 1), 1.0)


def wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate = edit_distance(words) / len(reference_words).
    """
    ref = reference.split()
    hyp = hypothesis.split()
    return min(_edit_distance(ref, hyp) / max(len(ref), 1), 1.0)


def text_coverage(reference: str, hypothesis: str) -> float:
    """
    Доля уникальных слов эталона, найденных в гипотезе.
    """
    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())
    if not ref_words:
        return 0.0
    return len(ref_words & hyp_words) / len(ref_words)


def compute_metrics(reference: str, hypothesis: str) -> dict:
    """
    Считает метрики.
    """
    c   = cer(reference, hypothesis)
    w   = wer(reference, hypothesis)
    cov = text_coverage(reference, hypothesis)
    return {
        "CER":        c,
        "WER":        w,
        "Coverage":   cov,
        "Accuracy_%": round((1 - c) * 100, 2),
    }


# Сохранение результатов
def save_results(results: dict) -> None:
    """
    Сохраняет метрики и результаты парсинга
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Сохранение метрик
    json_data = {
        name: {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in metrics.items()
        }
        for name, metrics in results.items()
    }
    METRICS_JSON.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f'Результаты сохранены в папку {RESULTS_DIR}')


# Визуализация
BG_COLOR    = "#FFFFFF"
TEXT_COLOR  = "#2E2E2E"
GRID_COLOR  = "#E0E0E0"
BORDER_COLOR = "#BCBCBC"

PARSER_COLORS = {
    "Мой парсер": "#2A4D69",
    "PyMuPDF":    "#5B7065",
    "Marker": "#A3B18A",
}


def plot_results(results: dict[str, dict]) -> None:
    parsers       = list(results.keys())
    metrics       = ["CER", "WER", "Coverage"]
    metric_labels = ["CER (↓ лучше)", "WER (↓ лучше)", "Coverage (↑ лучше)"]
    colors        = [PARSER_COLORS.get(p, "#AAAAAA") for p in parsers]
    plt.rcParams["font.family"] = "serif"

    n = len(parsers)
    x = np.arange(len(metrics))
    width = 0.22
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    fig.patch.set_facecolor(BG_COLOR)

    ax = axes[0]
    ax.set_facecolor(BG_COLOR)

    for i, (parser, color) in enumerate(zip(parsers, colors)):
        vals = [results[parser][m] for m in metrics]
        bars = ax.bar(
            x + offsets[i], vals, width, label=parser,
            color=color, alpha=0.88, edgecolor="#444444", linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=9, color=TEXT_COLOR, fontweight="medium",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=TEXT_COLOR, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Значение метрики", color=TEXT_COLOR, fontsize=11)
    ax.set_title("Сравнение парсеров PDF", color=TEXT_COLOR,
                 fontsize=14, fontweight="bold", pad=20)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines[:].set_color(BORDER_COLOR)
    ax.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, edgecolor=BG_COLOR, fontsize=10)
    ax.grid(axis="y", color=GRID_COLOR, linestyle='--', linewidth=0.7)

    # ── Таблица справа ──
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    ax2.axis("off")

    col_labels = ["Парсер", "CER", "WER", "Coverage", "Точность"]
    table_data = [
        [
            p,
            f"{results[p]['CER']:.4f}",
            f"{results[p]['WER']:.4f}",
            f"{results[p]['Coverage']:.4f}",
            f"{results[p]['Accuracy_%']:.1f}%",
        ]
        for p in parsers
    ]

    tbl = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 2.5)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER_COLOR)
        if row == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(color=TEXT_COLOR, fontweight="bold")
        else:
            cell.set_facecolor(BG_COLOR)
            cell.set_text_props(color=TEXT_COLOR)

    # Подсветка лучшего значения в каждом числовом столбце
    best_highlight = "#DDEEDD"
    best_text = "#1E3A28"

    best_map = {
        "CER":      (min, 1),
        "WER":      (min, 2),
        "Coverage": (max, 3),
    }
    for metric, (fn, col_idx) in best_map.items():
        vals = [results[p][metric] for p in parsers]
        best_row = vals.index(fn(vals)) + 1
        tbl[best_row, col_idx].set_facecolor(best_highlight)
        tbl[best_row, col_idx].set_text_props(color=best_text, fontweight="bold")

    ax2.set_title("Сводная таблица", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout(pad=3.0)
    METRICS_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(METRICS_PDF, dpi=150, bbox_inches="tight")
    plt.show()


# Главная функция
def run() -> None:
    console.print(Panel("Тест модуля парсинга PDF", expand=False))

    # Загружаем готовые тексты
    print("Загрузка файлов")
    reference = load_text(REFERENCE_TEXT_FILE)
    my_parser = load_text(MY_PARSER_TEXT_FILE)
    pymupdf_text = load_text(PYMUPDF_TEXT_FILE)
    marker_text = load_text(MARKER_TEXT_FILE)
    print(f"  Эталон:       {len(reference)} символов")
    print(f"  Мой парсер:   {len(my_parser)} символов")
    print(f"  Pymupdf:      {len(pymupdf_text)} символов")
    print(f"  Marker:       {len(marker_text)} символов")

    # Подсчет метрик
    print("Подсчёт метрик")
    candidates: dict[str, str] = {
        "Мой парсер": my_parser,
        "PyMuPDF": pymupdf_text,
        "Marker": marker_text,
    }

    results:   dict[str, dict]  = {}

    for name, norm_text in candidates.items():
        print(f"\nПодсчет метрик для [{name}]")
        results[name] = compute_metrics(reference, norm_text)

    # Вывод таблицы в консоль
    table = Table(
        title="Метрики парсинга", box=box.ROUNDED,
        header_style="bold magenta", show_lines=True,
    )
    table.add_column("Парсер",     style="cyan",  min_width=14)
    table.add_column("CER ↓",      style="white", justify="right")
    table.add_column("WER ↓",      style="white", justify="right")
    table.add_column("Coverage ↑", style="white", justify="right")
    table.add_column("Точность %", style="green", justify="right")

    best_cer      = min(results[p]["CER"]      for p in results)
    best_wer      = min(results[p]["WER"]      for p in results)
    best_coverage = max(results[p]["Coverage"] for p in results)

    for name, m in results.items():
        def fmt(val, best):
            is_best = val == best
            s = f"{val:.4f}"
            return f"[bold green]{s}[/bold green]" if is_best else s

        table.add_row(
            name,
            fmt(m["CER"], best_cer),
            fmt(m["WER"], best_wer),
            fmt(m["Coverage"], best_coverage),
            f"{m['Accuracy_%']:.1f}%",
        )

    print()
    console.print(table)

    # Сохранение результатов
    save_results(results)

    # Вывод графика
    plot_results(results)

    print("Тест парсинга завершен")


if __name__ == "__main__":
    run()
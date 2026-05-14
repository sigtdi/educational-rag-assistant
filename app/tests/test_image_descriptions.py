"""
test_image_descriptions.py
Оценка качества описаний изображений, сгенерированных моим парсером.

Структура папки с данными:
    data/images/
        figure_1.png
        figure_1.txt - описание от моего парсера
        ...

Судья: Gemini 3 Flash
Критерии оценки (1–10):
    accuracy — описание соответствует содержимому изображения
    completeness — все важные элементы упомянуты
    relevance — описание полезно в контексте учебника по алгоритмам
    clarity — текст понятен и структурирован
"""

import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box
from google import genai
from google.genai import types

load_dotenv()

IMAGES_DIR     = Path("data/images")
client = genai.Client()
REQUEST_DELAY  = 15.0

RESULTS_DIR      = Path("data/results")
JSON_OUT         = RESULTS_DIR / "image_desc_results.json"
PDF_PER_IMAGE    = RESULTS_DIR / "image_desc_per_image.pdf"
PDF_BY_TYPE      = RESULTS_DIR / "image_desc_by_type.pdf"
PDF_RADAR        = RESULTS_DIR / "image_desc_radar.pdf"

CRITERIA    = ["accuracy", "completeness", "relevance", "clarity"]
CRITERIA_RU = ["Точность", "Полнота", "Релевантность", "Ясность"]

# Типы изображений и их отображаемые названия
IMAGE_TYPES: dict[str, str] = {
    "stepwise":   "Пошаговое выполнение",
    "scheme":     "Простые схемы",
    "table":      "Таблицы и псевдокоды",
    "graph_tree": "Графы и деревья",
}


# Разметка изображений по типам
IMAGE_TYPE_MAP: dict[str, str] = {
    # Пошаговое выполнение
    "1aa8a335-4417-4183-9b40-3d7680a89362": "stepwise",
    "c4e9244f-211d-432e-ad25-7b31b804f220": "stepwise",
    "ca8128a8-9be1-48f7-86fc-ffaed78ce3f6":  "stepwise",
    # Простые схемы
    "4cb8c4de-fdb9-452d-9254-170b8d10b160": "scheme",
    "938f9136-8776-4dbc-ad24-9e7d7e143df0": "scheme",
    "da51ce3c-0fa8-47f0-ba56-2ea9bac9fc6b": "scheme",
    # Таблицы и псевдокоды
    "706126cf-de3c-4f55-89bb-401e272c9b73": "table",
    "adc7ac10-b2ca-4fb4-9685-ce71d4285ccb": "table",
    "c048ec14-cb8d-42e9-8b9c-d09628625b48": "table",
    # Графы и деревья
    "4faa1c81-d2b7-4bb5-a5bb-a90632fd3770": "graph_tree",
    "734d47d4-e342-4b3b-bcb6-0821dccc9c94": "graph_tree",
    "fecc4c77-8845-440b-95f3-dcc9d601c8ef": "graph_tree",
}

console = Console()


# Загрузка изображений и описаний
def load_pairs(images_dir: Path) -> list[dict]:
    """
    Собор изображений и соответствующих им описаний.
    """
    pairs = []
    for img_path in sorted(images_dir.glob("*.png")):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"Не найдено описание для {img_path.name}")
            continue

        img_type = IMAGE_TYPE_MAP.get(img_path.stem)
        if img_type is None:
            print(f"Не найден тип изображения для {img_path.name}")
            continue

        description = txt_path.read_text(encoding="utf-8").strip()
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        pairs.append({
            "name":        img_path.stem,
            "type":        img_type,
            "image_bytes":   image_bytes,
            "description": description,
        })
    return pairs


# Работа с Gemini
JUDGE_PROMPT = """
Ты — эксперт по оценке качества автоматических описаний изображений из учебников по алгоритмам и структурам данных.

Тебе дадут изображение и текстовое описание, сгенерированное автоматической системой.
Оцени описание по четырём критериям от 1 до 10:

- accuracy     (точность):      насколько описание соответствует содержимому изображения
- completeness (полнота):       все ли важные элементы изображения упомянуты
- relevance    (релевантность): насколько описание полезно в контексте учебника по алгоритмам
- clarity      (ясность):       насколько описание понятно и хорошо структурировано

Верни ТОЛЬКО JSON без пояснений:
{
  "accuracy": <int 1-10>,
  "completeness": <int 1-10>,
  "relevance": <int 1-10>,
  "clarity": <int 1-10>,
  "comment": "<одно предложение — главный недостаток или достоинство>"
}
""".strip()


def evaluate_with_gemini(image_bytes: bytes, description: str) -> dict | None:
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                ),
                f"Описание от системы:\n\n{description}"
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                system_instruction=JUDGE_PROMPT
            )
        )

        text = response.text
        match = re.search(r'```json\s+(.*?)\s+```', text, re.DOTALL)
        if match:
            clean_json = match.group(1)
        else:
            clean_json = text.strip()

        return json.loads(clean_json)
    except Exception as e:
        print(f"Ошибка при работе с Gemini: {e}")
        return None


# Агрегация по типам
def aggregate_by_type(eval_results: list[dict]) -> dict[str, dict[str, float]]:
    """
    Возвращает средние оценки по каждому типу.
    """
    buckets: dict[str, list[dict]] = {t: [] for t in IMAGE_TYPES}
    for r in eval_results:
        t = r.get("type")
        if t in buckets:
            buckets[t].append(r["scores"])

    result = {}
    for t, items in buckets.items():
        if not items:
            result[t] = {c: 0.0 for c in CRITERIA}
        else:
            result[t] = {
                c: float(np.mean([item[c] for item in items]))
                for c in CRITERIA
            }
    return result


# Сохранение
def save_json(eval_results: list[dict], by_type: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = {
        "per_image": [
            {
                "name":    r["name"],
                "type":    r["type"],
                "scores":  {k: round(v, 4) for k, v in r["scores"].items()},
                "average": round(float(np.mean(list(r["scores"].values()))), 4),
                "comment": r["comment"],
            }
            for r in eval_results
        ],
        "by_type": {
            t: {k: round(v, 4) for k, v in scores.items()}
            for t, scores in by_type.items()
        },
        "overall": {
            c: round(float(np.mean([r["scores"][c] for r in eval_results])), 4)
            for c in CRITERIA
        },
    }
    JSON_OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Резултаты сохранены в папку {RESULTS_DIR}")


# Визуализация
BG_COLOR     = "#FFFFFF"
TEXT_COLOR   = "#2E2E2E"
GRID_COLOR   = "#E0E0E0"
BORDER_COLOR = "#BCBCBC"

CRITERIA_COLORS = [
    "#1D3A4D",
    "#436D7A",
    "#5B7065",
    "#9FB0A2",
]

TYPE_COLORS = {
    "stepwise":   "#224254",
    "scheme":     "#537785",
    "table":      "#8A9D91",
    "graph_tree": "#435C4E",
}


def plot_per_image(eval_results: list[dict]) -> None:
    """
    Grouped bar chart: каждая группа — одно изображение, бары — критерии.
    """
    sorted_results = sorted(eval_results, key=lambda r: list(IMAGE_TYPES).index(r["type"]))
    plt.rcParams["font.family"] = "serif"

    names   = [r["name"] for r in sorted_results]
    types   = [r["type"] for r in sorted_results]
    n       = len(names)
    x       = np.arange(n)
    width   = 0.18
    offsets = np.linspace(-1.5, 1.5, len(CRITERIA)) * width

    fig, ax = plt.subplots(figsize=(max(12, int(n * 2.4)), 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Фоновая подсветка по типу
    prev_type, group_start = types[0], 0
    for i, t in enumerate(types + [None]):
        if t != prev_type:
            color = TYPE_COLORS.get(prev_type, "#EEEEEE")
            ax.axvspan(group_start - 0.5, i - 0.5,
                       color=color, alpha=0.04, zorder=0)

            mid = (group_start + i - 1) / 2
            ax.text(mid, 11.2, IMAGE_TYPES[prev_type],
                    ha="center", va="bottom", fontsize=9,
                    color=color, fontweight="bold", alpha=0.9)
            prev_type, group_start = t, i

    # Бары критериев
    for ci, (criterion, crit_ru, color) in enumerate(
        zip(CRITERIA, CRITERIA_RU, CRITERIA_COLORS)
    ):
        vals = [r["scores"].get(criterion, 0) for r in sorted_results]
        bars = ax.bar(x + offsets[ci], vals, width, label=crit_ru,
                      color=color, alpha=0.88, edgecolor='#444444', linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    str(int(val)), ha="center", va="bottom",
                    fontsize=8, color=TEXT_COLOR, fontweight="medium")

    # Разделители между типами
    prev_type = types[0]
    for i, t in enumerate(types[1:], start=1):
        if t != prev_type:
            ax.axvline(x=i - 0.5, color=BORDER_COLOR, linewidth=1.0, linestyle="--")
            prev_type = t

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", color=TEXT_COLOR, fontsize=11)
    ax.set_ylim(0, 12.0)
    ax.set_ylabel("Оценка (1–10)", color=TEXT_COLOR, fontsize=11)
    ax.set_title("Оценки описаний изображений по критериям", color=TEXT_COLOR,
                 fontsize=14, fontweight="bold", pad=20)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines[:].set_color(BORDER_COLOR)
    ax.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, edgecolor=BORDER_COLOR,
              fontsize=9, loc="lower right")
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.7)
    ax.axhline(y=7, color=BORDER_COLOR, linestyle="--", linewidth=1.0, alpha=0.7)

    plt.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(PDF_PER_IMAGE, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()


def plot_by_type(by_type: dict[str, dict[str, float]]) -> None:
    """
    Два графика рядом:
      LEFT  — grouped bar: типы изображений × критерии
      RIGHT — heatmap: типы × критерии
    """
    types = list(IMAGE_TYPES.keys())
    type_labels= [IMAGE_TYPES[t] for t in types]
    n_types = len(types)

    fig = plt.figure(figsize=(16, 6), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35, width_ratios=[1.6, 1])

    # LEFT
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG_COLOR)

    x = np.arange(len(CRITERIA))
    width  = 0.18
    n = n_types
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, t in enumerate(types):
        vals  = [by_type[t][c] for c in CRITERIA]
        color = TYPE_COLORS[t]
        bars  = ax1.bar(x + offsets[i], vals, width,
                        label=IMAGE_TYPES[t],
                        color=color, alpha=0.9, edgecolor="#444444", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.1,
                     f"{val:.1f}", ha="center", va="bottom",
                     fontsize=8, color=TEXT_COLOR, fontweight="medium")

    ax1.set_xticks(x)
    ax1.set_xticklabels(CRITERIA_RU, color=TEXT_COLOR, fontsize=11)
    ax1.set_ylim(0, 11.5)
    ax1.set_ylabel("Средняя оценка (1–10)", color=TEXT_COLOR, fontsize=11)
    ax1.set_title("Средние оценки по типам изображений", color=TEXT_COLOR,
                  fontsize=13, fontweight="bold", pad=15)
    ax1.tick_params(colors=TEXT_COLOR)
    ax1.spines[:].set_color(BORDER_COLOR)
    ax1.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
    ax1.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, edgecolor=BORDER_COLOR, fontsize=9)
    ax1.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.5)
    ax1.axhline(y=7, color=BORDER_COLOR, linestyle="--", linewidth=1.0, alpha=0.7)

    # RIGHT
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG_COLOR)

    data = np.array([[by_type[t][c] for c in CRITERIA] for t in types])
    cmap = LinearSegmentedColormap.from_list(
        "img", ["#F4F6F4", "#CDE0D5", "#5B7065", "#2A4D69"], N=256
    )
    im = ax2.imshow(data, cmap=cmap, vmin=0, vmax=10, aspect="auto")

    ax2.set_xticks(range(len(CRITERIA)))
    ax2.set_xticklabels(CRITERIA_RU, color=TEXT_COLOR, fontsize=10, rotation=15, ha="right")
    ax2.set_yticks(range(n_types))
    ax2.set_yticklabels(type_labels, color=TEXT_COLOR, fontsize=10)
    ax2.set_title("Тепловая карта оценок", color=TEXT_COLOR,
                  fontsize=13, fontweight="bold", pad=15)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.spines[:].set_color(BORDER_COLOR)

    for i in range(n_types):
        for j in range(len(CRITERIA)):
            val = data[i, j]
            txt_color = "white" if val > 6.5 else TEXT_COLOR
            ax2.text(j, i, f"{val:.1f}", ha="center", va="center",
                     fontsize=11, color=txt_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    cbar.outline.set_edgecolor(BORDER_COLOR)

    plt.tight_layout()
    plt.savefig(PDF_BY_TYPE, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()


def plot_radar(avg_scores: dict[str, float], by_type: dict[str, dict[str, float]]) -> None:
    """
    Radar chart: общая средняя + отдельная линия для каждого типа.
    """
    labels = CRITERIA_RU
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    for t, color in TYPE_COLORS.items():
        vals = [by_type[t][c] for c in CRITERIA] + [by_type[t][CRITERIA[0]]]
        ax.plot(angles, vals, color=color, linewidth=1.5, linestyle="--", alpha=0.8)

    overall_vals = [avg_scores[c] for c in CRITERIA] + [avg_scores[CRITERIA[0]]]
    ax.plot(angles, overall_vals, color=TEXT_COLOR, linewidth=2.5)
    ax.fill(angles, overall_vals, color=TEXT_COLOR, alpha=0.08)

    # Подписи значений на общей линии
    for angle, val in zip(angles[:-1], overall_vals[:-1]):
        ax.annotate(f"{val:.1f}", xy=(angle, val),
                    xytext=(angle, val + 0.8),
                    fontsize=10, color=TEXT_COLOR, fontweight="bold",
                    ha="center", va="center")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT_COLOR, fontsize=11, fontweight="medium")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="#888888", fontsize=8)
    ax.tick_params(colors=TEXT_COLOR, pad=15)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.8)
    ax.spines["polar"].set_color(BORDER_COLOR)

    # Легенда
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color=TEXT_COLOR, linewidth=2.5, label="Общее среднее")
    ] + [
        Line2D([0], [0], color=TYPE_COLORS[t], linewidth=1.5,
               linestyle="--", label=IMAGE_TYPES[t])
        for t in IMAGE_TYPES
    ]
    ax.legend(handles=legend_items, loc="upper right",
              bbox_to_anchor=(1.35, 1.1),
              facecolor=BG_COLOR, labelcolor=TEXT_COLOR,
              edgecolor=BORDER_COLOR, fontsize=9)

    ax.set_title("Средние оценки по критериям", color=TEXT_COLOR,
                 fontsize=14, fontweight="bold", pad=25)

    plt.tight_layout()
    plt.savefig(PDF_RADAR, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()


# Консольные таблицы
def print_per_image_table(eval_results: list[dict]) -> None:
    sorted_results = sorted(eval_results, key=lambda r: list(IMAGE_TYPES).index(r["type"]))

    table = Table(title="Оценки по изображениям", box=box.ROUNDED,
                  header_style="bold magenta", show_lines=True)
    table.add_column("Изображение",   min_width=20)
    table.add_column("Тип",  min_width=14)
    for ru in CRITERIA_RU:
        table.add_column(ru, justify="center")
    table.add_column("Среднее", justify="center", style="bold green")
    table.add_column("Комментарий", style="dim", max_width=38)

    prev_type = None
    for r in sorted_results:
        if r["type"] != prev_type and prev_type is not None:
            table.add_section()
        prev_type = r["type"]

        scores_vals = [str(int(r["scores"][c])) for c in CRITERIA]
        avg_img     = np.mean([r["scores"][c] for c in CRITERIA])
        table.add_row(
            r["name"],
            IMAGE_TYPES[r["type"]],
            *scores_vals,
            f"{avg_img:.1f}",
            r["comment"],
        )

    console.print()
    console.print(table)


def print_by_type_table(by_type: dict[str, dict[str, float]]) -> None:
    table = Table(title="Средние оценки по типам изображений", box=box.ROUNDED,
                  header_style="bold magenta", show_lines=True)
    table.add_column("Тип", min_width=20)
    for ru in CRITERIA_RU:
        table.add_column(ru, justify="center")
    table.add_column("Среднее", justify="center", style="bold green")

    for t, scores in by_type.items():
        vals = [scores[c] for c in CRITERIA]
        avg  = np.mean(vals)
        table.add_row(
            IMAGE_TYPES[t],
            *[f"{v:.2f}" for v in vals],
            f"[bold]{avg:.2f}[/bold]",
        )

    console.print()
    console.print(table)


# Главная функция
def run() -> None:
    console.print(Panel("Тест качества описаний изображений", expand=False))

    pairs = load_pairs(IMAGES_DIR)
    if not pairs:
        print('Не найдено пар изображения + описание')
        return

    type_counts = {}
    for p in pairs:
        type_counts[p["type"]] = type_counts.get(p["type"], 0) + 1

    print(f"Найдено пар: {len(pairs)}")
    for t, cnt in type_counts.items():
        console.print(f"  {IMAGE_TYPES[t]:25s} {cnt} шт.")

    print("Запрос к Gemini (судья)\n")

    eval_results = []
    for pair in track(pairs, description="Оценка"):
        scores_raw = evaluate_with_gemini(pair["image_bytes"], pair["description"])
        if scores_raw is None:
            continue
        eval_results.append({
            "name":    pair["name"],
            "type":    pair["type"],
            "scores":  {c: float(scores_raw.get(c, 0)) for c in CRITERIA},
            "comment": scores_raw.get("comment", ""),
        })
        time.sleep(REQUEST_DELAY)

    if not eval_results:
        print("Нет результатов")
        return

    # Агрегация
    by_type    = aggregate_by_type(eval_results)
    avg_scores = {
        c: float(np.mean([r["scores"][c] for r in eval_results]))
        for c in CRITERIA
    }
    overall = np.mean(list(avg_scores.values()))

    # Таблицы в консоль
    print_per_image_table(eval_results)
    print_by_type_table(by_type)
    print(f"Итоговая средняя оценка: {overall:.2f} / 10")

    # Сохранение
    print("Сохранение результатов")
    save_json(eval_results, by_type)

    # Графики
    print("Построение графиков]")
    plot_per_image(eval_results)
    plot_by_type(by_type)
    plot_radar(avg_scores, by_type)

    print("Оценка описаний изображений завершена")


if __name__ == "__main__":
    run()
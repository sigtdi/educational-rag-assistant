"""
test_rag_agent.py
Оценка качества RAG-агента с помощью библиотеки RAGAS.

Тестируем 2 модели × 2 режима (с RAG / без RAG):
    - qwen3.5:9b
    - Gemini 3 Flash

Метрики RAGAS:
    - faithfulness        не галлюцинирует ли модель
    - answer_relevancy    релевантен ли ответ вопросу
    - context_precision   нет ли лишних чанков в контексте
    - context_recall      покрывает ли контекст эталонный ответ

Судья для RAGAS: Gemini 3 Flash
"""

import gc
import json
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box
from pydantic import BaseModel, Field
from google import genai


from app.services.rag.chains.agent import make_agent
from app.tests.data.agent_dataset import DATASET

load_dotenv()

RESULTS_DIR = Path("data/results")
JSON_OUT    = RESULTS_DIR / "rag_agent_results.json"
PDF_OUT     = RESULTS_DIR / "rag_agent_metrics.pdf"

MODEL_NAMES = [ "qwen3.5:9b",
                "gemini-3-flash-preview"
                ]

MODEL_LABELS = {
    "qwen3.5:9b": "Qwen 3.5 9B",
    "gemini-3-flash-preview": "Gemini 3 Flash",
}

RAGAS_METRICS_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
RAGAS_METRICS_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}

console = Console()


# Агенты
def get_agent_with_rag(model_name: str):
    if model_name == 'qwen3.5:9b':
        model_type = 'local'
    else:
        model_type = 'cloud_model'
    return make_agent(from_yaml=False, with_tools=True, model_name=model_name, temperature=0.0, model_type=model_type)


def get_agent_without_rag(model_name: str):
    if model_name == 'qwen3.5:9b':
        model_type = 'local'
    else:
        model_type = 'cloud_model'

    prompt = (
        'Ты — учебный ассистент по алгоритмам и структурам данных. '
        'Твоя цель — помочь студенту понять материал глубоко, а не просто предоставить готовое решение. '
    
        'ЭКСПЕРТНАЯ БАЗА. Ты опираешься на свои внутренние знания фундаментальных алгоритмов, '
        'математического анализа и теории графов. Давай ответы, соответствующие уровню '
        'академической литературы (напр. Кормен, Седжвик). '
    
        'ДОСТОВЕРНОСТЬ И ЛОГИКА. Каждый шаг алгоритма или этап доказательства должен быть '
        'логически обоснован. Если алгоритм имеет вариации, выбери наиболее стандартную '
        'и укажи это. Если вопрос касается узкой темы, в которой есть неопределенность, '
        'сформулируй ответ исходя из общепринятых определений. '
    
        'РАССУЖДЕНИЕ. Ты должен проводить глубокий аналитический разбор: '
        'доказывать теоремы через базовые определения, разбирать работу алгоритмов '
        'на концептуальном уровне, комбинировать известные факты для решения задач. '
        'Логические переходы между фактами должны быть строгими и последовательными. '
    
        'СТРУКТУРА ОТВЕТА. Строй ответ строго по смыслу вопроса: '
        '— Алгоритм: идея, пошаговое описание, асимптотическая сложность (временная и пространственная); '
        '— Определение/Теорема: строгая формулировка и подробное пояснение логики; '
        '— Сравнение: четкие критерии, сильные и слабые стороны каждого подхода. '
        
        'ТЕРМИНОЛОГИЯ И ОФОРМЛЕНИЕ. Объясняй сложные термины при первом упоминании. '
        'Используй единообразную терминологию. Все математические выражения, индексы '
        'и оценки сложности оформляй строго в LaTeX.'
    )

    return make_agent(from_yaml=False, with_tools=False, model_name=model_name, temperature=0.0, model_type=model_type, prompt=prompt)


def cleanup_agent(agent) -> None:
    """
    Удаляет агента и освобождает память.
    """
    try:
        import torch
        del agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        del agent
        gc.collect()


# RAGAS: инициализация судьи
class JudgeScores(BaseModel):
    faithfulness:      float = Field(ge=0.0, le=10,
        description="Ответ основан на контексте, нет галлюцинаций")
    answer_relevancy:  float = Field(ge=0.0, le=10,
        description="Ответ релевантен вопросу")
    context_precision: float = Field(ge=0.0, le=10,
        description="Контекст не содержит лишних нерелевантных чанков")
    context_recall:    float = Field(ge=0.0, le=10,
        description="Контекст покрывает эталонный ответ")
    comment: str = Field(description="Одно предложение — главный недостаток или достоинство")


JUDGE_PROMPT = """
Ты — эксперт по оценке качества RAG-систем для учебников по алгоритмам.

Вопрос: {question}
Эталонный ответ: {ground_truth}
Контекст из базы знаний:
{context}
Ответ системы: {answer}

Оцени по четырём метрикам от 0.0 до 10:
- faithfulness:      ответ основан на контексте, нет выдуманных фактов (10 = нет галлюцинаций)
- answer_relevancy:  ответ отвечает на вопрос (10 = полностью отвечает)
- context_precision: контекст точен, нет лишних нерелевантных чанков (10 = только нужное)
- context_recall:    контекст покрывает эталонный ответ (10 = всё необходимое есть)

Если контекст пустой (режим без RAG):
- context_precision и context_recall = 0.0
- faithfulness оцени по фактической корректности ответа
""".strip()

GEMINI_JUDGE_MODEL = "gemini-3-flash-preview"
JUDGE_REQUEST_DELAY = 3.0


def build_gemini_client() -> genai.Client:
    import os
    return genai.Client(api_key=os.getenv('GEMINI_API_KEY'))


def judge_single(
        client: genai.Client,
        question: str,
        ground_truth: str,
        answer: str,
        contexts: list[str],
) -> Optional[JudgeScores]:
    """
    Оценивает один ответ агента через Gemini. Возвращает None при ошибке.
    """
    context_str = "\n---\n".join(contexts) if contexts else "(контекст не использовался)"
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        context=context_str,
        answer=answer,
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_JUDGE_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JudgeScores,
                temperature=0.1,
            ),
        )
        return JudgeScores.model_validate_json(response.text)
    except Exception as e:
        print(f"Ошибка судьи Gemini: {e}")
        return None


# Тестирование одного режима
def run_mode(
    model_name: str,
    use_rag: bool,
    dataset: list[dict],
    gemini_client: genai.Client,
    mode_key: str
) -> dict[str, float]:
    """
    Создает агента и считает метрики.
    """
    mode_label = f"{model_name} {'+ RAG' if use_rag else '(без RAG)'}"
    print(f"\n {mode_label}")

    agent = get_agent_with_rag(model_name) if use_rag else get_agent_without_rag(model_name)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in track(dataset, description=f"  Вопросы"):
        try:
            result = agent.ask(item["question"])
            answer = result["answer"]
            raw_contexts = result.get("contexts", [])
            ctx = [c for c in raw_contexts if isinstance(c, str) and c.strip()]
        except Exception as e:
            time.sleep(3)
            print(f"Ошибка агента: {e}")
            answer = ""
            ctx = []

        time.sleep(12)
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(ctx if ctx else [""])
        ground_truths.append(item["ground_truth"])

    # Сохранение ответов модели
    raw_out = RESULTS_DIR / f"raw_{model_name.replace(':', '_')}_{mode_key}.json"
    raw_out.write_text(
        json.dumps(
            {"questions": questions, "answers": answers,
             "contexts": contexts},
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )

    print(f"Очистка памяти после {model_name}")
    cleanup_agent(agent)

    print("Оценка RAGAS ")
    all_scores: list[JudgeScores] = []
    for q, ans, ctx, gt in track(
            list(zip(questions, answers, contexts, ground_truths)),
            description="  Оценка",
    ):
        score = judge_single(gemini_client, q, gt, ans, ctx)
        if score is not None:
            all_scores.append(score)
        time.sleep(JUDGE_REQUEST_DELAY)

    if not all_scores:
        print("Нет оценок")
        return {m: 0.0 for m in RAGAS_METRICS_KEYS}

    return {
        m: float(np.mean([getattr(s, m) for s in all_scores]))
        for m in RAGAS_METRICS_KEYS
    }



# Сохранение
def save_json(results: dict) -> None:
    serializable = {
        model: {
            mode: {k: round(v, 4) for k, v in scores.items()}
            for mode, scores in modes.items()
        }
        for model, modes in results.items()
    }
    JSON_OUT.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# Консольная таблица
def print_table(results: dict) -> None:
    table = Table(title="RAGAS-метрики агента", box=box.ROUNDED,
                  header_style="bold magenta", show_lines=True)
    table.add_column("Модель",  style="cyan",   min_width=14)
    table.add_column("Режим",   style="yellow", min_width=10)
    for key in RAGAS_METRICS_KEYS:
        table.add_column(
            RAGAS_METRICS_LABELS[key].replace("\n", " "),
            justify="right", style="white",
        )
    table.add_column("Среднее", justify="right", style="bold green")

    for model in MODEL_NAMES:
        for mode, mode_label in [("no_rag", "Без RAG"), ("with_rag", "С RAG")]:
            scores = results[model][mode]
            vals   = [scores.get(m, 0.0) for m in RAGAS_METRICS_KEYS]
            table.add_row(
                MODEL_LABELS[model].replace("\n", " "),
                mode_label,
                *[f"{v:.3f}" for v in vals],
                f"{np.mean(vals):.3f}",
            )

        # Строка с дельтой
        delta_cells = []
        for m in RAGAS_METRICS_KEYS:
            d    = results[model]["with_rag"].get(m, 0) - results[model]["no_rag"].get(m, 0)
            sign = "+" if d >= 0 else ""
            col  = "green" if d >= 0 else "red"
            delta_cells.append(f"[{col}]{sign}{d:.3f}[/{col}]")
        d_avg  = (np.mean([results[model]["with_rag"].get(m, 0) for m in RAGAS_METRICS_KEYS])
                - np.mean([results[model]["no_rag"].get(m, 0)   for m in RAGAS_METRICS_KEYS]))
        sign   = "+" if d_avg >= 0 else ""
        col    = "green" if d_avg >= 0 else "red"
        table.add_row("", "[bold]Δ RAG[/bold]", *delta_cells,
                      f"[{col}][bold]{sign}{d_avg:.3f}[/bold][/{col}]")

    console.print()
    console.print(table)


# Визуализация
BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#2E2E2E"
GRID_COLOR = "#E5E5E5"
BORDER_COLOR = "#BCBCBC"

MODEL_COLORS = ["#1D3A4D", "#5B7065", "#436D7A"]
HEATMAP_COLORS = ["#F7F9F9", "#9FB0A2", "#5B7065", "#1D3A4D"]


def plot_results(results: dict) -> None:
    # Глобальные настройки
    plt.rcParams['text.color'] = TEXT_COLOR
    plt.rcParams['axes.labelcolor'] = TEXT_COLOR
    plt.rcParams['xtick.color'] = TEXT_COLOR
    plt.rcParams['ytick.color'] = TEXT_COLOR

    fig = plt.figure(figsize=(20, 12), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

    metric_labels = [RAGAS_METRICS_LABELS[m] for m in RAGAS_METRICS_KEYS]

    # TOP
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(BG_COLOR)

    n_models = len(MODEL_NAMES)
    n_metrics = len(RAGAS_METRICS_KEYS)
    group_w = 0.75
    bar_w = group_w / (n_models * 2.2)
    x = np.arange(n_metrics)

    for mi, (model, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
        no_rag_vals = [results[model]["no_rag"].get(m, 0) for m in RAGAS_METRICS_KEYS]
        with_rag_vals = [results[model]["with_rag"].get(m, 0) for m in RAGAS_METRICS_KEYS]

        base_offset = (mi - (n_models - 1) / 2) * 2.2 * bar_w
        label_base = MODEL_LABELS[model].replace("\n", " ")

        ax1.bar(x + base_offset, no_rag_vals, bar_w,
                label=f"{label_base} (Base)",
                color=color, alpha=0.3, edgecolor=color, hatch="///", linewidth=0.5)

        bars = ax1.bar(x + base_offset + bar_w, with_rag_vals, bar_w,
                       label=f"{label_base} + RAG",
                       color=color, alpha=0.9, edgecolor="none")

        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         h + 0.015, f"{h:.2f}",
                         ha="center", va="bottom",
                         fontsize=8, color=TEXT_COLOR)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=11)
    ax1.set_ylim(0, 10.15)
    ax1.set_ylabel("Баллы (0–10)", fontsize=11)
    ax1.set_title("Сравнение производительности агентов: влияние RAG",
                  fontsize=15, fontweight="bold", pad=25)

    ax1.grid(axis="y", color=GRID_COLOR, linestyle='--', linewidth=0.6)
    for spine in ["top", "right"]: ax1.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]: ax1.spines[spine].set_color(BORDER_COLOR)

    ax1.legend(frameon=True, facecolor=BG_COLOR, edgecolor=BORDER_COLOR,
               fontsize=9, ncol=n_models, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    # Пороговая линия (benchmarking)
    ax1.axhline(0.7, color=BORDER_COLOR, linestyle="--", linewidth=1, alpha=0.8)

    # BOTTOM LEFT & RIGHT─
    ax2 = fig.add_subplot(gs[1, 0])
    _plot_heatmap_light(ax2, results, mode="no_rag", title="Производительность: Без RAG")

    ax3 = fig.add_subplot(gs[1, 1])
    _plot_heatmap_light(ax3, results, mode="with_rag", title="Производительность: С RAG")

    plt.savefig(PDF_OUT, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()


def _plot_heatmap_light(ax, results: dict, mode: str, title: str) -> None:
    data = np.array([
        [results[m][mode].get(metric, 0) for metric in RAGAS_METRICS_KEYS]
        for m in MODEL_NAMES
    ])

    cmap = LinearSegmentedColormap.from_list("academic_rag", HEATMAP_COLORS, N=256)

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_facecolor(BG_COLOR)

    ax.set_xticks(range(len(RAGAS_METRICS_KEYS)))
    ax.set_xticklabels(
        [RAGAS_METRICS_LABELS[m] for m in RAGAS_METRICS_KEYS],
        fontsize=9, rotation=20, ha="right"
    )
    ax.set_yticks(range(len(MODEL_NAMES)))
    ax.set_yticklabels(
        [MODEL_LABELS[m].replace("\n", " ") for m in MODEL_NAMES],
        fontsize=10
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)


    for i in range(len(MODEL_NAMES)):
        for j in range(len(RAGAS_METRICS_KEYS)):
            val = data[i, j]
            text_col = "white" if val > 0.7 else TEXT_COLOR
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="medium", color=text_col)

    # Настройка цветовой шкалы
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_edgecolor(BORDER_COLOR)


# Главная функция
def run() -> None:
    console.print(Panel("Тест RAG-агента (RAGAS)", expand=False))
    print(f"\nДатасет: {len(DATASET)} вопросов")
    print(f"Модели:   {', '.join(MODEL_NAMES)}")
    print(f"Судья:   Gemini 3 Flash")

    print("Инициализация RAGAS-метрик")
    gemini_client = build_gemini_client()

    results: dict = {}

    for model in MODEL_NAMES:
        print(f"\n═══ Модель: {model} ═══")
        results[model] = {}

        for use_rag, mode_key in [(False, "no_rag"), (True, "with_rag")]:
            if not use_rag:
                continue

            results[model][mode_key] = run_mode(
                model_name=model,
                use_rag=use_rag,
                dataset=DATASET,
                gemini_client=gemini_client,
                mode_key=mode_key
            )

    print_table(results)

    print("Сохранение данных")
    save_json(results)

    plot_results(results)

    console.print("Тестирование агентов завершено")


if __name__ == "__main__":
    run()
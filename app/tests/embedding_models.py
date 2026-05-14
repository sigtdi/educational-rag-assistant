"""
Оценка моделей эмбеддингов для RAG-сервиса.
Данные: чанки из теории графов (русскоязычный учебный материал).
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

MODELS: dict[str, dict] = {
    "jinaai/jina-embeddings-v4": {
        "type": "sentence_transformer",
        "name": "jinaai/jina-embeddings-v4",
        "prefix_query": "",
        "prefix_doc": "",
    },
    "Alibaba-NLP/gte-multilingual-base": {
        "type": "sentence_transformer",
        "name": "Alibaba-NLP/gte-multilingual-base",
        "prefix_query": "",
        "prefix_doc": "",
    },
    "BAAI/bge-large-en-v1.5": {
        "type": "sentence_transformer",
        "name": "BAAI/bge-large-en-v1.5",
        "prefix_query": "",
        "prefix_doc": "",
    },

    "multilingual-e5-large-instruct": {
        "type": "sentence_transformer",
        "name": "intfloat/multilingual-e5-large-instruct",
        "prefix_query": "Instruct: Given a mathematics and algorithm related query, retrieve relevant passages from the textbook\nQuery: ",
        "prefix_doc": "",
    },
    # Многоязычные модели
    "multilingual-e5-small": {
        "type": "sentence_transformer",
        "name": "intfloat/multilingual-e5-small",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
    },
    "e5-large-v2": {
        "type": "sentence_transformer",
        "name": "intfloat/e5-large-v2",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
    },
    "multilingual-e5-large": {
        "type": "sentence_transformer",
        "name": "intfloat/multilingual-e5-large",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
    },
    "bge-m3": {
        "type": "sentence_transformer",
        "name": "BAAI/bge-m3",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
    },
    # Русскоязычные модели
    "LaBSE": {
        "type": "sentence_transformer",
        "name": "sentence-transformers/LaBSE",
        "prefix_query": "",
        "prefix_doc": "",
    },
    "Roberta": {
        "type": "sentence_transformer",
        "name": "ai-forever/ru-en-RoSBERTa",
        "prefix_query": "",
        "prefix_doc": "",
    },
}

# Путь к файлу с чанками
DATA_PATH = Path(__file__).resolve().parents[1] / "document_processing" / "output" / "output_storage_preparer" / "Alg-graphs-full_images_processed_json_chunk_processed_json_storage_preparer_json.txt"

# Количество чанков для тестов (None = все)
MAX_CHUNKS = None

# Топ-K для метрики recall@K
RECALL_K_VALUES = [1, 3, 5, 10]


def load_chunks(path: Path, max_chunks: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Только текстовые чанки с непустым search_text
    chunks = [d for d in data if d.get("type") == "text" and d.get("search_text", "").strip()]
    if max_chunks:
        chunks = chunks[:max_chunks]
    print(f"Загружено чанков: {len(chunks)}")
    return chunks


# СИНТЕТИЧЕСКИЕ ЗАПРОСЫ (query–document пары для retrieval-тестов)
# Созданы вручную на основе реальных чанков из файла.

QUERY_DOC_PAIRS: list[dict] = [
    {
        "query": "Что такое ориентированный граф?",
        "relevant_keywords": ["ориентированный граф", "пара $(V, E)$", "конечное множество вершин"],
    },
    {
        "query": "определение неориентированного графа",
        "relevant_keywords": ["неориентированный граф", "множество рёбер", "пара вершин"],
    },
    {
        "query": "алгоритм обхода графа в глубину DFS",
        "relevant_keywords": ["обход в глубину", "DFS", "глубина"],
    },
    {
        "query": "алгоритм обхода графа в ширину BFS",
        "relevant_keywords": ["обход в ширину", "BFS", "ширина"],
    },
    {
        "query": "матрица смежности графа",
        "relevant_keywords": ["матрица смежности", "adjacency matrix"],
    },
    {
        "query": "список смежности для представления графа",
        "relevant_keywords": ["список смежности", "списки смежности"],
    },
    {
        "query": "алгоритм Дейкстры кратчайший путь",
        "relevant_keywords": ["Дейкстра", "кратчайший путь", "Dijkstra"],
    },
    {
        "query": "топологическая сортировка",
        "relevant_keywords": ["топологическая сортировка", "топологический порядок"],
    },
    {
        "query": "сильно связные компоненты графа",
        "relevant_keywords": ["сильно связные", "компоненты сильной связности"],
    },
    {
        "query": "максимальный поток в сети Ford-Fulkerson",
        "relevant_keywords": ["максимальный поток", "Форд-Фалкерсон", "Ford-Fulkerson"],
    },
    {
        "query": "остовное дерево минимальной стоимости",
        "relevant_keywords": ["остовное дерево", "минимальное остовное", "spanning tree"],
    },
    {
        "query": "задача коммивояжёра NP-полная",
        "relevant_keywords": ["коммивояжёр", "NP-полн", "TSP"],
    },
    {
        "query": "двудольный граф паросочетание",
        "relevant_keywords": ["двудольный", "паросочетание", "bipartite"],
    },
    {
        "query": "обнаружение сообществ в социальных сетях",
        "relevant_keywords": ["сообществ", "социальн", "community"],
    },
    {
        "query": "изоморфизм графов",
        "relevant_keywords": ["изоморфизм", "изоморфны"],
    },
    {
        "query": "эйлеров путь в графе",
        "relevant_keywords": ["Эйлер", "эйлеров", "Euler"],
    },
    {
        "query": "хроматическое число раскраска вершин",
        "relevant_keywords": ["хроматическое число", "раскраска", "chromatic"],
    },
    {
        "query": "планарный граф теорема Куратовского",
        "relevant_keywords": ["планарный", "Куратовский", "planar"],
    },
    {
        "query": "алгоритм Краскала минимальное дерево",
        "relevant_keywords": ["Краскал", "Kruskal", "минимальное дерево"],
    },
    {
        "query": "вершинное покрытие аппроксимация",
        "relevant_keywords": ["вершинное покрытие", "аппроксимац"],
    },
]


@dataclass
class EmbedModel:
    name: str
    encode_fn: Callable[[list[str]], np.ndarray]
    prefix_query: str = ""
    prefix_doc: str = ""

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self.encode_fn([self.prefix_doc + t for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_fn([self.prefix_query + t for t in texts])


def load_sentence_transformer_model(cfg: dict) -> EmbedModel:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["name"])
    return EmbedModel(
        name=cfg["name"],
        encode_fn=lambda texts: model.encode(texts, show_progress_bar=False, normalize_embeddings=True),
        prefix_query=cfg.get("prefix_query", ""),
        prefix_doc=cfg.get("prefix_doc", ""),
    )


MODEL_LOADERS: dict[str, Callable] = {
    "sentence_transformer": load_sentence_transformer_model
}


def load_model(model_id: str) -> EmbedModel | None:
    cfg = MODELS[model_id]
    loader = MODEL_LOADERS.get(cfg["type"])
    if loader is None:
        print(f"  Неизвестный тип модели: {cfg['type']}")
        return None
    try:
        print(f"  Загрузка {cfg['name']}")
        return loader(cfg)
    except Exception as e:
        print(f"  Не удалось загрузить {cfg['name']}: {e}")
        return None


# ТЕСТЫ
@dataclass
class TestResult:
    model_id: str
    test_name: str
    metrics: dict = field(default_factory=dict)
    error: str = ""
    duration_sec: float = 0.0


# Тест 1: Скорость индексирования
def test_indexing_speed(model: EmbedModel, chunks: list[dict]) -> TestResult:
    """Измеряет время векторизации всего корпуса чанков."""
    texts = [c["search_text"] for c in chunks]
    t0 = time.perf_counter()
    vecs = model.embed_docs(texts)
    dt = time.perf_counter() - t0
    return TestResult(
        model_id=model.name,
        test_name="indexing_speed",
        metrics={
            "total_sec": round(dt, 3),
            "chunks_per_sec": round(len(texts) / dt, 1),
            "dim": vecs.shape[1],
        },
        duration_sec=dt,
    )


# Тест 2: Retrieval — Recall@K по синтетическим запросам
def test_retrieval_recall(
    model: EmbedModel,
    chunks: list[dict],
    query_doc_pairs: list[dict],
    k_values: list[int] = RECALL_K_VALUES,
) -> TestResult:
    """
    Для каждого запроса считаем top-K по косинусной близости.
    Релевантным считается чанк, search_text которого содержит хотя бы одно
    из ключевых слов запроса.
    """
    texts = [c["search_text"] for c in chunks]
    t0 = time.perf_counter()
    doc_vecs = model.embed_docs(texts)

    queries = [p["query"] for p in query_doc_pairs]
    q_vecs = model.embed_queries(queries)
    dt = time.perf_counter() - t0

    recall_at_k = {k: [] for k in k_values}
    mrr_scores = []

    sims = cosine_similarity(q_vecs, doc_vecs)  # (Q, D)

    for qi, pair in enumerate(query_doc_pairs):
        keywords = [kw.lower() for kw in pair["relevant_keywords"]]
        ranked_idx = np.argsort(-sims[qi])

        # MRR: позиция первого релевантного результата
        first_rel_rank = None
        for rank, idx in enumerate(ranked_idx, start=1):
            txt = texts[idx].lower()
            if any(kw in txt for kw in keywords):
                first_rel_rank = rank
                break
        mrr_scores.append(1.0 / first_rel_rank if first_rel_rank else 0.0)

        # Recall@K
        for k in k_values:
            top_k_texts = [texts[i].lower() for i in ranked_idx[:k]]
            hit = any(
                any(kw in t for kw in keywords)
                for t in top_k_texts
            )
            recall_at_k[k].append(float(hit))

    metrics = {"mrr": round(float(np.mean(mrr_scores)), 4)}
    for k in k_values:
        metrics[f"recall@{k}"] = round(float(np.mean(recall_at_k[k])), 4)

    return TestResult(
        model_id=model.name,
        test_name="retrieval_recall",
        metrics=metrics,
        duration_sec=dt,
    )


# Тест 3: Семантическая связность внутри раздела
def test_intra_section_coherence(model: EmbedModel, chunks: list[dict]) -> TestResult:
    """
    Для каждого раздела (section_path) считаем среднюю попарную косинусную
    близость. Хорошая модель даст высокое значение: чанки одного раздела
    должны быть близки.
    """
    from collections import defaultdict
    section_map: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        sp = c.get("section_path", "").strip()
        if sp:
            section_map[sp].append(c["search_text"])

    # Берём только разделы с 2+ чанками
    sections = {k: v for k, v in section_map.items() if len(v) >= 2}
    if not sections:
        return TestResult(model_id=model.name, test_name="intra_section_coherence",
                          error="Нет разделов с 2+ чанками")

    t0 = time.perf_counter()
    scores = []
    for texts in sections.values():
        vecs = model.embed_docs(texts)
        sim_matrix = cosine_similarity(vecs)
        # Среднее верхнего треугольника (без диагонали)
        n = len(vecs)
        upper = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
        scores.append(np.mean(upper))
    dt = time.perf_counter() - t0

    return TestResult(
        model_id=model.name,
        test_name="intra_section_coherence",
        metrics={
            "mean_intra_sim": round(float(np.mean(scores)), 4),
            "min_intra_sim": round(float(np.min(scores)), 4),
            "max_intra_sim": round(float(np.max(scores)), 4),
            "sections_evaluated": len(sections),
        },
        duration_sec=dt,
    )


# Тест 4: Межсекционное разделение (inter/intra ratio)
def test_inter_intra_ratio(model: EmbedModel, chunks: list[dict]) -> TestResult:
    """
    Отношение средней внутрисекционной близости к межсекционной.
    Чем выше ratio, тем лучше модель разделяет темы.
    Используем центроиды разделов для эффективности.
    """
    from collections import defaultdict
    section_map: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        sp = c.get("section_path", "").strip()
        if sp:
            section_map[sp].append(c["search_text"])

    sections = {k: v for k, v in section_map.items() if len(v) >= 2}
    if len(sections) < 2:
        return TestResult(model_id=model.name, test_name="inter_intra_ratio",
                          error="Недостаточно разделов")

    t0 = time.perf_counter()
    centroids = {}
    intra_sims = []
    for sec, texts in sections.items():
        vecs = model.embed_docs(texts)
        centroids[sec] = vecs.mean(axis=0)
        if len(vecs) > 1:
            sim_matrix = cosine_similarity(vecs)
            n = len(vecs)
            upper = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
            intra_sims.extend(upper)

    centroid_matrix = np.stack(list(centroids.values()))
    inter_sim_matrix = cosine_similarity(centroid_matrix)
    n = len(centroid_matrix)
    inter_sims = [inter_sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]

    mean_intra = float(np.mean(intra_sims))
    mean_inter = float(np.mean(inter_sims))
    ratio = mean_intra / mean_inter if mean_inter > 0 else 0.0
    dt = time.perf_counter() - t0

    return TestResult(
        model_id=model.name,
        test_name="inter_intra_ratio",
        metrics={
            "mean_intra_sim": round(mean_intra, 4),
            "mean_inter_sim": round(mean_inter, 4),
            "intra_inter_ratio": round(ratio, 4),
        },
        duration_sec=dt,
    )


# Тест 5: Устойчивость к перефразированию
PARAPHRASE_PAIRS = [
    ("граф с направленными рёбрами", "ориентированный граф — пара вершин и дуг"),
    ("обход вершин графа в глубину", "рекурсивный алгоритм DFS"),
    ("минимальное остовное дерево", "дерево наименьшего суммарного веса рёбер"),
    ("максимальный поток в транспортной сети", "алгоритм Форда-Фалкерсона для потоков"),
    ("задача о кратчайшем пути", "алгоритм Дейкстры для взвешенного графа"),
    ("паросочетание в двудольном графе", "максимальное сопоставление вершин двух долей"),
    ("вершины графа одного цвета не смежны", "правильная раскраска вершин хроматическое число"),
    ("список соседей каждой вершины", "представление графа через списки смежности"),
    ("сильно связная компонента", "подграф с путём между любыми двумя вершинами"),
    ("планарный граф без пересечений рёбер", "граф укладываемый на плоскость"),
]


def test_paraphrase_stability(model: EmbedModel) -> TestResult:
    """
    Похожие по смыслу, но сформулированные по-разному тексты должны давать
    высокую косинусную близость.
    """
    t0 = time.perf_counter()
    sims = []
    for a, b in PARAPHRASE_PAIRS:
        va, vb = model.embed_queries([a, b])
        sim = float(cosine_similarity([va], [vb])[0][0])
        sims.append(sim)
    dt = time.perf_counter() - t0

    return TestResult(
        model_id=model.name,
        test_name="paraphrase_stability",
        metrics={
            "mean_sim": round(float(np.mean(sims)), 4),
            "min_sim": round(float(np.min(sims)), 4),
            "max_sim": round(float(np.max(sims)), 4),
            "pairs_evaluated": len(sims),
        },
        duration_sec=dt,
    )


# Тест 6: Разделение нерелевантных пар
NEGATIVE_PAIRS = [
    ("алгоритм Дейкстры кратчайший путь", "обнаружение сообществ в социальных сетях"),
    ("матрица смежности", "задача коммивояжёра NP-полная"),
    ("эйлеров путь", "хроматическое число раскраска вершин"),
    ("топологическая сортировка", "планарный граф Куратовский"),
    ("двудольное паросочетание", "обход дерева в глубину"),
]


def test_negative_separation(model: EmbedModel) -> TestResult:
    """
    Несвязанные тексты должны давать низкую косинусную близость.
    Сравниваем с paraphrase-парами — gap между ними важен.
    """
    t0 = time.perf_counter()
    sims = []
    for a, b in NEGATIVE_PAIRS:
        va, vb = model.embed_queries([a, b])
        sim = float(cosine_similarity([va], [vb])[0][0])
        sims.append(sim)
    dt = time.perf_counter() - t0

    return TestResult(
        model_id=model.name,
        test_name="negative_separation",
        metrics={
            "mean_sim": round(float(np.mean(sims)), 4),
            "max_sim": round(float(np.max(sims)), 4),
            "pairs_evaluated": len(sims),
        },
        duration_sec=dt,
    )



# ЗАПУСК ВСЕХ ТЕСТОВ
ALL_TESTS = [
    ("Скорость индексирования", test_indexing_speed),
    ("Retrieval Recall@K + MRR", test_retrieval_recall),
    ("Связность внутри раздела", test_intra_section_coherence),
    ("Inter/Intra ratio", test_inter_intra_ratio),
    ("Устойчивость к перефразированию", test_paraphrase_stability),
    ("Разделение нерелевантных пар", test_negative_separation),
]


def run_all(model_ids: list[str] | None = None) -> list[TestResult]:
    chunks = load_chunks(DATA_PATH, max_chunks=MAX_CHUNKS)

    if model_ids is None:
        model_ids = list(MODELS.keys())

    all_results: list[TestResult] = []

    for mid in model_ids:
        print(f"\n{'='*60}")
        print(f"Модель: {mid}")
        print(f"{'='*60}")
        model = load_model(mid)
        if model is None:
            continue

        for test_name, test_fn in ALL_TESTS:
            print(f"  → {test_name}...", end=" ", flush=True)
            try:
                if test_fn in (test_paraphrase_stability, test_negative_separation):
                    result = test_fn(model)
                elif test_fn == test_retrieval_recall:
                    result = test_fn(model, chunks, QUERY_DOC_PAIRS)
                else:
                    result = test_fn(model, chunks)
                result.model_id = mid
                print(f"✓ ({result.duration_sec:.2f}s)")
                for k, v in result.metrics.items():
                    print(f"       {k}: {v}")
            except Exception as e:
                result = TestResult(model_id=mid, test_name=test_name, error=str(e))
                print(f"✗ ОШИБКА: {e}")
            all_results.append(result)

    return all_results


# СВОДНАЯ ТАБЛИЦА
def print_summary(results: list[TestResult]) -> None:
    """Сводная таблица ключевых метрик по всем моделям."""
    from collections import defaultdict

    key_metrics = {
        "indexing_speed": "chunks_per_sec",
        "retrieval_recall": "recall@5",
        "retrieval_recall_mrr": "mrr",
        "intra_section_coherence": "mean_intra_sim",
        "inter_intra_ratio": "intra_inter_ratio",
        "paraphrase_stability": "mean_sim",
        "negative_separation": "mean_sim",
    }

    # Группировка по модели и тесту
    by_model: dict[str, dict] = defaultdict(dict)
    for r in results:
        if not r.error:
            by_model[r.model_id][r.test_name] = r.metrics

    print("\n" + "=" * 100)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 100)
    header = f"{'Модель':<40} {'speed(c/s)':>10} {'recall@5':>9} {'MRR':>7} {'intra_sim':>10} {'i/i_ratio':>10} {'para_sim':>9} {'neg_sim':>8}"
    print(header)
    print("-" * 100)

    for mid, tests in by_model.items():
        speed     = tests.get("indexing_speed", {}).get("chunks_per_sec", "-")
        recall5   = tests.get("retrieval_recall", {}).get("recall@5", "-")
        mrr       = tests.get("retrieval_recall", {}).get("mrr", "-")
        intra     = tests.get("intra_section_coherence", {}).get("mean_intra_sim", "-")
        ratio     = tests.get("inter_intra_ratio", {}).get("intra_inter_ratio", "-")
        para      = tests.get("paraphrase_stability", {}).get("mean_sim", "-")
        neg       = tests.get("negative_separation", {}).get("mean_sim", "-")

        short_name = mid if len(mid) <= 38 else mid[:35] + "..."
        print(f"{short_name:<40} {str(speed):>10} {str(recall5):>9} {str(mrr):>7} "
              f"{str(intra):>10} {str(ratio):>10} {str(para):>9} {str(neg):>8}")

    print("=" * 100)
    print()
    print("Легенда:")
    print("  speed(c/s)  — чанков в секунду при индексировании (выше = лучше)")
    print("  recall@5    — доля запросов, где релевантный чанк в топ-5 (выше = лучше)")
    print("  MRR         — Mean Reciprocal Rank (выше = лучше)")
    print("  intra_sim   — средняя близость чанков одного раздела (выше = лучше)")
    print("  i/i_ratio   — intra/inter ratio по разделам (выше = лучше)")
    print("  para_sim    — близость перефразов (выше = лучше)")
    print("  neg_sim     — близость нерелевантных пар (ниже = лучше)")


# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
def save_results(results: list[TestResult], path: str = "eval_results.json") -> None:
    out = []
    for r in results:
        out.append({
            "model_id": r.model_id,
            "test_name": r.test_name,
            "metrics": r.metrics,
            "error": r.error,
            "duration_sec": r.duration_sec,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Оценка моделей эмбеддингов для RAG")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Список model_id для запуска. Доступны: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=MAX_CHUNKS,
        help="Максимум чанков из файла (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=str, default="eval_results.json",
        help="Куда сохранить результаты (default: %(default)s)",
    )
    args = parser.parse_args()

    MAX_CHUNKS = args.max_chunks

    results = run_all(model_ids=args.models)
    print_summary(results)
    save_results(results, args.output)
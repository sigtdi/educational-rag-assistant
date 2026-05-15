"""
Оценка качества гибридного поиска (с реранкером и без).

Метрики (при k = 1, 3, 5, 10, 20):
    - MRR@k    Mean Reciprocal Rank
    - Recall@k
    - Precision@k
    - NDCG@k
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from tqdm import tqdm

from app.services.rag.retrieval.retriever import HybridRetriever
from app.services.rag.retrieval.retriever_config import RetrieverConfig

K_VALUES = [1, 3, 5, 10, 20]
REPORT_K = 5

RESULTS_DIR  = Path("data/results")
JSON_OUT     = RESULTS_DIR / "retrieval_metrics.json"
PDF_OUT      = RESULTS_DIR / "retrieval_metrics.pdf"

console = Console()

# Датасет: вопросы и релевантные чанки
DATASET: list[dict] = [
    {
        "topic": "Мастер-Теорема",
        "queries": {
            "Основная теорема о рекуррентных соотношениях метод декомпозиции": [
                "560b3ea3-d006-40db-a3d8-8e490c47b68a",
                "e4231b82-3eec-4258-bc58-4ae995dd07eb",
                "a42c1f8e-eaff-42d0-9ce9-777fb586524f",
                "36d5215e-9efb-4824-9078-d1494d8d49da"
            ],
            "Оценка сложности алгоритмов типа разделяй и властвуй": [
                "635393f3-b7c4-4ac9-8d98-f3ad3df780ad",
                "d116137f-3c18-48ee-83ca-3d19d40e5be1",
                "171a3206-581f-430f-b169-31e01b0e6e89",
                "37196ada-0042-4519-8751-fea83d0f6416"
            ],
            "Решение уравнения $T(n) = aT(n/b) + f(n)$": [
                "5e9be28a-e6e9-4b78-8c35-ad9942de3563",
                "e4231b82-3eec-4258-bc58-4ae995dd07eb",
                "99ad6b37-b3ff-4d32-a1df-8960df4a0c04",
                "196047a2-3112-426d-8258-9384cd8219f5"
            ]
        }
    },
    {
        "topic": "Свойства красно-черных деревьев",
        "queries": {
            "Определение и пять условий красно-черного дерева.": [
                "7545e77b-691c-427a-a776-7d94884c5a7b"
            ],
            "Сбалансированное бинарное дерево поиска с цветовыми метками узлов и черной высотой.": [
                "271c5ea7-21b2-4e21-8648-964cac72fe16",
                "cbf88ca3-20d0-4996-b775-e6d76bc56418"
            ],
            "Ограничение высоты дерева через количество узлов как $h \\le 2\\log(n+1)$.": [
                "8f81fd89-08a5-428a-9295-f670fcf40d3e",
                "a9a0a5c4-1958-4311-bb51-e0ae27af0d96",
                "4b473b76-28b4-4a44-98ed-69de3a19c278"
            ]
        }
    },
    {
        "topic": "Алгоритм Кнута-Морриса-Пратта",
        "queries": {
            "Префикс-функция в алгоритме КМП.": [
                "3da89a53-2b6c-4608-9596-0e07175ca415",
                "07ef6d4a-b8bd-43fc-a2f2-66c158818e60",
                "d04c48dc-ad93-4c3c-be19-7f717cf1dde0"
            ],
            "Поиск подстроки в строке с использованием таблицы сдвигов по префиксам.": [
                "07ef6d4a-b8bd-43fc-a2f2-66c158818e60",
                "dfc6482e-d8ed-49e6-851b-18c68d7e1fcf",
                "7797bd8b-9953-4ea0-9bb3-df888090a045"
            ],
            "Вычисление значений $\\pi[q] = \\max \\{k : k < q \\text{ и } P_k \\sqsupset P_q\\}$": [
                "b350503a-4cc4-43bc-9b7c-fcdfee7e159d",
                "cb4e9bc4-68b2-4eea-86ff-662c08e9bf39",
                "1cb8a2cd-630c-45d9-b31d-bf3967611307",
                "6f311ac9-d07c-4cb7-a7d6-ee490e0dfae1"
            ]
        }
    },
    {
        "topic": "Сортировка кучей (Heapsort)",
        "queries": {
            "Алгоритм пирамидальной сортировки и свойства кучи.": [
                "c4a3da5d-656f-4b2d-ab1c-f0933754bd33",
                "639d886a-7e85-4b39-86e0-f959642bab1b",
                "6b6c1c16-3989-4048-a97a-7072f000dff4",
                "cf0a46e0-6a36-41e4-b6e8-f9c1fbff7be2"
            ],
            "Поддержание основного свойства невозрастающего (или неубывающего) дерева в массиве.": [
                "49d9884a-e18b-4591-b169-0c331a108657"
            ],
            "Время работы процедуры MAX-HEAPIFY для узла на высоте h.": [
                "c27aed40-2a09-433b-9558-1d67fdcc74ca",
                "0318eab0-da6c-4081-b75a-1aff92ec1300",
                "5c887ec5-a028-4e98-baf7-ebee4710c37e"
            ]
        }
    },
    {
        "topic": "Алгоритм Дейкстры",
        "queries": {
            "Поиск кратчайших путей из одной вершины в графе": [
                "6ac7b16a-cd6c-41b6-b728-78f326e2acd3",
                "8749310b-eea4-4766-a68f-e6dc67c5fca9",
                "da4d804c-384a-426d-918e-ee95ab299608",
                "977f6c19-30e3-4024-9b23-47acc59b832a"
            ],
            "Жадный алгоритм для нахождения минимального расстояния в графе с неотрицательными весами ребер": [
                "08bc4165-4426-4ad6-9de5-f20039245b8e",
                "5edd7d6c-ed44-47d6-b77d-65faa9a91a77"
            ],
            "Релаксация ребра $(u, v)$ через условие $d[v] > d[u] + w(u, v)$.": [
                "94df83d4-153f-4f89-8ea2-8b644981437c",
                "59e6f7fb-f7fc-4d0e-bceb-a05dc2dc104f",
                "6520863a-3176-42a2-ad8e-2c80b44589f1",
                "7cd7c195-3544-4661-9038-d8a242115ec5"
            ]
        }
    },
    {
        "topic": "Расстояние Левенштейна",
        "queries": {
            "Редакционное расстояние между строками и операции редактирования.": [
                "e94d95f6-8f1b-4147-a128-1fb7108e91f5",
                "d6076804-f850-4ebe-9250-eb3399dba861",
                "883509bb-b6e2-4aeb-868e-b4ed03c4f6f0"
            ],
            "Минимальное количество замен, вставок и удалений для трансформации одной последовательности в другую.": [],
            "Формула динамического программирования $D(i, j) = \\min \\{D(i-1, j)+1, D(i, j-1)+1, D(i-1, j-1)+m(a_i, b_j)\\}$.": [
                "83377449-a840-47fc-8694-76c446d7ee1b",
                "74f7b87c-59b2-4917-9cbd-dd0014269a83",
                "eb36896b-e580-427d-a352-895d045dbd78"
            ]
        }
    },
    {
        "topic": "Хеширование с открытой адресацией",
        "queries": {
            "Методы разрешения коллизий при помощи линейного или квадратичного исследования.": [
                "294946e0-0a83-4ce7-8218-d1b03d74b464"
            ],
            "Заполнение хеш-таблицы без использования связанных списков.": [
                "9a15ed47-cc52-462c-aef7-2e7d3c6e4a9e",
                "d4f21c86-f151-4ea7-8440-235af221c830"
            ],
            "Функция пробирования вида $h(k, i) = (h'(k) + i) \\pmod m$.": [
                "6bb50d94-9bdb-4fbe-a143-57df67ee7faa"
            ]
        }
    },
    {
        "topic": "Алгоритм Форда-Фалкерсона",
        "queries": {
            "Нахождение максимального потока в транспортной сети.": [
                "9ec86cc1-4d32-4203-8546-0d23e49d33d8",
                "1fe0c9a1-532b-49ad-9114-af50a87fe550",
                "b33ce44d-a2da-4319-9740-6a05f518c2e0",
                "36328297-5dc4-4189-a850-4a485a18fd79"
            ],
            "Метод увеличивающих путей и остаточных сетей в графе.": [
                "71ea62fc-315f-4760-8ea4-277114e9c2b8",
                "3cb8feb1-68e8-4c10-8091-334f453f9526",
                "17db6e9b-04b2-4a7b-9db4-fb76224d07ac"
            ],
            "Теорема о максимальном потоке и минимальном разрезе $|f| = c(S, T)$": [
                "df8bd264-2bf8-4e2e-a1a7-b1dc9de79093",
                "ba44b2e9-7397-4b62-a276-9548f757ee95",
                "527f2ca0-57fa-4629-b72e-18c32bbfd5f1",
                "d7aced31-099b-4fa5-a042-e44ef787f2a3"
            ]
        }
    },
    {
        "topic": "B-деревья",
        "queries": {
            "Структура и свойства B-дерева для внешних систем памяти.": [
                "0cc0a031-cc67-48c2-bf1f-9e19bb07afbf",
                "7c3f0029-97fc-47e1-aafc-782674777cc9",
                "d4166537-5ca0-4de2-8f52-1fedc5bd5acb"
            ],
            "Многоходовое сбалансированное дерево поиска с заданным минимальным ветвлением.": [
                "a3651873-9b51-4c9a-82bd-0cba9e8bc93b",
                "5fcdaf11-3b47-4dfe-8a27-db05f9506237",
                "e4294f92-7b20-4b51-b6e4-54f81f653a66"
            ],
            "Условие на количество ключей в узле $t-1 \\le n[x] \\le 2t-1$.": [
                "1c755777-9d3c-4446-aff0-2a810c4206d2"
            ]
        }
    },
    {
        "topic": "Алгоритм Беллмана-Форда",
        "queries": {
            "Поиск кратчайших путей в графах с отрицательными весами ребер.": [
                "62a8e7b4-2cfe-46c3-ab49-aaccb94123e3",
                "8a685653-ecda-4bdb-8b3f-1267b16e626a",
                "b7a026f6-6eff-4bee-a137-cd596bd467ff",
                "2897843a-f151-425d-9a48-5526d8931bfb"
            ],
            "Алгоритм обнаружения циклов отрицательного веса из заданного источника.": [
                "ed152b69-ec02-422b-8649-9296fa3b931e",
                "a7ad569a-f039-4312-974e-13aaa1c8fb14",
                "23b3a0aa-9fcb-4757-956f-04b13502dc4c"
            ],
            "Итеративная проверка условия $d[v] \\le d[u] + w(u, v)$ для всех $E$ ребер.": [
                "5c0855df-065f-447b-aada-f1f598873544",
                "59e6f7fb-f7fc-4d0e-bceb-a05dc2dc104f",
                "c543d55d-6273-4901-8b7f-f39d3be740da",
                "f8328835-3710-4293-b891-49bb1214990c"
            ]
        }
    },
    {
        "topic": "Суффиксные деревья",
        "queries": {
            "Построение суффиксного дерева для строки.": [
                "d16cb6d4-4761-4429-bec2-df11a1c24d10",
                "21684221-647b-49f4-9e99-cfc6eeb3068d",
                "5bce4529-ae42-4b1d-adc9-0a4c35078761"
            ],
            "Сжатое дерево всех суффиксов заданной последовательности символов.": [
                "4890dc0f-9e64-4743-ad75-40ae78a868db",
                "215fc10f-2f48-482a-9d53-27abc0367042",
                "013c1874-7bf6-41e7-bd04-06533d57e708"
            ],
            "Алгоритм Укконена для построения структуры за время O(n).": [
                "6017f3b1-61ac-4563-a0b4-af59027c5d58",
                "f8363d97-d6b7-4166-9930-6ef2c704744a",
                "10b3c719-2b80-4fe5-a0bb-f8f892ebceff",
                "8a604bcf-9a9b-4986-a206-50d6eb10531d"
            ]
        }
    },
    {
        "topic": "Биномиальные кучи",
        "queries": {
            "Определение и операции над биномиальными очередями с приоритетами.": [
                "89a29395-238d-46a2-8d8c-9af009e5d899",
                "8c972495-bc8d-4041-830b-4c7b7e8b9081",
                "39b5c214-4f1f-43f1-9fd9-17a18d28af60",
                "bdb941e5-c926-42ff-b922-90e6589ca180"
            ],
            "Объединение набора биномиальных деревьев с логарифмическим временем работы.": [
                "1f1d78b9-9669-4611-938c-a1970cab5d21",
                "4fa8dd1c-2a2f-4ce8-84e4-62d16f3006d6",
                "c8635773-f6b5-45c5-a27b-d3a291cb8067"
            ],
            "Количество узлов в дереве $B_k$ равное $2^k$.": [
                "35f2647d-217b-4d2f-a6b9-0bf78806caae",
                "1b1e926c-8874-40b8-85b7-7739225d7e84"
            ]
        }
    },
    {
        "topic": "Алгоритм Флойда-Уоршелла",
        "queries": {
            "Нахождение кратчайших путей между всеми парами вершин графа.": [
                "e7b5760d-06c9-444b-8a33-97307c4c6344",
                "c65bc8e2-2076-4722-8881-0cc3165343c0",
                "6b320906-9fa1-41a4-b4b2-3798a1f6ec9b",
                "4b26571f-c04f-4c51-a2b8-ab9614463831"
            ],
            "Метод динамического программирования для вычисления матрицы расстояний.": [
                "4fd5917f-4a4e-499c-9a62-ef555637149d",
                "d252bf32-fa4d-4ba9-9bb8-3846823e3344",
                "456a92b2-8ae0-42cf-8e19-87a3a89bc881",
                "8b8f3ee9-13dd-4c00-a3d6-6afe76236d1d"
            ],
            "Обновление значений по формуле $d_{ij}^{(k)} = \\min(d_{ij}^{(k-1)}, d_{ik}^{(k-1)} + d_{kj}^{(k-1)})$.": [
                "fbf70f70-c3e5-4aea-91dc-72b17c9c323f",
                "0ba1bc3c-db8a-429e-aa89-54c5a1e43415",
                "e9e00046-0c6e-4481-83db-2eacda1e01e0"
            ]
        }
    },
    {
        "topic": "Амортизационный анализ",
        "queries": {
            "Метод потенциалов и бухгалтерский метод оценки сложности.": [
                "aacdc265-7ba6-409c-baf9-4073bb684e2f",
                "1cfa4468-45cf-470b-a58f-cc21f7b26ee4",
                "0f1d1ad4-c5ab-42c9-bb98-5f9f23354e6a",
                "31e050eb-72f2-4d7d-9fd4-a3c6e07d4c04"
            ],
            "Среднее время выполнения последовательности операций в худшем случае.": [
                "d9114420-197c-411f-b87f-eec3bd6a3d29",
                "d2c7df89-06d7-4a9a-913a-672d78488ff3",
                "a93c9efe-63e3-45dd-b356-9021fa0e8bd9",
                "9f3d51cb-d3a1-4e9e-a622-0f825ea2c475"
            ],
            "Определение амортизированной стоимости как $\\hat{c}_i = c_i + \\Phi(D_i) - \\Phi(D_{i-1})$.": [
                "31e050eb-72f2-4d7d-9fd4-a3c6e07d4c04",
                "a43631d5-406c-4235-b7f1-d09f80585d67",
                "f866725d-da63-4730-9287-ff06ae572bb1"
            ]
        }
    },
    {
        "topic": "Алгоритм Хоара (Quicksort)",
        "queries": {
            "Метод быстрой сортировки с выбором опорного элемента.": [
                "819c31cd-7174-4e89-831d-bcce7b4a2ac2",
                "20a388fe-fcf2-4a9d-bc48-955152286f1a",
                "7b742716-b901-401c-912a-a1d7003d6939",
                "22b1db4a-45c0-48df-980d-a1cca41c280d",
                "a1e3394c-c39e-4343-91c2-861e38e26d61"
            ],
            "Разделение массива на две части относительно пивота (partitioning)": [
                "70680f43-dd0a-47f9-aab5-d3a32db54c25",
                "d3fdce19-5f8b-423f-b0bc-cbca61a20d62",
                "81c6d403-a735-41b8-952f-49512f56b628",
                "25b9b8db-b553-4f3a-b283-d68183abed9a"
            ],
            "Математическое ожидание времени работы при случайном выборе $E[T(n)] = O(n \\log n)$": [
                "cd6b6861-838f-41e2-b45e-7e0809caa01c",
                "f3073301-44b0-4c9f-b7c3-cd57fc180908",
                "729a8d3d-9be4-4ef3-badb-5a7448798a87"
            ]
        }
    },
    {
        "topic": "Система непересекающихся множеств (DSU)",
        "queries": {
            "Структура данных 'Лес непересекающихся множеств'.": [
                "5c976f5e-8351-49fc-a987-9391e14d26dc",
                "f5461edf-9ad7-4e8d-bb7a-597db2e7f33f",
                "b46bda6f-3ebd-4329-ae74-10de4e465db0"
            ],
            "Операции объединения по рангу и сжатия путей.": [
                "9d92fb61-3e3d-4ad9-a92b-d0159621d9cc",
                "be34fc53-a877-4f3a-9831-4e7b4755516c",
                "5fbfd86f-6a31-4c6f-809e-4ec4422aabab",
                "48b0b737-c210-454f-b063-fb6136f5cce9",
                "6bf2dbca-5195-4a32-8a64-e18c27fa5ea7",
                "8b4e3cb0-555e-4f98-a5b0-1b7a0650bdab"
            ],
            "Оценка сложности через обратную функцию Аккермана $\\alpha(n)$.": [
                "167ef53d-f6d7-4f5c-8a03-0a0badf0382a",
                "047f912f-fdc6-416e-bc51-723e890d9010",
                "52d697ee-b037-4f05-b786-d7bd16d14ed5"
            ]
        }
    },
    {
        "topic": "NP-полнота",
        "queries": {
            "Определение класса NP-полных задач и полиномиальная сводимость.": [
                "faf8e823-4168-47cc-a34b-2d2d80d6cee2",
                "0f661091-b75b-4027-b37c-8754528ed2a5",
                "f3af3e1f-6cd0-4252-bd93-36811d04cbe8",
                "cbcd5588-3594-49d7-9642-14bcc3a7b7ae"
            ],
            "Задачи, к которым сводится любая задача из класса NP за полиномиальное время?": [
                "0f661091-b75b-4027-b37c-8754528ed2a5",
                "cbcd5588-3594-49d7-9642-14bcc3a7b7ae"
            ],
            "Теорема Кука-Левина о выполнимости булевых формул ($SAT$).": [
                "fcc8c131-b554-4cfa-b105-489549a43e4f",
                "e1e69aa7-184a-4496-89ff-60ebe4d5c226",
                "0735260e-4ebd-47d4-bcd9-fec999210ded"
            ]
        }
    },
    {
        "topic": "Алгоритм Бойера-Мура",
        "queries": {
            "Поиск подстроки с использованием эвристики «плохого символа» и «хорошего суффикса»": [
                "cbd14609-6bbc-4b92-ba36-7f51ab8bbba0",
                "6b56ab77-bc8d-4fe6-9db5-11a7c4337897",
                "82605daa-de09-4eaa-9057-181f4974097b",
                "07ef6d4a-b8bd-43fc-a2f2-66c158818e60"
            ],
            "Алгоритм быстрого сопоставления строк путем сканирования символов справа налево": [
                "5490b308-8b16-489b-8c2f-bc8d210a7ce6",
                "23b08f94-1da5-433a-88b9-6aa3f8c82563",
                "6c5ac418-cc40-4f8d-9e8b-23dd3eecd30f",
                "82605daa-de09-4eaa-9057-181f4974097b"
            ],
            "Сдвиг шаблона на основе функции $\\gamma(j)$  и таблицы стоп-символов": [
                "07ef6d4a-b8bd-43fc-a2f2-66c158818e60",
                "af83427a-69a6-437e-9070-2365e5e10c07",
                "e3c9e762-8611-4526-b3c4-286624a04caa"
            ]
        }
    },
    {
        "topic": "Генерация перестановок",
        "queries": {
            "Алгоритмы комбинаторной генерации всех перестановок множества": [
                "6e5ccc94-2458-4177-9725-3ac24148d965",
                "2abed4f7-2ac7-4591-963c-42467fc79878",
                "14198ef3-fbab-4d9a-a9c4-444016b147a0",
                "4ef6b3c2-299a-4ce9-91b0-db9abc6daff6"
            ],
            "Построение лексикографического порядка последовательностей элементов": [
                "e69d8d20-3ed0-473b-99dc-7e79ee70da53",
                "3720524b-f2a4-4470-ae8a-7b3383ebbc87",
                "44887aa0-2742-4785-8bb3-224bd5b37dba",
                "d70be34f-910b-4512-bdcc-6a58017bc3b2",
                "4943dcb5-75bb-4b3a-b979-4b864148cb59",
                "97aa532f-108e-49af-a386-b0b39c4a14f9"
            ],
            "Формула общего количества перестановок для n элементов: n!": [
                "358d58d9-61a0-407d-8881-60aec176b813",
                "754d175d-36ad-48a6-85fe-83b7ccfa438a",
                "89673923-2270-4797-8628-f9be1dcc8194"
            ]
        }
    },
    {
        "topic": "Алгоритм Краскала",
        "queries": {
            "Построение минимального остовного дерева на основе сортировки ребер": [
                "da6d3336-1744-40ac-939f-8d0ad8ec2acf",
                "90efd5c5-be4a-4a97-933d-a3af5cdb8673",
                "e3ac474f-e0bd-4c59-a9bd-37146acbb94f"
            ],
            "Жадный алгоритм добавления ребер минимального веса, не образующих цикла": [
                "a64408ad-ddb0-445a-acae-b3a070f1b6a7",
                "e3ac474f-e0bd-4c59-a9bd-37146acbb94f",
                "3b6ff823-09d0-416d-a2b9-4b75717c6f2d"
            ],
            "Использование DSU для проверки связности компонент $find-set(u) \\neq find-set(v)$": [
                "0d622e7f-f28a-4ca3-a598-8f34760f08a4",
                "c60cc6dc-ad3d-451b-a312-4a1598d6ee83"
            ]
        }
    },
    {
        "topic": "Динамическое программирование: задача о рюкзаке",
        "queries": {
            "Метод решения задачи о 0-1 рюкзаке через таблицу состояний": [
                "132d5ca4-ef29-48c1-8505-2103e9cce210",
                "63fc4a0f-3f44-45b6-86e9-31df7e3fa322",
                "4904f4d8-2879-45a1-b8f6-135e6749fd85"
            ],
            "Оптимизация выбора предметов с заданным весом и ценностью при ограниченной вместимости": [
                "693fd288-d42f-409f-bd95-9997ddf372bb",
                "9a28fb80-1a45-463e-86cb-8f1bf9929aa4",
                "79f7f8cf-5f54-472c-a1c8-5b48b784bfe5"
            ],
            "Рекуррентное соотношение $V[i, w] = \\max(V[i-1, w], v_i + V[i-1, w - w_i])$": []
        }
    },
    {
        "topic": "Точки сочленения в графе",
        "queries": {
            "Поиск шарниров и мостов в неориентированном графе.": [
                "1786f65b-f98c-4bf6-8124-d56fcaa6dfaf",
                "b1519ec3-3274-4e94-8e68-a76b0516a6a0",
                "670cc353-43ed-4e12-938b-1d7d36b1ed7d"
            ],
            "Вершины, удаление которых увеличивает число компонент связности.": [
                "2c93b80f-c884-47c7-8ac8-77f99c835765",
                "e7ef1165-7f5c-4074-91ec-62303b9a14f7",
                "b8614776-68cc-40ea-b5f5-d8f92d9b6e3b"
            ],
            "Условие $low[v] \\ge disc[u]$ при обходе в глубину (DFS).": [
                "f8aedfd6-1f77-450a-94c2-5b5005cf0461",
                "1da880cf-330d-4dce-b857-13ba79d1eccb",
                "b1f87c9c-662b-4b5a-bc78-4e299d08473d"
            ]
        }
    },
    {
        "topic": "Сортировка подсчетом (Counting Sort)",
        "queries": {
            "Линейный алгоритм сортировки целых чисел в ограниченном диапазоне.": [
                "fa161da1-1dfb-45da-9cf3-ec9e3474f439",
                "2a672d7c-b963-46c0-ab80-a88160fe82f1",
                "79217255-34cc-49b2-aebc-f08233bf982e"
            ],
            "Распределение элементов по индексам вспомогательного массива на основе их значений": [
                "22b1db4a-45c0-48df-980d-a1cca41c280d",
                "fc492428-1c42-438e-8f57-f2db47991e23"
            ],
            "Время работы алгоритма O(n + k), где k — диапазон значений": [
                "e65b1ce4-3ff3-4ab0-892b-461ca32a97b7",
                "b03918c7-7356-4068-9875-f58e339764b9",
                "36882260-f0ed-4493-ae31-40774d59d3c0",
                "eeffae15-4d99-4355-9081-c0233075b57a"
            ]
        }
    },
    {
        "topic": "Умножение матриц по Штрассену",
        "queries": {
            "Метод декомпозиции для быстрого перемножения квадратных матриц": [
                "00e98e4f-84e6-4338-8083-ecbb6f23ba50",
                "a78999fe-f53c-4577-aeba-fb95f82deea3",
                "935cff0a-2dba-4239-b06e-e7b39fb6cad2",
                "e910c145-df41-4efd-8092-fc558ce77085"
            ],
            "Рекурсивное вычисление произведения матриц с использованием семи умножений вместо восьми": [
                "6c202ad7-f80e-4121-93e0-689a99800c71",
                "29d2df1d-4710-42a8-83fb-0f8ea8581268",
                "117c22f5-850a-4445-b254-481423db8d18"
            ],
            "Оценка сложности алгоритма как $O(n^{\\log_2 7})$": []
        }
    },
    {
        "topic": "Деревья отрезков (Segment Trees)",
        "queries": {
            "Структура данных для выполнения групповых операций на интервалах": [
                "39bf58c0-cb23-403f-9d64-1ff1c21981c1",
                "f51eae51-e783-4eab-8340-5b1c06fa0e56",
                "5033e00c-de35-4323-98fe-69d52f0a4b1d"
            ],
            "Эффективное вычисление суммы или минимума на подотрезке массива": [
                "1e7fbc92-66c3-4d44-b85d-c6e9fcd1cd04",
                "bf6bd549-f5f7-49a3-9fe0-5e6285e2082c",
                "dd181e30-d3d9-4672-933b-c6ac7199994f"
            ],
            "Сложность запроса и обновления элемента $O(\\log n)$": [
                "e6d07ae3-64cd-495b-810f-1bfae81e8c79",
                "4128444a-db52-4dea-9a35-aa914fdea113",
                "9ef36029-d6ad-4830-9913-4640ebca8987",
                "9efabe81-6f29-42f9-bff1-b76c49c3e317"
            ]
        }
    },
    {
        "topic": "Коды Хаффмана",
        "queries": {
            "Алгоритм префиксного кодирования с минимальной избыточностью": [
                "d13c5d96-865b-40f0-a760-8331f2e11b65",
                "73d0fa5b-4aba-4d9d-9285-f299a1ece710",
                "00d45847-7a39-4cee-8874-01a004426b91"
            ],
            "Построение оптимального бинарного дерева на основе частот появления символов": [
                "d5274c0a-d39c-4268-aacd-22341758dec8",
                "02a2acfd-a1ec-4889-8b3f-87bbb8e691e6",
                "4e473032-7fd5-481b-b49f-dc7c8619460d",
                "7a6791f1-a56a-4eba-b070-38689057d993"
            ],
            "Минимизация функции взвешенной длины пути $L(C) = \\sum f_i \\cdot d_i$": [
                "9474ce92-7499-4a97-98f4-951e85e22186",
                "1f78ed2d-cacf-40f4-aec6-438ec70fac43",
                "3cf27e1e-cf5b-4b64-8049-726baffa0fc8"
            ]
        }
    },
    {
        "topic": "Сильно связные компоненты",
        "queries": {
            "Алгоритм Тарьяна или Косарайю для ориентированных графов": [
                "b066ac2d-b40b-4e0a-bc5a-1a6026673525"
            ],
            "Разбиение орграфа на максимальные подграфы, в которых любые две вершины взаимно достижимы": [
                "dbab9887-ed7c-4983-bb32-b32ce386eee9",
                "2c93b80f-c884-47c7-8ac8-77f99c835765",
                "650057fd-f79d-4bb9-990c-ec44b928cba2"
            ],
            "Использование инвертированного графа $G^T$ в алгоритме Косарайю": [
                "13fcf9a7-efe3-4347-a477-8eb3d99847f5",
                "80bea99a-4bdb-4fa2-a491-0b9a0dc8479c",
                "c9ce8d40-b561-4b5e-8506-52736f358070"
            ]
        }
    }
]


# Ретривер
def create_retriever(use_reranker: bool, top_k: int):
    """
    Создаёт и возвращает инстанс ретривера с заданными параметрами.
    """
    config = RetrieverConfig.from_yaml()

    if not use_reranker:
        config.reranker_score_threshold = None
        config.reranker_model_name = None
    config.top_k_final = top_k

    retriever = HybridRetriever(config)
    return retriever


def do_search(retriever, query: str) -> list[str]:
    """
    Выполняет поиск.
    """
    results = retriever.search(query)
    return [r.id for r in results.top_chunks]


# Метрики
def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_mode(
    dataset: list[dict],
    use_reranker: bool,
    k_values: list[int],
) -> dict:
    """
    Собирает метрики для каждого типа ретривера.
    """
    max_k = max(k_values)

    # Собираем все результаты поиска при max_k
    mode_label = "Reranker" if use_reranker else "без реранкера"
    print(f"    Создание ретривера (k={max_k}, {mode_label})", end="\t")
    retriever = create_retriever(use_reranker=use_reranker, top_k=max_k)
    print("✓")

    retrieved_cache: dict[str, list[str]] = {}
    for item in tqdm(dataset, 'Поиск по запросам'):
        for query in item["queries"].keys():
            retrieved_cache[query] = do_search(retriever, query)

    # Считаем метрики по всем запросам
    all_queries: list[dict] = []

    for item in dataset:
        topic_rrs = []

        for query, relevant in item["queries"].items():
            retrieved = retrieved_cache[query]
            rr = reciprocal_rank(retrieved, relevant)
            topic_rrs.append(rr)
            all_queries.append({
                "query":    query,
                "rr":       rr,
                "recall":   {k: recall_at_k(retrieved, relevant, k) for k in k_values},
                "ndcg":     {k: ndcg_at_k(retrieved, relevant, k)   for k in k_values},
                "precision": {k: precision_at_k(retrieved, relevant, k) for k in k_values},
            })

    n = len(all_queries)
    if n == 0:
        return {}

    return {
        "MRR":      {k: float(np.mean([q["rr"]        for q in all_queries])) for k in k_values},
        "Recall":   {k: float(np.mean([q["recall"][k] for q in all_queries])) for k in k_values},
        "NDCG":     {k: float(np.mean([q["ndcg"][k]   for q in all_queries])) for k in k_values},
        "Precision": {k: float(np.mean([q["precision"][k] for q in all_queries])) for k in k_values}
    }


# Сохранение
def save_json(no_rr: dict, with_rr: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    def clean(d: dict) -> dict:
        """Убираем per_topic для краткости верхнего уровня."""
        return {
            m: {str(k): round(v, 4) for k, v in vals.items()}
            for m, vals in d.items()
            if m != "per_topic"
        }

    data = {
        "hybrid_no_reranker": clean(no_rr),
        "hybrid_reranker":    clean(with_rr),
        "per_topic_no_rr":    no_rr.get("per_topic", []),
        "per_topic_with_rr":  with_rr.get("per_topic", []),
    }
    JSON_OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# Консольные таблицы
def print_metrics_table(no_rr: dict, with_rr: dict, k_values: list[int]) -> None:
    table = Table(title="Метрики поиска", box=box.ROUNDED,
                  header_style="bold magenta", show_lines=True)
    table.add_column("Режим", style="cyan", min_width=22)
    for metric in ["MRR", "Recall", "NDCG", "Precision"]:
        for k in k_values:
            lbl = "MRR" if metric == "MRR" else f"{metric}@{k}"
            table.add_column(lbl, justify="right", style="white")

    def row_vals(data: dict) -> list[str]:
        vals = []
        for metric in ["MRR", "Recall", "NDCG", "Precision"]:
            for k in k_values:
                vals.append(f"{data[metric][k]:.4f}")
        return vals

    table.add_row("Hybrid (без реранкера)", *row_vals(no_rr))
    table.add_row("Hybrid + Reranker",      *row_vals(with_rr))

    delta_vals = []
    for metric in ["MRR", "Recall", "NDCG", "Precision"]:
        for k in k_values:
            d    = with_rr[metric][k] - no_rr[metric][k]
            sign = "+" if d >= 0 else ""
            col  = "green" if d >= 0 else "red"
            delta_vals.append(f"[{col}]{sign}{d:.4f}[/{col}]")
    table.add_row("[bold]Δ (прирост)[/bold]", *delta_vals)

    console.print()
    console.print(table)


# Визуализация
BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#2E2E2E"
GRID_COLOR = "#E0E0E0"
BORDER_COLOR = "#BCBCBC"

COLOR_NO_RR = "#537785"
COLOR_WITH_RR = "#435C4E"
COLOR_NDCG_NO = "#8A9D91"
COLOR_NDCG_RR = "#1D3A4D"


def plot_results(no_rr: dict, with_rr: dict, k_values: list[int], report_k: int) -> None:
    plt.rcParams["font.family"] = "serif"
    fig = plt.figure(figsize=(18, 12), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    ks = sorted(k_values)

    # TOP LEFT
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG_COLOR)

    metric_keys = ["MRR", "Recall", "NDCG", "Precision"]
    metric_lbl  = ["MRR", f"Recall@{report_k}", f"NDCG@{report_k}", f"Precision@{report_k}"]
    x = np.arange(len(metric_keys))
    width = 0.32

    for offset, label in ((-width / 2, "Hybrid (без РР)"), (+width / 2, "Hybrid + Reranker")):
        data = no_rr if "без" in label else with_rr
        color = COLOR_NO_RR if "без" in label else COLOR_WITH_RR
        vals = [data[m][report_k] for m in metric_keys]

        bars = ax1.bar(x + offset, vals, width, label=label,
                       color=color, alpha=0.9, edgecolor="#444444", linewidth=0.6)
        for bar in bars:
            # Значения над столбцами выводим темно-серым цветом
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.015,
                     f"{bar.get_height():.3f}",
                     ha="center", va="bottom", fontsize=9,
                     color=TEXT_COLOR, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_lbl, color=TEXT_COLOR, fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Значение метрики", color=TEXT_COLOR, fontsize=11)
    ax1.set_title(f"Метрики при k={report_k}", color=TEXT_COLOR,
                  fontsize=13, fontweight="bold", pad=15)

    ax1.tick_params(colors=TEXT_COLOR)
    ax1.spines[:].set_color(BORDER_COLOR)
    ax1.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
    ax1.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, edgecolor=BORDER_COLOR, fontsize=10)
    ax1.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.6)

    # TOP RIGHT
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG_COLOR)

    ax2.plot(ks, [no_rr["Recall"][k] for k in ks], "o--",
             color=COLOR_NO_RR, linewidth=2, markersize=6, label="Recall (без РР)")
    ax2.plot(ks, [with_rr["Recall"][k] for k in ks], "o-",
             color=COLOR_WITH_RR, linewidth=2, markersize=6, label="Recall + RR")
    ax2.plot(ks, [no_rr["NDCG"][k] for k in ks], "s--",
             color=COLOR_NDCG_NO, linewidth=2, markersize=6, label="NDCG (без РР)")
    ax2.plot(ks, [with_rr["NDCG"][k] for k in ks], "s-",
             color=COLOR_NDCG_RR, linewidth=2, markersize=6, label="NDCG + RR")
    ax2.plot(ks, [no_rr["Precision"][k] for k in ks], "^--",
             color="#9FB0A2", linewidth=2, markersize=7, label="Precision (без РР)")
    ax2.plot(ks, [with_rr["Precision"][k] for k in ks], "^-",
             color="#3F5159", linewidth=2, markersize=7, label="Precision + RR")

    ax2.set_xticks(ks)
    ax2.set_xticklabels([str(k) for k in ks], color=TEXT_COLOR)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("k", color=TEXT_COLOR, fontsize=11)
    ax2.set_ylabel("Значение метрики", color=TEXT_COLOR, fontsize=11)
    ax2.set_title("Recall@k и NDCG@k", color=TEXT_COLOR,
                  fontsize=13, fontweight="bold", pad=15)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.spines[:].set_color(BORDER_COLOR)
    ax2.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
    ax2.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, edgecolor=BORDER_COLOR, fontsize=10)
    ax2.grid(color=GRID_COLOR, linestyle="--", linewidth=0.6)

    plt.savefig(PDF_OUT, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()


# Главная функция
def run() -> None:
    console.print(Panel("Тест модуля поиска", expand=False))

    n_topics  = len(DATASET)
    n_queries = sum(len(item["queries"]) for item in DATASET)
    print(f"Датасет: {n_topics} тем, {n_queries} запросов")
    print(f"k-значения: {K_VALUES}")

    print("Режим: без реранкера")
    no_rr_results = evaluate_mode(DATASET, use_reranker=False, k_values=K_VALUES)

    print("Режим: с реранкером")
    with_rr_results = evaluate_mode(DATASET, use_reranker=True, k_values=K_VALUES)

    print_metrics_table(no_rr_results, with_rr_results, K_VALUES)

    print("Сохранение результатов")
    save_json(no_rr_results, with_rr_results)

    plot_results(no_rr_results, with_rr_results, K_VALUES, REPORT_K)

    print("Тест ретривера завершен")


if __name__ == "__main__":
    run()
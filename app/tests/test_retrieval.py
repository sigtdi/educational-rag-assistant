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
            "Основная теорема о рекуррентных соотношениях метод декомпозиции": [],
            "Оценка сложности алгоритмов типа разделяй и властвуй": [],
            "Решение уравнения $T(n) = aT(n/b) + f(n)$": []
        }

    },
    {
        "topic": "Свойства красно-черных деревьев",
        "queries": {
            "Определение и пять условий красно-черного дерева.": [
                "18707283-ebff-4e15-a0ac-522d3cd62e61",
            ],
            "Сбалансированное бинарное дерево поиска с цветовыми метками узлов и черной высотой.": [
                "4ce59dba-0b4f-49c5-a370-27422f608542",
                "64bfe6c2-8ce4-4e4a-b1d8-43b52f9636b5"
            ],
            "Ограничение высоты дерева через количество узлов как $h \\le 2\\log(n+1)$.": [
                "4954807e-0a08-4216-ad88-158fb5e140dc",
                "320b0dbd-e8b2-456a-848e-ca17b12cb287"
            ]
        }
    },
    {
        "topic": "Алгоритм Кнута-Морриса-Пратта",
        "queries": {
            "Префикс-функция в алгоритме КМП.": [
                "3f2dfdae-9053-4ad1-8ac9-008c14f8e1ab",
                "aa4379d0-22c7-4a92-a153-aea1ab678f44",
                "2c1bf5b9-fb9a-4e5f-ab8f-ba0592b6007c",
                "c2184193-e9d6-42dc-a8b2-b0699d5d3d58",
                "5609d156-7e09-4ac1-9045-a7d2ffe7a885"
            ],
            "Поиск подстроки в строке с использованием таблицы сдвигов по префиксам.": [
                "ffbbaa5b-f624-4d4d-bca5-b8c279b20ca8",
                "69d582ae-3ca5-4268-8b5d-1f745ebf9e3b",
                "1c7660bc-6874-46fe-a82a-145f60225ec4"
            ],
            "Вычисление значений $\\pi[q] = \\max \\{k : k < q \\text{ и } P_k \\sqsupset P_q\\}$": [
                "a24ea2a5-c45e-45a5-ae6e-748dec97c122",
                "e0f94183-aefc-4ef6-b510-16ea27d63eb3",
                "b5c114f8-7c63-4e82-af2b-86a6f5a5fd86",
                "8defcc45-a327-4636-bf76-83e8c8ea651d",
                "325af13d-f363-4303-9f95-e800c2ba4c27"
            ]
        }
    },
    {
        "topic": "Сортировка кучей (Heapsort)",
        "queries": {
            "Алгоритм пирамидальной сортировки и свойства кучи.": [
                "c4816eb4-4804-4db2-b66a-354b69c06278",
                "ce358338-5d62-4110-b23e-2ec605b3b4b2",
                "d44b8303-bed7-497d-a4e3-fd913cc79f5d",
                "ed723a49-701b-47cb-97bc-8c75d7b0b1c7"
            ],
            "Поддержание основного свойства невозрастающего (или неубывающего) дерева в массиве.": [
                "e6efc93d-33d9-44c6-ac09-d5d6617b88a2"
            ],
            "Время работы процедуры MAX-HEAPIFY для узла на высоте h.": [
                "c7bf99a1-4c88-4f49-9458-670cba66ac0a",
                "1296b7e2-5b41-4ed2-8877-e17c24d9e2d2",
                "08ca5caf-8d66-4553-bb59-dedacf623858",
                "af5908ec-6f58-4877-8011-43bd0b3b28dc"
            ]
        }
    },
    {
        "topic": "Алгоритм Дейкстры",
        "queries": {
            "Поиск кратчайших путей из одной вершины в графе": [
                "4d0fd291-3ca3-4f1c-9a21-af30af24dcb8",
                "f7f45a03-f652-44f0-aaa3-ab02fe2d74dc"
            ],
            "Жадный алгоритм для нахождения минимального расстояния в графе с неотрицательными весами ребер": [
                "2652a9dd-66d0-4b93-43aa-8d563b61e9ca",
                "17dd0597-c84e-43ee-adf7-2899f463498b"
            ],
            "Релаксация ребра $(u, v)$ через условие $d[v] > d[u] + w(u, v)$.": [
                "897dbef8-f8c0-4a38-bd2b-72cd1fa1c813",
                "0d7a3284-9e97-40c9-aa03-e9733d62d9fb",
                "516467fc-0f64-493b-9082-a524c47914fd",
                "4b5e28e5-24ad-4041-ae8a-7e4654035bfe"
            ]
        }
    },
    {
        "topic": "Расстояние Левенштейна",
        "queries": {
            "Редакционное расстояние между строками и операции редактирования.": [
                "da1f8ce3-e697-4d47-98d0-3bb159a86aa5",
                "d2184082-bbe8-4a36-8a32-5209a1e17d15",
                "308f0aae-067f-44a6-8914-768a811eb059",
                "4e5c98ee-efc2-4c32-bb65-cdf10849b0e9",
                "48a8813f-0c58-4a8e-9103-e99c8716a78f",
            ],
            "Минимальное количество замен, вставок и удалений для трансформации одной последовательности в другую.": [
                "da1f8ce3-e697-4d47-98d0-3bb159a86aa5",
                "b21ba271-ac31-4b35-9c45-1b7f9537f1f8"
            ],
            "Формула динамического программирования $D(i, j) = \\min \\{D(i-1, j)+1, D(i, j-1)+1, D(i-1, j-1)+m(a_i, b_j)\\}$.": [
                "b944b163-f708-4b1f-a373-f18a354fc0f8",
                "29e174ae-6e5f-4452-940c-6ef97073a370",
                "bafe163f-4146-4b77-8c52-6f6dfc8b014a"
            ]
        }
    },
    {
        "topic": "Хеширование с открытой адресацией",
        "queries": {
            "Методы разрешения коллизий при помощи линейного или квадратичного исследования.": [
                "946ca39e-f521-470a-b6ee-bf05874d70f8",
                "ee8281cb-aed4-408a-88a0-3caa7c2d54ef"
            ],
            "Заполнение хеш-таблицы без использования связанных списков.": [
                "df04302c-e15d-4fd8-8c28-b69700a916c8",
                "ae75e075-eb56-48d0-8d64-11d96d70c042",
                "f380e81f-f99a-4349-8fd5-83fe2f8b746d"
            ],
            "Функция пробирования вида $h(k, i) = (h'(k) + i) \\pmod m$.": [
                "d03eb278-39df-4d9f-bc4a-20a85e265999",
                "784876c5-6094-4621-bd63-7d429f45e495"
            ]
        }
    },
    {
        "topic": "Алгоритм Форда-Фалкерсона",
        "queries": {
            "Нахождение максимального потока в транспортной сети.": [
                  "24bcdd52-5f4e-49a9-87e8-ea32994eabbd",
                  "2c4e593a-0749-478c-8be9-cdf73384c3ba",
                  "cc42e333-43f6-425c-8e3c-817318a5b04a",
                  "c6ae4c98-2b6c-43ba-9f92-eca09e524094"
            ],
            "Метод увеличивающих путей и остаточных сетей в графе.": [
                "92e91436-84ab-4695-bbe1-98efee171ba0",
                "931b24a5-51a5-4043-992e-57a69f1fd163",
                "563bb565-3bcc-4cdf-bbeb-f236e53d55e3",
                "25454a23-d857-40a7-8c90-92a4d788689c"
            ],
            "Теорема о максимальном потоке и минимальном разрезе $|f| = c(S, T)$": [
                "cc42e333-43f6-425c-8e3c-817318a5b04a",
                "2c4e593a-0749-478c-8be9-cdf73384c3ba",
                "af4f6609-d67e-454f-b14b-a3415150984f",
                "ed4b681b-6673-4fb9-b430-02ae155f7b81",
                "345d170b-983c-461e-acd7-b90e93c4ccb1"
            ]
        }
    },
    {
        "topic": "B-деревья",
        "queries": {
            "Структура и свойства B-дерева для внешних систем памяти.": [
                "c75f3c1e-4440-4a59-9291-61213724422c",
                "8f9b2956-f9a1-4746-95ee-24b929371bf1",
                "97a9968c-b9c3-4d44-9765-350ea9f0f829"
            ],
            "Многоходовое сбалансированное дерево поиска с заданным минимальным ветвлением.": [
                "09884a0d-6e52-4201-8c00-284b660670e8",
                "626d231b-1977-491a-acd4-a14bb517e4a2",
                "d67d0cfd-5e50-4922-8427-e141e68db8c2"
            ],
            "Условие на количество ключей в узле $t-1 \\le n[x] \\le 2t-1$.": [
                "78e910a2-dbea-4c4a-a48a-d24f3df578a2",
                "7295bbee-e3c0-4514-808c-9dd80688b567"
            ]
        }
    },
    {
        "topic": "Алгоритм Беллмана-Форда",
        "queries": {
            "Поиск кратчайших путей в графах с отрицательными весами ребер.": [
                "d824c247-3358-4e04-b4d6-8b9297833a68",
                "d71aaa9b-f0ae-4cc8-829a-6b63c1626e79",
                "ca402020-650f-4ab3-aec0-3158b1efcfb3",
                "5e4c5be3-f3ed-4176-b5fa-005be91d0612"
            ],
            "Алгоритм обнаружения циклов отрицательного веса из заданного источника.": [
                "3612384f-0b1b-4a93-a9d6-5c7c441ff437",
                "a9809064-3d48-4bcb-a11b-5d924573951c",
                "3f1bf90f-b3a5-4084-b099-f8ca3db0c98c",
                "8e1b77cf-5fef-4785-bcc0-e4f45d50e335"
            ],
            "Итеративная проверка условия $d[v] \\le d[u] + w(u, v)$ для всех $E$ ребер.": [
                "d71aaa9b-f0ae-4cc8-829a-6b63c1626e79",
                "f74016d9-eeaf-4eed-a571-db50b69dfb3a"
            ]
        }
    },
    {
        "topic": "Суффиксные деревья",
        "queries": {
            "Построение суффиксного дерева для строки.": [
                "9037f7cf-c1ae-42cf-9d07-607b4144e7a7",
                "481b7d27-0ebf-4c7e-b3e1-32ae1e3772b5",
                "79f6d72d-fb4e-41ba-a50d-befcc36f9218",
                "65d61eee-d075-4bd2-bb5a-72715de4d6ca"
            ],
            "Сжатое дерево всех суффиксов заданной последовательности символов.": [
                "16e2c0fc-b421-4133-9234-24e52cbd83d7",
                "d40a82aa-4fc4-4e44-a25f-0a029b527dc0",
                "d245a7dd-1ca7-4456-ae7b-b1357ddaefb8"
            ],
            "Алгоритм Укконена для построения структуры за время O(n).": [
                "c0517af0-4114-4390-affc-15e5b8b71806",
                "14b63f14-21b5-4da9-b6b1-1a481135f1de",
                "1ecdfdab-fa89-4082-8b19-3990eaaadf6e",
                "02ac13bc-566d-464d-a56a-d575b72b31c6"
            ]
        }
    },
    {
        "topic": "Биномиальные кучи",
        "queries": {
            "Определение и операции над биномиальными очередями с приоритетами.": [
                "0f88d590-81ac-498d-bbf2-af1f8ae28b0",
                "ef4656e2-ca26-4d5c-945e-2ba51d61cbac",
                "e5672354-0350-431b-8aa3-edf95bde78d5"
            ],
            "Объединение набора биномиальных деревьев с логарифмическим временем работы.": [
                "d027b802-4c00-4535-8dd0-b055c1572000",
                "c58c37cc-0ac4-44af-9671-343fb6c73121"
            ],
            "Количество узлов в дереве $B_k$ равное $2^k$.": [
                "ce5ba67d-a75c-4481-b8c3-3b1984c30369",
                "28bcba6b-633e-45ab-a773-5636b00e4056"
            ]
        }
    },
    {
        "topic": "Алгоритм Флойда-Уоршелла",
        "queries": {
            "Нахождение кратчайших путей между всеми парами вершин графа.": [
                "6ae36437-df30-4ae6-84bf-5589a2fad223",
                "b3250192-03ca-4e86-aac4-4e4c1aa5dac1",
                "70cfb6cd-7ac3-4fc8-aa36-e605f232b817",
                "cd4de65b-1cdf-474e-8929-34296022cc01"
            ],
            "Метод динамического программирования для вычисления матрицы расстояний.": [
                "f7ece088-bba0-4633-a44a-30e65c5c1c9f",
                "292dab40-15ed-4796-9e8e-220155359b05",
                "dfc55302-c5ad-4ff6-893f-d205a5600a60"
            ],
            "Обновление значений по формуле $d_{ij}^{(k)} = \\min(d_{ij}^{(k-1)}, d_{ik}^{(k-1)} + d_{kj}^{(k-1)})$.": [
                "fe3d6bc2-8c3c-4b5b-b0aa-3b0d15df0401",
                "de0d849c-28c6-40c6-ab4d-a86c83df64d6"
            ]
        }
    },
    {
        "topic": "Амортизационный анализ",
        "queries": {
            "Метод потенциалов и бухгалтерский метод оценки сложности.": [
                "4db11826-b853-45c2-8fa1-ae625706d19b",
                "39a1f4b8-845c-46b3-a7d0-5738575a5fe9",
                "1eb5b9dc-c151-4784-9a89-759bac7569ab",
                "d4b80698-afe2-426b-8fab-a38799750e40"
            ],
            "Среднее время выполнения последовательности операций в худшем случае.": [
                "7edfb1a5-4b5c-41d5-b3cb-3d65bd5541ee",
                "06a39141-a93d-4d34-bfdf-24fe6e2be08c",
                "a469ce43-e7a8-4025-9c59-cff870552348"
            ],
            "Определение амортизированной стоимости как $\\hat{c}_i = c_i + \\Phi(D_i) - \\Phi(D_{i-1})$.": [
                "a355dcbb-f298-4c4c-bf1b-00319dc6f1e5",
                "20ffd8de-607a-42a3-b009-e078f1e26d59",
                "36f9a46e-a42a-4c8f-a8b6-420567381eda"
            ]
        }
    },
    {
        "topic": "Алгоритм Хоара (Quicksort)",
        "queries": {
            "Метод быстрой сортировки с выбором опорного элемента.": [
                "766c97be-adec-4ecf-b223-82465e1ca00b",
                "d0996933-564b-4634-b6a2-eb9d8e78acd9",
                "2928225f-a8ea-47ae-9e77-722322cd7497"
            ],
            "Разделение массива на две части относительно пивота (partitioning)": [
                "45eca0ac-2566-46b3-a263-c56b8ba42293",
                "6cf0c5eb-9bac-4f9d-87a7-25f85ff45742",
                "eab33d7b-0cc1-440f-b0b8-fa2e8ed82393"
            ],
            "Математическое ожидание времени работы при случайном выборе $E[T(n)] = O(n \\log n)$": [
                "e1e6753c-4f23-45bb-8b51-18eb58a525e3",
                "79b65f2c-0b1b-4719-93ee-36165f80d4b4",
                "3de727f1-6604-4821-bfd2-e06e3ffcd138"
            ]
        }
    },
    {
        "topic": "Система непересекающихся множеств (DSU)",
        "queries": {
            "Структура данных 'Лес непересекающихся множеств'.": [
                "f195774d-642e-4d8f-8e5f-d6d326f2a116",
                "989d1398-9bf4-4c40-b754-24aed66ec18e"
            ],
            "Операции объединения по рангу и сжатия путей.": [
                "5d9e0d42-1ea8-4e03-b23d-2e174bbf9663",
                "7119a3d4-0899-46bc-b244-fcc25ea72956",
                "dc0ce037-cb50-469e-835a-9b745dfbfb21"
            ],
            "Оценка сложности через обратную функцию Аккермана $\\alpha(n)$.": [
                "9bbd2691-0031-4cc6-967e-56eb6f626d4c",
                "39908279-9c54-4e1a-89c0-2f618d5d6365"
            ]
        }
    },
    {
        "topic": "NP-полнота",
        "queries": {
            "Определение класса NP-полных задач и полиномиальная сводимость.": [
                "5d7f0ab6-152e-45e8-9557-7fc3c622175a",
                "799a052d-2f52-418e-8395-6292c21e3be9",
                "1c71cc66-54d3-4ffb-b727-ec897e6168f5"
            ],
            "Задачи, к которым сводится любая задача из класса NP за полиномиальное время?": [
                "aa0093dd-aaca-4997-be0e-24c740053049",
                "f8ff1468-65c0-4151-a918-8a07a09db75c",
                "1c71cc66-54d3-4ffb-b727-ec897e6168f5"
            ],
            "Теорема Кука-Левина о выполнимости булевых формул ($SAT$).": [
                "8d681b5e-62b3-4f3a-830b-90a4de2642e6",
                "8f92ba88-21be-4cb0-af86-1263060c9fa3",
                "78dca4f0-5fee-43c1-9956-15b5844fd619"
            ]
        }
    },
    {
        "topic": "Алгоритм Бойера-Мура",
        "queries": {
            "Поиск подстроки с использованием эвристики «плохого символа» и «хорошего суффикса»": [
                "6a0ae51c-f4f0-491c-bf8e-715329d6562a",
                "5a36671d-5aa0-4c18-a426-baeca04e8aaf",
                "b49856a9-571d-4054-92ed-348c94a9fd4c"
            ],
            "Алгоритм быстрого сопоставления строк путем сканирования символов справа налево": [
                "0529e255-9f21-40c7-9b24-640a2b9be4a2",
                "f0fb8f99-5e1a-4c6a-a582-7ac02ef129e8",
                "f62477e1-a7d8-4661-98c5-8bc21025eddf",
                "788e51af-4d1f-4f0d-a430-ee92ebd27ace"
            ],
            "Сдвиг шаблона на основе функции $\\gamma(j)$  и таблицы стоп-символов": [
                "0c268a27-6cd2-4068-b4e6-e36f5cdf6eea",
                "3347c96f-d39e-4a24-8f87-f711c00d7296",
                "6ac73851-3c59-474e-9865-324ce4fd1b72",
                "f8ed94f3-2857-4b6d-beeb-29d02308fb25"
            ]
        }
    },
    {
        "topic": "Генерация перестановок",
        "queries": {
            "Алгоритмы комбинаторной генерации всех перестановок множества": [
                "1f5c94ae-4230-4969-be79-781817420d09",
                "0a14b713-4003-4e87-af61-261a9f0efc60"
            ],
            "Построение лексикографического порядка последовательностей элементов": [
                "da967adc-b998-4d53-b7f5-53b438f0db10",
                "82d70672-a613-43f1-a829-39c90f15b6ce",
                "7581c571-c0fd-4127-877e-1a57c4d2eb60",
                "f16337a3-ca10-40ac-8317-4809c8a9c1d2"
            ],
            "Формула общего количества перестановок для n элементов: n!": [
                "0a14b713-4003-4e87-af61-261a9f0efc60",
                "5080fcd3-11ed-46fc-8b55-b18a5394f16c",
                "8a939562-d2c7-42e2-98a2-b9ae14a7e06e"
            ]
        }
    },
    {
        "topic": "Алгоритм Краскала",
        "queries": {
            "Построение минимального остовного дерева на основе сортировки ребер": [
                "43b3b8cd-8b74-4b48-a821-ace8fe55a34f",
                "4e5c8efc-7f81-4d8d-8b30-8665bb9ba4f4"
            ],
            "Жадный алгоритм добавления ребер минимального веса, не образующих цикла": [
                "5477a3f9-bfb3-4d51-bece-06f27ec75069",
                "34b0b23f-aa04-4449-80f5-2e073044acb0",
                "b783982a-4c61-44c4-bc04-bfe60037675e",
                "64ba70d8-22b1-4d37-9f6c-1fc7f6647640"
            ],
            "Использование DSU для проверки связности компонент $find-set(u) \\neq find-set(v)$": [
                "84b924a4-5cec-49c0-b47d-d06aea8c0429",
                "9a49ddc1-a57d-4967-bc2d-7a505e6965af",
                "cb4e7864-b1e7-47c0-aed7-a90a55d35d79"
            ]
        }
    },
    {
        "topic": "Динамическое программирование: задача о рюкзаке",
        "queries": {
            "Метод решения задачи о 0-1 рюкзаке через таблицу состояний": [
                "0c9e5f68-e927-49c9-a67c-a0509fe72eed",
                "c65aba13-e256-4a6a-bc1b-68fad3362899"
            ],
            "Оптимизация выбора предметов с заданным весом и ценностью при ограниченной вместимости": [
                "0c9e5f68-e927-49c9-a67c-a0509fe72eed",
                "c65aba13-e256-4a6a-bc1b-68fad3362899",
                "5ebc5fe1-6ebd-425d-bb3c-5cb4fc8bede4"
            ],
            "Рекуррентное соотношение $V[i, w] = \\max(V[i-1, w], v_i + V[i-1, w - w_i])$": [
                "0c9e5f68-e927-49c9-a67c-a0509fe72eed"
            ]
        }
    },
    {
        "topic": "Точки сочленения в графе",
        "queries": {
            "Поиск шарниров и мостов в неориентированном графе.": [
                "1ccff4f5-9102-4f78-b011-28ca4c3ca467",
                "d657eb6a-eee2-4554-b143-b9df01149343",
                "c3e16d3f-1cf8-4a47-ab95-708604627a78",
                "6b6695f5-952d-48ba-942e-bf10f9e0878a",
                "4aed74b8-d5a4-4f15-8a78-4613684ab4ec"
            ],
            "Вершины, удаление которых увеличивает число компонент связности.": [
                "ed69cd05-4e82-4189-8dcd-cc10358a5b76",
                "6b6695f5-952d-48ba-942e-bf10f9e0878a",
                "4aed74b8-d5a4-4f15-8a78-4613684ab4ec"
            ],
            "Условие $low[v] \\ge disc[u]$ при обходе в глубину (DFS).": [
                "c3e16d3f-1cf8-4a47-ab95-708604627a78",
                "4aed74b8-d5a4-4f15-8a78-4613684ab4ec"
            ]
        }
    },
    {
        "topic": "Сортировка подсчетом (Counting Sort)",
        "queries": {
            "Линейный алгоритм сортировки целых чисел в ограниченном диапазоне.": [
                "06c42d79-b38c-48a9-8eef-87ee5534bdf3",
                "0a7de7e4-5825-485b-9d45-8dec552ce463",
                "0482dea2-9bd1-4ac2-a501-c64093c6c04c",
                "309e9d33-9b68-4cc1-b5b6-a520e109b2af",
                "cd0417d7-c91f-4023-929b-1730c90b5a15",
                "99c70cb4-287d-451c-83b0-c28eb0c7fcba"
            ],
            "Распределение элементов по индексам вспомогательного массива на основе их значений": [
                "06c42d79-b38c-48a9-8eef-87ee5534bdf3",
                "0a7de7e4-5825-485b-9d45-8dec552ce463",
                "309e9d33-9b68-4cc1-b5b6-a520e109b2af",
                "cd0417d7-c91f-4023-929b-1730c90b5a15",
                "99c70cb4-287d-451c-83b0-c28eb0c7fcba",
                "f5e23b10-837a-47b0-bb51-c23952c5ca11",
                "7dd78a58-fca7-4efc-8871-af16751dc621"
            ],
            "Время работы алгоритма O(n + k), где k — диапазон значений": [
                "0482dea2-9bd1-4ac2-a501-c64093c6c04c",
                "309e9d33-9b68-4cc1-b5b6-a520e109b2af",
                "f5e23b10-837a-47b0-bb51-c23952c5ca11"
            ]
        }
    },
    {
        "topic": "Умножение матриц по Штрассену",
        "queries": {
            "Метод декомпозиции для быстрого перемножения квадратных матриц": [
                "9038abc3-d219-4592-9f69-5c265332d156",
                "657b12f6-beaa-4a4f-b5f4-d3c73b51c590",
                "d540eefd-e395-468e-858e-16863a821732"
            ],
            "Рекурсивное вычисление произведения матриц с использованием семи умножений вместо восьми": [
                "9038abc3-d219-4592-9f69-5c265332d156",
                "657b12f6-beaa-4a4f-b5f4-d3c73b51c590"
            ],
            "Оценка сложности алгоритма как $O(n^{\\log_2 7})$": [
                "d540eefd-e395-468e-858e-16863a821732"
            ]
        }
    },
    {
        "topic": "Деревья отрезков (Segment Trees)",
        "queries": {
            "Структура данных для выполнения групповых операций на интервалах": [
                "ef201013-bede-49ad-9c09-49621f81ae82",
                "fa3f97d5-73f2-4b2e-80df-8e36bd3c9b8a",
                "1bfc659a-a0d3-4052-8379-c6d399e80ea1",
                "a5e3b9b8-01fa-458a-8123-0eca4cc52989",
                "54562795-b98b-416d-8c48-a5616751a41e"
            ],
            "Эффективное вычисление суммы или минимума на подотрезке массива": [
                "ef201013-bede-49ad-9c09-49621f81ae82",
                "54562795-b98b-416d-8c48-a5616751a41e"
            ],
            "Сложность запроса и обновления элемента $O(\\log n)$": [
                "fa3f97d5-73f2-4b2e-80df-8e36bd3c9b8a",
                "a5e3b9b8-01fa-458a-8123-0eca4cc52989"
            ]
        }
    },
    {
        "topic": "Коды Хаффмана",
        "queries": {
            "Алгоритм префиксного кодирования с минимальной избыточностью": [
                "98b50985-4a4c-4a92-96be-7e16fcfa5606",
                "c91e1e65-64fc-4c31-8ea5-a9eb78388a4d",
                "f02d2ca7-d432-408d-b270-0ac85643824e",
                "b93f00e3-e4d7-430b-830b-381655aa7146",
                "14742ac4-b28a-44ca-a6a4-28df92b06172",
                "00f66cad-11fd-4150-9e4d-06baecbbc217"
            ],
            "Построение оптимального бинарного дерева на основе частот появления символов": [
                "c91e1e65-64fc-4c31-8ea5-a9eb78388a4d",
                "f02d2ca7-d432-408d-b270-0ac85643824e",
                "b93f00e3-e4d7-430b-830b-381655aa7146",
                "14742ac4-b28a-44ca-a6a4-28df92b06172",
                "00f66cad-11fd-4150-9e4d-06baecbbc217"
            ],
            "Минимизация функции взвешенной длины пути $L(C) = \\sum f_i \\cdot d_i$": [
                "98b50985-4a4c-4a92-96be-7e16fcfa5606",
                "c91e1e65-64fc-4c31-8ea5-a9eb78388a4d",
                "f02d2ca7-d432-408d-b270-0ac85643824e"
            ]
        }
    },
    {
        "topic": "Сильно связные компоненты",
        "queries": {
            "Алгоритм Тарьяна или Косарайю для ориентированных графов": [
                "c1e7f80e-fe98-4ca8-a947-6f5ea98a1934",
                "763ab358-ca5c-4ca8-8959-ef1e1f383e98"
            ],
            "Разбиение орграфа на максимальные подграфы, в которых любые две вершины взаимно достижимы": [
                "c1e7f80e-fe98-4ca8-a947-6f5ea98a1934",
                "1bd2a637-7d51-4802-aaf9-3e1e97b9c634"
            ],
            "Использование инвертированного графа $G^T$ в алгоритме Косарайю": [
                "c1e7f80e-fe98-4ca8-a947-6f5ea98a1934",
                "763ab358-ca5c-4ca8-8959-ef1e1f383e98"
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
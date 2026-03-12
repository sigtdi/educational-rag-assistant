import re
from app.logger_setup import log

def answer_fixer(ai_message):
    json_content = ai_message.content.strip()

    try:
        start_pattern = re.search(r'"text"\s*:\s*"', json_content)
        start_idx = start_pattern.end()
        end_idx = json_content.rfind('"')
        text_content = json_content[start_idx:end_idx]

        text_content = latex_fixer(text_content)
        text_content = backslash_fixer(text_content)
        text_content = wrap_latex(text_content)

        fixed_json = '{"text": "' + text_content + '"}'
        return fixed_json

    except Exception as e:
        log.errror(f"Критическая ошибка парсинга: {e}")
        return json_content

def backslash_fixer(text):
    # Заменяем более 2 идущих подряд слешей на 2
    # Для случая, если vlm все же какие-то из них экранирует
    text = re.sub(r'\\{2,}', r'\\\\', text)
    n_latex_dict = ['nabla', 'natural', 'ne', 'ni', 'noalign', 'nocite', 'nofiles', 'noindent', 'nolinebreak',
                    'nolimits', 'nonumber', 'nonfrenchspacing', 'nonstopmode', 'nopagebreak', 'normal', 'not', 'nu',
                    'numberline', 'nwarrow']

    def transform(match):
        slashes = match.group(1)  # Сами слеши
        tail = match.group(2)  # Текст после слешей

        # Если это два слеша и после них пробел, то делаем четыре
        if slashes == r'\\' and tail.isspace():
            return r'\\\\' + tail

        # Если это два слеша и после них не кавычка и не \n, то оставляем два слеша, иначе заменяем на 1 для проверки на latex
        if slashes == r'\\' and not (tail.startswith('"') or tail.startswith("'") or tail.startswith('n')):
            return slashes + tail
        else:
            slashes = '\\'

        # Если это один слеш и после него не кавычка\перенос строки, то делаем два
        # Следим, чтобы \n было не началом latex команды, а именно переносом строки
        if slashes == '\\':
            if not (tail.startswith('"') or tail.startswith("'") or tail.startswith('n')) or any(tail.startswith(cmd) for cmd in n_latex_dict):
                return r'\\' + tail
            else:
                return slashes + tail

        return slashes + tail

    # Ищем последовательности из 1 или 2 слешей и один любой символ после них
    fixed_text = re.sub(r'(\\{1,2})(\w+|.)', transform, text, flags=re.DOTALL)

    return fixed_text

def latex_fixer(text):
    # Стандартизируем скобки
    text = text.replace(r'\\[', '$$').replace(r'\\]', '$$')
    text = text.replace(r'\\(', '$').replace(r'\\)', '$')
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
    text = text.replace(r'\(', '$').replace(r'\)', '$')

    # Чистим пробелы внутри формул
    text = re.sub(r'\$\$\s*(.*?)\s*\$\$', lambda m: f"$${m.group(1).strip()}$$", text, flags=re.DOTALL)
    text = re.sub(r'\$\s*(.*?)\s*\$', lambda m: f"${m.group(1).strip()}$", text, flags=re.DOTALL)

    # Убираем лишние пробелы в тексте
    text = re.sub(r'(?<!^)[ ]{2,}', ' ', text, flags=re.MULTILINE)

    return text.strip()

def wrap_latex(text):
    clean_text = ""
    ranges = []
    last_idx = 0

    # Извлекаем существующие формулы и получаем текст без $
    pattern = re.compile(r'(\${1,2})(.*?)(\1)', re.DOTALL)

    for m in pattern.finditer(text):
        clean_text += text[last_idx:m.start()]
        start_in_clean = len(clean_text)
        content = m.group(2)
        clean_text += content
        end_in_clean = len(clean_text)
        ranges.append([start_in_clean, end_in_clean])
        last_idx = m.end()

    clean_text += text[last_idx:]

    # Ищем latex команды в чистом тексте
    pattern = re.compile(r'\\{2,}[^\s,.!?;]*')
    for a in pattern.finditer(clean_text):
        ranges.append(list(a.span()))

    if not ranges:
        return text

    # Функция для слияния пересекающихся отрезков
    def merge_intervals(intervals):
        if not intervals: return []
        intervals.sort()
        merged = [intervals[0]]
        for curr in intervals[1:]:
            prev = merged[-1]
            if curr[0] <= prev[1]:  # Пересекаются или касаются
                prev[1] = max(prev[1], curr[1])
            else:
                merged.append(curr)
        return merged

    # Расширение от опорных точек
    expanded_ranges = []
    for r_start, r_end in ranges:
        # Расширение влево
        while r_start > 0:
            prefix = clean_text[:r_start].rstrip(' ,')
            if not prefix:
                break

            match = re.search(r'[^\s,]+$', prefix)
            if not match:
                break

            word = match.group(0)
            if re.search(r'[.!?]', word):
                break

            if (re.search(r'[_^0-9=+\-*/<>\\{}\[\]]', word) or
                    re.match(r'^[a-zA-Z]{1,5}$', word)):
                r_start = clean_text.rfind(word, 0, r_start)
            else:
                break

        # Расширение вправо
        while r_end < len(clean_text):
            suffix = clean_text[r_end:].lstrip(' ,')
            if not suffix:
                break

            match = re.search(r'^[^\s,]+', suffix)
            if not match:
                break

            word = match.group(0)
            if re.search(r'[.!?]', word):
                break

            if (re.search(r'[_^0-9=+\-*/<>\\{}\[\]]', word) or
                    re.match(r'^[a-zA-Z]{1,5}$', word)):
                r_end = clean_text.find(word, r_end) + len(word)
            else:
                break

        expanded_ranges.append([r_start, r_end])

    # Сливаем границы перед балансировкой
    current_ranges = merge_intervals(expanded_ranges)

    # Выполняем балансировку скобок (), [], {}
    final_ranges = []
    for r_start, r_end in current_ranges:
        while True:
            content = clean_text[r_start:r_end]
            # Считаем разницу в количестве открывающих и закрывающих скобок
            balances = {
                'round': content.count('(') - content.count(')'),
                'square': content.count('[') - content.count(']'),
                'curly': content.count('{') - content.count('}')
            }

            changed = False
            # Если закрывающих больше расширяем влево
            if any(b < 0 for b in balances.values()) and r_start > 0:
                r_start -= 1
                changed = True
            # Если открывающих больше расширяем вправо
            if any(b > 0 for b in balances.values()) and r_end < len(clean_text):
                r_end += 1
                changed = True

            if not changed: break
        final_ranges.append([r_start, r_end])

    # Повторно сливаем после балансировки
    final_ranges = merge_intervals(final_ranges)

    # Оборачиваем найденные формулы в $
    result = []
    last_pos = 0
    for s, e in final_ranges:
        # Текст до формулы
        result.append(clean_text[last_pos:s])
        # Сама формула (убираем лишние пробелы по краям для красоты)
        formula = clean_text[s:e].strip()
        if formula:
            result.append(f"${formula}$")
        last_pos = e

    result.append(clean_text[last_pos:])

    # Объединяем текст и убираем двойные пробелы
    final_output = "".join(result)
    final_output = re.sub(r'\$\s+\$', ' ', final_output)

    return final_output
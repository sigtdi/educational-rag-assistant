import re

def answer_fixer(ai_message):
    text = ai_message.content
    text = latex_fixer(text)
    text = backslash_fixer(text)
    return text

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

        # Если это два слеша и после них не кавычка, то оставляем два слеша
        if slashes == r'\\' and not (tail.startswith('"') or tail.startswith("'")):
            return slashes + tail

        # Если это один слеш и после него не кавычка\перенос строки, то делаем два
        # Следим, чтобы \n не было началом latex команды, а именно переносом строки
        if slashes == '\\':
            if not (tail.startswith('"') or tail.startswith("'") or tail.startswith('n')) or any(tail.startswith(cmd) for cmd in n_latex_dict):
                return r'\\' + tail
            else:
                return slashes + tail

        # Для всех остальных случаев
        return slashes + tail

    # Ищем последовательности из 1 или 2 слешей и один любой символ после них
    fixed_text = re.sub(r'(\\{1,2})(.)', transform, text, flags=re.DOTALL)

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


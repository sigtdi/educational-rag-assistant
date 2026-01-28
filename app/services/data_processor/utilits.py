import re

def selective_backslash_fixer(ai_message):
    text = ai_message.content
    # Заменяем более 2 идущих подряд слешей на 2
    # Для случая, если vlm все же какие-то из них экранирует
    text = re.sub(r'\\{2,}', r'\\\\', text)

    def transform(match):
        slashes = match.group(1)  # Сами слеши
        char_after = match.group(2)  # Символ сразу после них

        # Если это два слеша и после них пробел, то делаем четыре
        if slashes == r'\\' and char_after.isspace():
            return r'\\\\' + char_after

        # Если это два слеша и после них не кавычка, то оставляем два слеша
        if slashes == r'\\' and char_after not in ['"', "'"]:
            return slashes + char_after

        # Если это один слеш и после него не кавычка, то делаем два
        if slashes == '\\':
            if char_after not in ['"', "'"]:
                return r'\\' + char_after
            else:
                return slashes + char_after

        # Для всех остальных случаев
        return slashes + char_after

    # Ищем последовательности из 1 или 2 слешей и один любой символ после них
    fixed_text = re.sub(r'(\\{1,2})(.)', transform, text, flags=re.DOTALL)

    return fixed_text
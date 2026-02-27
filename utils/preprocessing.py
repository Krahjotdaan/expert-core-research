import sys
try:
    import pymorphy3
    sys.modules['pymorphy2'] = pymorphy3
except ImportError:
    pass

import os
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
LEMMATIZED_DIR = os.path.join(BASE_DIR, "lemmatized")

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

def preprocess_and_lemmatize(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    # Список частей речи, которые нужно удалить
    REMOVE_POS = {
        'PREP',   # Предлоги
        'CONJ',   # Союзы
        'PRCL',   # Частицы
        'INTJ',   # Междометия
        'PNCT',   # Знаки препинания
        'PRON',   # Местоимения
    }

    filtered_words = []

    for token in doc.tokens:
        # Пропускаем, если токен не имеет морфологической информации
        if not hasattr(token, 'pos') or token.pos is None:
            continue

        # Если часть речи — оценочная — пропускаем
        if token.pos in REMOVE_POS:
            continue

        # Сохраняем лемму (нормальную форму)
        try:
            token.lemmatize(morph_vocab)
            lemma = getattr(token, 'lemma', token.text.lower())
            if lemma and lemma.strip():
                filtered_words.append(lemma)
        except:
            continue

    return ' '.join(filtered_words)


def apply_preprocessing(splits):
    print("Запуск лемматизации и очистки...\n")

    tqdm.pandas()
    for name, df in splits.items():
        print(f"Обработка сплита: {name}")
        df['text_lemmatized'] = df['text'].progress_apply(preprocess_and_lemmatize)
        df_lemmatized = df.drop(columns=['text'])

        if not os.path.exists(LEMMATIZED_DIR):
            os.makedirs(LEMMATIZED_DIR)

        print(f"    Сохранение в lemmatized/{name}_lemmatized.csv...")
        df_lemmatized.to_csv(os.path.join(LEMMATIZED_DIR, f"{name}_lemmatized.csv"), index=False)

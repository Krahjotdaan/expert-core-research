import os
from tqdm import tqdm
from langdetect import detect, DetectorFactory, LangDetectException


DetectorFactory.seed = 42
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")

def is_russian(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return False
    try:
        return detect(text) == 'ru'
    except LangDetectException:
        return False


def apply_cleaning(splits):
    cleaned_data = {}

    print("Запуск очистки данных от иностранных отзывов...\n")

    for name, df in splits.items():
        print(f"Обработка сплита: {name}")
        original_count = len(df)

        texts = df['text'].tolist()
        mask = []

        for text in tqdm(texts, desc=f"   Сканирование {name}", unit="rows"):
            mask.append(is_russian(text))

        df_clean = df[mask].reset_index(drop=True)
        filtered_count = len(df_clean)
        removed_count = original_count - filtered_count

        cleaned_data[name] = df_clean

        if not os.path.exists(CLEANED_DIR):
            os.makedirs(CLEANED_DIR)

        print(f"    Удалено: {removed_count} ({(removed_count / original_count) * 100:.2f}%)")
        print(f"    Сохранение в cleaned/{name}_cleaned.csv...")
        df_clean.to_csv(os.path.join(CLEANED_DIR, f"{name}_cleaned.csv"), index=False)
        print("-" * 50)

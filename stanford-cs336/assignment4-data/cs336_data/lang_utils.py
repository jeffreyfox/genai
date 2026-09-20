import fasttext

MODEL_PATH = "local-shared-data/classifiers/lid.176.bin"

language_model = fasttext.load_model(MODEL_PATH)

def identify_language(text: str) -> tuple[str, float]:
    # fastText expects each prediction input to be a single line.
    text = " ".join(text.split())

    if not text:
        return "unknown", 0.0

    labels, proba = language_model.predict(text, k=1)
    language = labels[0].removeprefix("__label__")
    score = float(proba[0])
    print(f"{labels=}, {proba=}, {language=}, {score=}")

    return language, score


def main() -> None:
    text = "Hola Como estas? Bien."

    language, score = identify_language(text)

    print(f"Language: {language}")
    print(f"Confidence: {score:.3f}")


if __name__ == "__main__":
    main()
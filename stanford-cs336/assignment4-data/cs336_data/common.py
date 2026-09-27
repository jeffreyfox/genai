from pathlib import Path
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import re

import modal

MODEL_PATH = "local-shared-data/classifiers/lid.176.bin"

language_model = fasttext.load_model(MODEL_PATH)

MODAL_SHARED_PATH = Path("/shared-data")


def get_shared_assets_path() -> Path:
    if modal.is_local():
        (local_path := Path("local-shared-data").resolve()).mkdir(exist_ok=True, parents=True)
        return local_path
    else:
        return MODAL_SHARED_PATH


def extract_text_from_html_bytes(html_bytes: bytes):
    enc = detect_encoding(html_bytes)
    decoded = html_bytes.decode(enc, errors="replace")
    text = extract_plain_text(decoded)
    return text


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

EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+1[\s.-]?)?"
    r"(?:\(\d{3}\)|\d{3})[\s.-]?"
    r"\d{3}[\s.-]?\d{4}(?!\w)"
)
OCTET = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
IPV4_PATTERN = re.compile(
    rf"(?<![\w.]){OCTET}(?:\.{OCTET}){{3}}(?!\w)"
)


def mask_emails(text: str) -> tuple[str, int]:
    return EMAIL_PATTERN.subn("|||EMAIL_ADDRESS|||", text)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    return PHONE_PATTERN.subn("|||PHONE_NUMBER|||", text)

def mask_ips(text: str) -> tuple[str, int]:
    return IPV4_PATTERN.subn("|||IP_ADDRESS|||", text)
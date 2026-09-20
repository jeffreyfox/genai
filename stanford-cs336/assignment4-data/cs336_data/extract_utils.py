from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes):
    enc = detect_encoding(html_bytes)
    decoded = html_bytes.decode(enc, errors="replace")
    text = extract_plain_text(decoded)
    return text

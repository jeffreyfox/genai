import gzip
from cs336_data.extract_utils import extract_text_from_html_bytes
from warcio.archiveiterator import ArchiveIterator


def extract():
    with gzip.open("local-shared-data/CC/example.warc.gz", "rb") as file:
        html_bytes = file.read()
        text = extract_text_from_html_bytes(html_bytes)
        print(text)

def extract2():
    with open("local-shared-data/CC/example.warc.gz", "rb") as stream:
        i = 0
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue
            i += 1

            html_bytes = record.content_stream().read()
            text = extract_text_from_html_bytes(html_bytes)

            url = record.rec_headers.get_header("WARC-Target-URI")
            print("=" * 80)
            print(f"URL #{i}: {url}")
            print(f"Content:\n{text[:500]}")
            print("=" * 80)
            print()
            if i == 10:
                break


if __name__ == "__main__":
    # extract()
    extract2()
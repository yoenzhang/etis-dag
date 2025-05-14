import re
import pandas as pd

def parse_additional_info(text):
    chunks = text.strip().split('\r\n\r\n')
    results = []
    for chunk in chunks:
        lines = chunk.strip().split('\r\n')
        if not lines:
            continue

        # first line: e.g. "Hindustan Times; 2023-08-22"
        first_line = lines[0]
        url_pattern = r'https?://[^\s]+'
        maybe_link = lines[-1]
        link_match = re.search(url_pattern, maybe_link)

        # parse first line -> title & published
        parts = first_line.split(';')
        if len(parts) == 2:
            title = parts[0].strip()
            published = parts[1].strip()
        else:
            title = first_line.strip()
            published = None

        # summary = lines except first & last (if last is link)
        if link_match:
            link = link_match.group()
            summary_text = " ".join(lines[1:-1]).strip()
        else:
            link = ""
            summary_text = " ".join(lines[1:]).strip()

        results.append({
            "title": title,
            "link": link,
            "published": published,
            "summary": summary_text
        })
    return results

df = pd.read_csv("./data/open_source_records.csv", skiprows=1, encoding="utf-8")

rows = []
for _, row in df.iterrows():
    info = row.get("additional_information", "")
    if isinstance(info, str) and info.strip():
        for entry in parse_additional_info(info):
            rows.append(entry)

extracted_df = pd.DataFrame(rows, columns=["title","link","published","summary"])



# If you want to print them in CSV format:
extracted_df.to_csv("./data/extracted_open_source_records.csv", index=False)

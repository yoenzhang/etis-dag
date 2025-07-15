import csv
import re

input_file = 'training/data/extracted_negative_examples.csv'
output_file = 'training/data/extracted_negative_examples.csv'

cleaned_rows = []
header = ['title', 'link', 'published', 'summary', 'label']

# Helper to clean encoding artifacts
def clean_field(val):
    if not isinstance(val, str):
        val = str(val) if val is not None else ''
    val = val.replace('=3D', '=')
    val = re.sub(r'\s+', ' ', val)  # Collapse whitespace
    val = val.replace('"""', '"')
    return val.strip()

with open(input_file, 'r', encoding='utf-8', errors='replace', newline='') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        # Clean and fill fields
        title = clean_field(row.get('title', 'N/A')) or 'N/A'
        link = clean_field(row.get('link', 'N/A')) or 'N/A'
        published = clean_field(row.get('published', 'N/A')) or 'N/A'
        summary = clean_field(row.get('summary', ''))
        label = clean_field(row.get('label', '0'))
        # Skip if summary is missing or empty
        if not summary:
            continue
        cleaned_rows.append({
            'title': title,
            'link': link,
            'published': published,
            'summary': summary,
            'label': label
        })

with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=header, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for row in cleaned_rows:
        writer.writerow(row)

print(f"Cleaned file saved to {output_file}. Rows: {len(cleaned_rows)}") 
import re
import pandas as pd
from pathlib import Path

URL_RE = re.compile(r'https?://[^\s"]+')

def parse_additional_info(text: str):
    """
    Given one cell's worth of multi-line additional_information,
    splits on *any* blank line, then for each chunk:
      1. Tries to parse "Title; YYYY-MM-DD" from the first line (optional)
      2. Joins all middle lines into one summary string
      3. Finds ALL URLs in the chunk
      4. Creates one record per URL found
    Yields tuples: (title, link, published, summary, label)
    """
    # split on any blank line (one or more newlines with optional spaces)
    raw_chunks = re.split(r'\n\s*\n', text.strip())
    for chunk in raw_chunks:
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if len(lines) < 1:
            continue

        # Try to parse title and date from first line (optional)
        title = None
        published = None
        if len(lines) > 0:
            try:
                if ';' in lines[0]:
                    title_part, pub_part = lines[0].split(';', 1)
                    title = title_part.strip()
                    published = pub_part.strip()
                else:
                    # If no semicolon, treat first line as title
                    title = lines[0].strip()
            except ValueError:
                # If parsing fails, treat first line as title
                title = lines[0].strip()

        # Find ALL URLs in the chunk
        urls = []
        for line in lines:
            for match in URL_RE.finditer(line):
                urls.append(match.group(0))

        # If no URLs found, skip this chunk
        if not urls:
            continue

        # Create summary from all lines except the first (if it was a header)
        summary_lines = lines[1:] if title and published and ';' in lines[0] else lines
        summary = " ".join(summary_lines).strip()

        # Create one record per URL
        for url in urls:
            yield {
                "title": title,
                "link": url,
                "published": published,
                "summary": summary,
                "label": 1  # All extracted articles are positive examples
            }

def extract_to_df(csv_path: str):
    """
    Extract structured data from the additional_information column of a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing additional_information column
        
    Returns:
        DataFrame with columns: title, link, published, summary, label
    """
    # For v2 format, don't skip rows as it has proper headers
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
    
    if "additional_information" not in df.columns:
        raise ValueError(f"Column 'additional_information' not found in {csv_path}. Available columns: {list(df.columns)}")
    
    rows = []
    for info in df["additional_information"].dropna():
        for rec in parse_additional_info(info):
            rows.append(rec)
    
    return pd.DataFrame(rows, columns=["title", "link", "published", "summary", "label"])

if __name__ == "__main__":
    # Define paths relative to the script location
    script_dir = Path(__file__).parent
    input_file = script_dir / "data" / "open_source_records_v2.csv"
    output_file = script_dir / "data" / "extracted_open_source_records_v2.csv"
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        exit(1)
    
    # Extract data
    out_df = extract_to_df(str(input_file))
    
    # Print number of links before deduplication
    print(f"Total links extracted (including duplicates): {len(out_df)}")
    
    # Deduplicate by 'link' column
    dedup_df = out_df.drop_duplicates(subset=["link"])
    print(f"Unique links after deduplication: {len(dedup_df)}")
    
    # Save deduplicated results
    dedup_df.to_csv(output_file, index=False)
    print(f"Successfully extracted {len(dedup_df)} unique records to {output_file}") 
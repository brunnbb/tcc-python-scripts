import re

import pandas as pd


def sanitize_key(key):
    """Use the Zotero key as BibTeX cite key."""
    return re.sub(r"[^a-zA-Z0-9_]", "", key) if pd.notna(key) else "unknown"


def clean_field(value):
    """Clean a field value for BibTeX."""
    if pd.isna(value) or str(value).strip() == "":
        return None
    value = str(value).strip()
    # Escape special BibTeX characters (except braces already there)
    value = value.replace("\\", "\\\\")
    value = value.replace("%", "\\%")
    value = value.replace("&", "\\&")
    value = value.replace("#", "\\#")
    value = value.replace("_", "\\_")
    value = value.replace("^", "\\^{}")
    value = value.replace("~", "\\~{}")
    return value


def format_authors(author_str):
    """Format author field for BibTeX (already in Last, First format usually)."""
    if pd.isna(author_str) or str(author_str).strip() == "":
        return None
    # Zotero exports authors separated by "; "
    authors = str(author_str).split(";")
    authors = [a.strip() for a in authors if a.strip()]
    return " and ".join(authors)


def row_to_bibtex(row):
    """Convert a CSV row to a BibTeX entry."""
    item_type = str(row.get("Item Type", "")).strip().lower()

    # Map Zotero item types to BibTeX entry types
    type_map = {
        "journalarticle": "article",
        "book": "book",
        "booksection": "incollection",
        "conferencepaper": "inproceedings",
        "thesis": "phdthesis",
        "report": "techreport",
        "webpage": "misc",
        "magazinearticle": "article",
        "newspaperarticle": "article",
        "preprint": "misc",
    }
    bib_type = type_map.get(item_type, "misc")

    cite_key = sanitize_key(row.get("Key", "unknown"))

    fields = {}

    # Title
    title = clean_field(row.get("Title"))
    if title:
        fields["title"] = "{" + title + "}"

    # Authors
    authors = format_authors(row.get("Author"))
    if authors:
        fields["author"] = "{" + authors + "}"

    # Year
    year = clean_field(row.get("Publication Year"))
    if not year:
        date = clean_field(row.get("Date"))
        if date:
            match = re.search(r"\b(19|20)\d{2}\b", str(date))
            if match:
                year = match.group(0)
    if year:
        fields["year"] = "{" + year + "}"

    # Journal / book title
    pub_title = clean_field(row.get("Publication Title"))
    if pub_title:
        if bib_type == "article":
            fields["journal"] = "{" + pub_title + "}"
        elif bib_type in ("incollection", "inproceedings"):
            fields["booktitle"] = "{" + pub_title + "}"
        else:
            fields["journal"] = "{" + pub_title + "}"

    # Volume, issue, pages
    volume = clean_field(row.get("Volume"))
    if volume:
        fields["volume"] = "{" + volume + "}"

    issue = clean_field(row.get("Issue"))
    if issue:
        fields["number"] = "{" + issue + "}"

    pages = clean_field(row.get("Pages"))
    if pages:
        # Normalize page range separator
        pages = re.sub(r"\s*[-–—]+\s*", "--", str(pages))
        fields["pages"] = pages

    # DOI
    doi = clean_field(row.get("DOI"))
    if doi:
        fields["doi"] = "{" + doi + "}"

    # URL
    url = clean_field(row.get("Url"))
    if url:
        fields["url"] = "{" + url + "}"

    # Abstract
    abstract = clean_field(row.get("Abstract Note"))
    if abstract:
        fields["abstract"] = "{" + abstract + "}"

    # Publisher / place
    publisher = clean_field(row.get("Publisher"))
    if publisher:
        fields["publisher"] = "{" + publisher + "}"

    place = clean_field(row.get("Place"))
    if place:
        fields["address"] = "{" + place + "}"

    # ISSN / ISBN
    issn = clean_field(row.get("ISSN"))
    if issn:
        fields["issn"] = "{" + issn + "}"

    isbn = clean_field(row.get("ISBN"))
    if isbn:
        fields["isbn"] = "{" + isbn + "}"

    # Language
    language = clean_field(row.get("Language"))
    if language:
        fields["language"] = "{" + language + "}"

    # Keywords (Manual Tags)
    tags = clean_field(row.get("Manual Tags"))
    if tags:
        fields["keywords"] = "{" + tags + "}"

    # Build the BibTeX entry
    lines = [f"@{bib_type}{{{cite_key},"]
    for field, value in fields.items():
        lines.append(f"  {field} = {value},")
    lines.append("}")
    return "\n".join(lines)


def main():
    input_file = "in_data/dados-fmt-bib.csv"
    output_file = "out_data/artigos_selecionados.bib"

    print(f"Lendo arquivo: {input_file}")
    df = pd.read_csv(input_file)

    # Filter only rows where Maioria == YES
    df_yes = df[df["Maioria"].str.strip().str.upper() == "YES"].copy()
    print(f"Total de artigos: {len(df)}")
    print(f"Artigos com Maioria=YES: {len(df_yes)}")

    entries = []
    for _, row in df_yes.iterrows():
        entry = row_to_bibtex(row)
        entries.append(entry)

    bib_content = "\n\n".join(entries)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(bib_content)

    print(f"\nArquivo .bib gerado com sucesso: {output_file}")
    print(f"Total de entradas exportadas: {len(entries)}")


if __name__ == "__main__":
    main()

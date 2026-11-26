import io
import re
from pypdf import PdfReader

def clean_text(text, fix_hyphens=True, remove_page_nums=True, fix_lines=True, clean_spaces=True):
    """Cleans messy text from PDFs or text files."""
    if not text:
        return ""

    text = text.replace("\x00", " ")

    if fix_hyphens:
        text = text.replace("-\n", "")

    if remove_page_nums:
        kept_lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.isdigit() or stripped.lower().startswith("page "):
                continue
            kept_lines.append(line)
        text = "\n".join(kept_lines)

    if fix_lines:
        text = text.replace("\n\n", "PARA_BREAK")
        text = text.replace("\n", " ")
        text = text.replace("PARA_BREAK", "\n\n")

    if clean_spaces:
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def clean_extract_uploaded_file(uploaded_file, cleaning_options):
    """Extracts and cleans text from PDF, TXT, MD files."""
    filename = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower()
    file_bytes = uploaded_file.getvalue()
    records = []

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_num in range(len(reader.pages)):
            raw_text = (reader.pages[page_num].extract_text() or "").strip()
            cleaned = clean_text(
                raw_text,
                cleaning_options["fix_hyphens"],
                cleaning_options["remove_page_nums"],
                cleaning_options["fix_lines"],
                cleaning_options["clean_spaces"],
            )
            if cleaned != "":
                records.append({"source": filename, "page": page_num + 1, "text": cleaned})

    elif ext in ["txt", "md"]:
        raw_text = file_bytes.decode("utf-8", errors="ignore")
        cleaned = clean_text(
            raw_text,
            cleaning_options["fix_hyphens"],
            cleaning_options["remove_page_nums"],
            cleaning_options["fix_lines"],
            cleaning_options["clean_spaces"],
        )
        if cleaned != "":
            records.append({"source": filename, "page": None, "text": cleaned})

    return records

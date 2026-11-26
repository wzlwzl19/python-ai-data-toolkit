"""Python & AI Data Toolkit: five small Streamlit portfolio demos."""

import base64
import hashlib
import json
import re
from collections import Counter
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).parent
ANNOTATION_COMPONENT = components.declare_component("annotation_canvas", path=str(ROOT / "annotator"))

SAMPLE_CSV = b"order_id,customer,amount,order_date\n1001, Nora ,29.90,01/10/2026\n1001,Nora,29.90,01/10/2026\n1002,Sam,,2026-01-11\n1003,Lee,42.00,2026-01-15\n,,,\n1004,Avery,unknown,2026-01-17\n"
SAMPLE_HTML = """<!doctype html>
<html><head><title>Field Notes Coffee</title></head><body><main>
<h1>Field Notes Coffee</h1>
<p>Independent roasters with a rotating selection of single-origin beans.</p>
<h2>Current lineup</h2>
<table><thead><tr><th>Coffee</th><th>Origin</th><th>Price (USD)</th></tr></thead>
<tbody><tr><td>Cedar Trail</td><td>Guatemala</td><td>18</td></tr>
<tr><td>Morning Light</td><td>Kenya</td><td>21</td></tr></tbody></table>
</main></body></html>"""
SAMPLE_DOCUMENT = b"""Field Notes Coffee | Store Handbook

Returns
The return window is 30 days after delivery for unopened merchandise.
To start a return, contact support@fieldnotes.example with your order number.

Shipping
Free shipping applies to domestic orders over $50.
Orders ship within two business days.

Support
Support hours are Monday through Friday, 9 a.m. to 5 p.m. Eastern time.
"""
SAMPLE_IMAGE_URL = "https://raw.githubusercontent.com/matplotlib/matplotlib/main/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg"


def page_header(title, description):
    st.title(title)
    st.caption(description)


def csv_bytes(frame, neutralize_formulas=False):
    """Return a UTF-8 CSV, optionally protecting spreadsheet formula cells."""
    output = frame.copy()
    if neutralize_formulas:
        def safe(value):
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                return "'" + value
            return value

        output = output.map(safe)
        output.columns = [safe(str(column)) for column in output.columns]
    return output.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def read_csv(uploaded_file, encoding, delimiter):
    raw = SAMPLE_CSV if not uploaded_file else uploaded_file.getvalue()
    if not raw.strip():
        raise ValueError("The CSV file is empty.")
    return pd.read_csv(BytesIO(raw), encoding=encoding, sep=delimiter, dtype=str,
                       keep_default_na=False, engine="python")


def apply_cleaning(frame, remove_missing, remove_duplicates, required_columns,
                   conversion_column, conversion, filter_column, filter_operator, filter_value):
    result = frame.apply(lambda series: series.str.strip()).replace("", pd.NA)
    conversion_failures = 0
    if remove_missing:
        result = result.dropna(how="all")
    if required_columns:
        result = result.dropna(subset=required_columns)
    if conversion_column and conversion != "Keep text":
        original = result[conversion_column]
        if conversion == "Number":
            converted = pd.to_numeric(original, errors="coerce")
        elif conversion == "Date (YYYY-MM-DD)":
            converted = pd.to_datetime(original, errors="coerce", format="mixed", utc=True).dt.strftime("%Y-%m-%d")
        else:
            converted = original.str.lower()
        conversion_failures = int((original.notna() & converted.isna()).sum())
        result[conversion_column] = converted
    if remove_duplicates:
        result = result.drop_duplicates()
    if filter_column and filter_value.strip():
        value = filter_value.strip()
        series = result[filter_column].astype("string")
        if filter_operator == "Contains":
            mask = series.str.contains(value, case=False, regex=False, na=False)
        elif filter_operator == "Equals":
            mask = series.str.casefold().eq(value.casefold())
        else:
            numbers = pd.to_numeric(series, errors="coerce")
            target = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(target):
                mask = pd.Series(False, index=result.index)
            elif filter_operator == "Greater than":
                mask = numbers > target
            else:
                mask = numbers < target
        result = result[mask]
    return result.reset_index(drop=True), conversion_failures


def data_cleaner():
    page_header("Data Cleaner", "Clean, transform, filter, preview, and export a CSV file.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is None:
        st.info("No file uploaded. The built-in sample CSV is ready to test.")
    encoding, delimiter = st.columns(2)
    selected_encoding = encoding.selectbox("File encoding", ["utf-8-sig", "cp1252", "gb18030"])
    selected_delimiter = delimiter.selectbox(
        "Delimiter", [",", ";", "\t", "|"], format_func=lambda item: "Tab" if item == "\t" else item
    )
    try:
        source = read_csv(uploaded, selected_encoding, selected_delimiter)
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        st.error(f"Could not read this CSV file. Check its format and encoding. Details: {error}")
        return
    if source.empty or len(source.columns) == 0:
        st.error("The CSV file has no usable columns or rows.")
        return
    with st.expander("Original data preview", expanded=True):
        st.dataframe(source.head(200), width="stretch")
    remove_missing, remove_duplicates = st.columns(2)
    drop_blank_rows = remove_missing.checkbox("Remove rows that are completely blank", value=True)
    drop_duplicate_rows = remove_duplicates.checkbox("Remove duplicate rows", value=True)
    required = st.multiselect("Remove rows missing values in selected columns", source.columns.tolist())
    conversion_column, conversion_type = st.columns(2)
    convert_column = conversion_column.selectbox(
        "Column format conversion", source.columns.tolist(), index=None, placeholder="Choose a column"
    )
    convert_to = conversion_type.selectbox(
        "Convert selected column to", ["Keep text", "Number", "Date (YYYY-MM-DD)", "Lowercase"]
    )
    filter_column, filter_operator, filter_value = st.columns(3)
    filter_by = filter_column.selectbox(
        "Filter column", source.columns.tolist(), index=None, placeholder="Choose a column"
    )
    filter_mode = filter_operator.selectbox("Filter condition", ["Contains", "Equals", "Greater than", "Less than"])
    filter_text = filter_value.text_input("Filter value", disabled=filter_by is None)
    cleaned, failures = apply_cleaning(
        source, drop_blank_rows, drop_duplicate_rows, required, convert_column, convert_to,
        filter_by, filter_mode, filter_text
    )
    if failures:
        st.warning(f"{failures} value(s) could not be converted and became blank.")
    metrics = st.columns(3)
    metrics[0].metric("Original rows", len(source))
    metrics[1].metric("Clean rows", len(cleaned))
    metrics[2].metric("Rows removed", len(source) - len(cleaned))
    st.subheader("Cleaned data preview")
    st.dataframe(cleaned.head(500), width="stretch")
    st.download_button("Download cleaned CSV", csv_bytes(cleaned), "cleaned_data.csv", "text/csv")


def parse_static_html(html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup.select("script, style, template, noscript, header, footer, nav, [hidden]"):
        tag.decompose()
    content_root = soup.select_one("main, article") or soup.body or soup
    headings = [node.get_text(" ", strip=True) for node in content_root.select("h1,h2,h3,h4,h5,h6")]
    tables = []
    for table in content_root.select("table")[:10]:
        try:
            parsed = pd.read_html(StringIO(str(table)), flavor="bs4")
        except (ValueError, TypeError, IndexError):
            continue
        if parsed:
            tables.append(parsed[0].iloc[:500, :50])
    text = content_root.get_text(" ", strip=True)
    return title, headings, text[:30000], tables


def web_scraper():
    page_header("Web Scraper", "Extract a title, text, and tables from a small static HTML sample.")
    st.warning("This is demo only, works for sample static HTML, not for arbitrary websites.")
    st.text_area("Static HTML input", SAMPLE_HTML, height=250, key="html_input")
    if st.button("Parse static HTML", type="primary"):
        html = st.session_state.html_input
        if not html.strip():
            st.warning("Paste or load some HTML before parsing.")
        elif len(html.encode("utf-8")) > 1_000_000:
            st.error("The HTML demo input is limited to 1 MB.")
        else:
            try:
                st.session_state.html_result = parse_static_html(html)
            except Exception as error:
                st.error(f"Could not parse the HTML sample. Details: {error}")
    if "html_result" not in st.session_state:
        st.info("The built-in static HTML sample is shown above. Click Parse static HTML to begin.")
        return
    title, headings, text, tables = st.session_state.html_result
    st.subheader(title or "Untitled HTML document")
    text_tab, tables_tab = st.tabs(["Extracted text", f"Extracted tables ({len(tables)})"])
    with text_tab:
        st.text_area("Extracted body text", text, height=180, disabled=True)
        if headings:
            st.write("Headings")
            st.write("\n".join(f"- {heading}" for heading in headings))
        rows = [{"type": "Title", "value": title}]
        rows.extend({"type": "Heading", "value": heading} for heading in headings)
        rows.append({"type": "Text", "value": text})
        st.download_button("Download extracted content CSV", csv_bytes(pd.DataFrame(rows), True),
                           "extracted_content.csv", "text/csv")
    with tables_tab:
        if not tables:
            st.info("No HTML tables were found in this sample.")
        else:
            table_index = st.selectbox("Select table", range(len(tables)), format_func=lambda item: f"Table {item + 1}")
            st.dataframe(tables[table_index], width="stretch")
            st.download_button("Download extracted table CSV", csv_bytes(tables[table_index], True),
                               f"extracted_table_{table_index + 1}.csv", "text/csv")


def load_sample_image():
    request = Request(SAMPLE_IMAGE_URL, headers={"User-Agent": "PythonAIDataToolkit/1.0"})
    with urlopen(request, timeout=10) as response:
        raw = response.read(8_000_000)
    with Image.open(BytesIO(raw)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def fallback_image():
    image = Image.new("RGB", (720, 360), "#e6ece7")
    draw = ImageDraw.Draw(image)
    draw.ellipse((85, 71, 311, 302), fill="#cb652e")
    draw.rounded_rectangle((380, 87, 623, 289), radius=35, fill="#265a73")
    return image


def image_annotation():
    page_header("Image Annotation Tool", "Draw rectangles, enter labels, undo annotations, and export pixel coordinates.")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
    try:
        if uploaded:
            with Image.open(BytesIO(uploaded.getvalue())) as candidate:
                if candidate.width * candidate.height > 20_000_000:
                    raise ValueError("Images must contain at most 20 million pixels.")
                image = ImageOps.exif_transpose(candidate).convert("RGB")
        else:
            image = load_sample_image()
            st.caption("Built-in sample image loaded from a public URL.")
    except Exception as error:
        if uploaded:
            st.error(f"Could not open this image. Details: {error}")
            return
        st.warning(f"The online sample image could not be loaded. Using a generated fallback image. Details: {error}")
        image = fallback_image()
    original_width, original_height = image.size
    display_image = image.copy()
    display_image.thumbnail((950, 580))
    buffer = BytesIO()
    display_image.save(buffer, format="JPEG", quality=90)
    filename = uploaded.name if uploaded else "sample_image.jpg"
    state_key = "annotation_" + hashlib.sha256(filename.encode() + image.tobytes()).hexdigest()
    boxes = ANNOTATION_COMPONENT(
        source="data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        width=original_width, height=original_height,
        boxes=st.session_state.get(state_key, []), key="canvas_" + state_key, default=None,
    )
    if boxes is not None:
        st.session_state[state_key] = boxes
    boxes = st.session_state.get(state_key, [])
    if boxes:
        st.dataframe(pd.DataFrame(boxes)[["label", "x", "y", "width", "height"]], width="stretch")
    annotation_data = {
        "image": {"filename": filename, "width": original_width, "height": original_height},
        "annotations": boxes,
    }
    st.download_button("Download annotation JSON", json.dumps(annotation_data, indent=2),
                       "image_annotations.json", "application/json", disabled=not boxes)


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    from transformers import pipeline

    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)


def sentiment_scores(text):
    predictions = load_sentiment_pipeline()(text, truncation=True, max_length=512)
    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    return sorted([(label_map.get(item["label"], item["label"].title()), float(item["score"]))
                   for item in predictions], key=lambda item: item[1], reverse=True)


def sentiment_analysis():
    page_header("Sentiment Analysis", "Classify English text as Positive, Neutral, or Negative with a Hugging Face model.")
    st.info("English text only. The first analysis downloads the public model and may take a little longer.")
    uploaded = st.file_uploader("Upload UTF-8 text file", type=["txt"])
    default_text = "I loved the product. The setup was easy and the support team was excellent."
    if uploaded:
        try:
            default_text = uploaded.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("This text file is not valid UTF-8. Save it as UTF-8 and try again.")
            return
    text = st.text_area("English text", default_text, height=180, max_chars=6000)
    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Enter English text before running the analysis.")
        else:
            try:
                with st.spinner("Loading the Hugging Face model and analyzing the text..."):
                    st.session_state.sentiment_result = (text, sentiment_scores(text))
            except Exception as error:
                st.error(f"The sentiment model could not run. Check the deployment dependencies and try again. Details: {error}")
    stored_result = st.session_state.get("sentiment_result")
    scores = stored_result[1] if stored_result and stored_result[0] == text else None
    if scores:
        st.metric("Predicted label", scores[0][0])
        st.bar_chart(pd.DataFrame({"Confidence": [score for _, score in scores]}, index=[label for label, _ in scores]))
        for label, score in scores:
            st.progress(score, text=f"{label}: {score:.1%}")


STOP_WORDS = set("a an the to and or in on is are was be for of what which when where how who why can i do does it with this that from my please tell me about you we our".split())


def document_chunks(raw, filename):
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(raw))
        if reader.is_encrypted or len(reader.pages) > 100:
            raise ValueError("Upload an unlocked PDF with no more than 100 pages.")
        pages = [(f"{filename} - page {index}", page.extract_text() or "")
                 for index, page in enumerate(reader.pages, start=1)]
    else:
        pages = [(filename, raw.decode("utf-8-sig"))]
    if sum(len(text) for _, text in pages) > 250_000:
        raise ValueError("The extracted document text is limited to 250,000 characters.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=100)
    return [(source, chunk) for source, text in pages for chunk in splitter.split_text(text)]


def terms(text):
    return set(re.findall(r"[^\W_]+", text.lower())) - STOP_WORDS


def answer_question(question, chunks):
    query_terms = terms(question)
    if not query_terms:
        return []
    token_sets = [terms(chunk) for _, chunk in chunks]
    counts = Counter(term for token_set in token_sets for term in token_set & query_terms)
    weights = {term: 1 + (len(chunks) / max(1, count)) for term, count in counts.items()}
    ranking = sorted(range(len(chunks)), key=lambda index: sum(
        weights.get(term, 1) for term in token_sets[index] & query_terms
    ), reverse=True)
    hits = []
    seen = set()
    minimum_overlap = 1 if len(query_terms) <= 2 else 2
    for index in ranking[:4]:
        source, chunk = chunks[index]
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk):
            overlap = terms(sentence) & query_terms
            if len(overlap) >= minimum_overlap and sentence not in seen:
                hits.append((sum(weights.get(term, 1) for term in overlap), sentence, source))
                seen.add(sentence)
    hits.sort(reverse=True)
    return [(sentence, source) for _, sentence, source in hits[:3]]


def show_answer(question, hits):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        if not hits:
            st.write("Cannot find relevant information from uploaded documents.")
            return
        for sentence, source in hits:
            st.write(sentence)
            st.caption(f"Source: {source}")


def rag_chatbot():
    page_header("Lightweight RAG Chatbot", "Ask questions about PDF or TXT documents with source and page citations.")
    uploaded_files = st.file_uploader("Upload PDF or TXT documents", type=["pdf", "txt"], accept_multiple_files=True)
    if len(uploaded_files) > 5:
        st.error("Upload no more than five documents at a time.")
        return
    documents = [(file.name, file.getvalue()) for file in uploaded_files]
    if not documents:
        st.info("No documents uploaded. The built-in handbook is ready to test.")
        documents = [("Store Handbook.txt", SAMPLE_DOCUMENT)]
    signature = [(name, hashlib.sha256(raw).hexdigest()) for name, raw in documents]
    if st.session_state.get("document_signature") != signature:
        chunks = []
        for name, raw in documents:
            try:
                chunks.extend(document_chunks(raw, name))
            except Exception as error:
                st.error(f"Could not read {name}. Details: {error}")
        st.session_state.document_signature = signature
        st.session_state.document_chunks = chunks
        st.session_state.document_messages = []
    chunks = st.session_state.get("document_chunks", [])
    if not chunks:
        st.warning("No readable text was found. Scanned PDFs need OCR before upload.")
        return
    st.caption(f"{len(chunks)} searchable passages")
    for question, hits in st.session_state.get("document_messages", []):
        show_answer(question, hits)
    question = st.chat_input("Ask a question about the uploaded documents", max_chars=500)
    if question:
        hits = answer_question(question, chunks)
        st.session_state.document_messages = (st.session_state.document_messages + [(question, hits)])[-20:]
        show_answer(question, hits)


DEMO_FUNCTIONS = {
    "Data Cleaner": data_cleaner,
    "Web Scraper": web_scraper,
    "Image Annotation Tool": image_annotation,
    "Sentiment Analysis": sentiment_analysis,
    "Lightweight RAG Chatbot": rag_chatbot,
}


st.set_page_config(page_title="Python & AI Data Toolkit", page_icon="🧰", layout="wide")
st.sidebar.title("Build five Streamlit demos")
selected_demo = st.sidebar.selectbox("Choose a demo", list(DEMO_FUNCTIONS))
st.sidebar.caption("Python & AI Data Toolkit")
DEMO_FUNCTIONS[selected_demo]()


# Python & AI Data Toolkit

This project contains five English-language Streamlit demos behind one `app.py` entry point. The sidebar title is `Build five Streamlit demos`, and the dropdown switches between Data Cleaner, Web Scraper, Image Annotation Tool, Sentiment Analysis, and Lightweight RAG Chatbot.

## Run locally

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first Sentiment Analysis request downloads the public Hugging Face model. The built-in samples allow each page to be tried without preparing input files.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, the `annotator/` folder, and this README to the repository root.
2. Open `https://share.streamlit.io/`, sign in, and choose **Create app**.
3. Select the repository, branch, and `app.py` as the main file.
4. Use a supported Python version in **Advanced settings**, then choose **Deploy**.
5. Open the generated `streamlit.app` URL and test every dropdown page with its built-in sample before sharing the link.

## Web Scraper network limitation

The Web Scraper page intentionally parses only the built-in or user-pasted static HTML string. It does not request arbitrary URLs, crawl websites, bypass robots rules, access login pages, or access local and private network addresses. The page shows this warning to users: `This is demo only, works for sample static HTML, not for arbitrary websites.`

## Notes

- Sentiment Analysis uses the public three-class `cardiffnlp/twitter-roberta-base-sentiment-latest` model through `transformers` and reports Positive, Neutral, and Negative confidence scores.
- Lightweight RAG Chatbot uses `pypdf` for PDF text extraction and LangChain text splitters for chunking. It cites the filename and PDF page when it finds relevant passages.
- Scanned PDFs require OCR before upload. If no relevant passage is found, the chatbot displays `Cannot find relevant information from uploaded documents.`


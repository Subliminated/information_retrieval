# Information Retrieval Indexer and Search

This repository contains a simple Python-based information retrieval system that builds an inverted index from a collection of text documents and provides a search interface for ranked and proximity-based retrieval.

## Overview

- **index.py**: Indexes all text files in a specified `./data` directory, creating an inverted index that maps terms to their occurrences (with position and line information) in each document. The index is stored in a specified output directory (e.g., `./doc_index`).
- **search.py**: Loads the generated inverted index and allows users to search for terms, returning ranked results and supporting advanced features like proximity and coverage scoring.

## Folder Structure

```
Project/
├── data/                # Place your plain text files here (one file per document)
│   ├── 1
│   ├── 2
│   └── ...
├── doc_index/           # Output folder for the generated inverted index and per-document line mappings
│   ├── inverted_index.json
│   ├── 1.json
│   ├── 2.json
│   └── ...
├── index.py             # Indexer script
├── search.py            # Search and ranking script
├── README.md            # This documentation
└── ...
```

## Data Format
- All documents should be placed in the `./data` folder.
- Each file should be a plain text file, readable with Python's `open(filename, 'r', encoding='utf-8')`.
- The filename (e.g., `1`, `2`, etc.) is used as the document ID.

## How to Run

### Indexing Documents
Run the following command to build the index:

```sh
python index.py ./data ./doc_index
```

- This will process all files in `./data` and create the inverted index and per-document line mappings in `./doc_index`.

### Searching the Index
Run the following command to start the search interface:

```sh
python search.py ./doc_index
```

- Enter your search queries at the prompt. Use `> ` before your query to display matching lines from the documents.

## Dependencies
- Python 3.x (standard library modules: `os`, `re`, `sys`, `shutil`, `json`, `bisect`, `collections`)
- [NLTK](https://www.nltk.org/) (Natural Language Toolkit) for tokenization, lemmatization, stemming, and POS tagging

Install NLTK with:
```sh
pip install nltk
```

The first run will automatically download required NLTK data packages.

## Notes
- The indexer uses lemmatization and stemming to normalize terms, so different forms of a word (e.g., "distribution", "distribute", "distributing") are indexed as the same token.
- The search program supports ranked retrieval, coverage, and proximity scoring.
- For more details, see the comments and instructions at the bottom of `index.py` and `search.py`.

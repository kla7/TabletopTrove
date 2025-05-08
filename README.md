# TabletopTrove

**Author:** Kasey La (kaseyla@brandeis.edu)

This project is a RAG System for board game recommendation (Advanced ML Methods for NLP 2 Semester Project).

## Instructions

The instructions for running the app are as follows:

### Requirements

* Python 3.10 or higher
* Install dependencies from [requirements.txt](https://github.com/kla7/TabletopTrove/blob/main/requirements.txt)
* Get a Llama 3.2 API key, setting the environment variable name to `LLAMA-3.2`

### Prepare files

* Download the dataset from [Kaggle](https://www.kaggle.com/datasets/gabrio/board-games-dataset)
* Run `python load_data.py` to retrieve `metadata.json`
* Run `python prepare_faiss.py` to retrieve `embeddings.npy` and `faiss.index`
    * Adding the flag `-s` will optionally sample 1000 entries from the dataset if a smaller subset is preferred.

### Streamlit

To run the app:
* In your terminal, run
```
streamlit run app.py
```

* The default server is http://localhost:8501/

## App Features

* A textbox for the search query.
* Filter options:
    * Only show base games (no expansions)
    * Include games for 12+ players
    * Include games with playtime over 360 minutes (6 hours)
    * Slider to select desired min/max number of players that result should support
    * Slider to select desired min/max amount of playtime (in minutes) that the result should have
    * Slider to select desired min/max average rating that the result should have
* An info message that shows how many results were found via FAISS.
    * The maximum number of results is 20. Generally, more results lead to a longer loading time for answer generation.

## Contents of this repository

This folder contains 9 files and 1 directory:

1. This **README** file.
2. **rag_output**, a directory containing evaluation efforts along with results.
3. **app.py**, a script containing the Streamlit app.
   * The code can be adjusted to utilize sampled files. See lines 21 and 27.
4. **evaluate.py**, a script that performs ROUGE evaluation for the LLM outputs.
    * Can be run from command line. Run `python evaluate.py -h` for the help menu.
5. **load_data.py**, a script that converts and cleans the raw dataset.
6. **prepare_faiss.py**, a script that retrieves the embeddings and index files for FAISS.
7. **query_faiss.py**, a script that handles querying with FAISS.
8. **presentation.pdf**, presentation slides that contain a brief summary of the project.
9. **report.pdf**, a report documenting the details of the project.
10. **requirements.txt**, the dependencies for running the project.
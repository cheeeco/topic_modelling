import re
from pathlib import Path

import click
import nltk
import pandas as pd
import PyPDF2
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download necessary resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("omw-1.4")


def pdf_to_text(pdf_path: str) -> str:
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text


def preprocess_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    processed_text = " ".join(tokens)
    return processed_text


class BagOfWordsModel(object):
    def __init__(self, id_document_dict, max_features=None, max_df=1.0, model=CountVectorizer):
        """Builds bow model.
        Args:
            id_document_dict: ids of documents and theirs contents in format
                "{id: 'text', ...}"
            max_features: If not None, build a vocabulary that only consider the top
                max_features ordered by term frequency across the corpus.
                This parameter is ignored if vocabulary is not None.
            max_df: When building the vocabulary ignore terms that have a
                document frequency strictly higher than the given threshold
                (corpus-specific stop words). If float, the parameter
                represents a proportion of documents, integer absolute counts.
                This parameter is ignored if vocabulary is not None.
        """
        logger.info(
            "Building bag-of-words model with max_features={0}, max_df={1}".format(
                max_features, max_df
            )
        )
        logger.info("Size of data set: " + str(len(id_document_dict)))

        if len(id_document_dict) != 0:
            logger.info("Building pandas dataframe")
            df = pd.DataFrame.from_dict(data=id_document_dict, orient="index")
            logger.info("Built pandas dataframe")
            ids = df.index
            self.index2id = dict(enumerate(ids))
            self.id2index = {v: k for k, v in self.index2id.items()}
            documents_corpus = df[0].values
            del df
            if max_features is None:
                logger.info(
                    "Training CountVectorizer with all {0} features".format(len(ids))
                )
            else:
                logger.info(
                    "Training CountVectorizer with max {0} features".format(
                        max_features
                    )
                )
            vectorizer = model(
                max_features=max_features, max_df=max_df, stop_words="english"
            ).fit(documents_corpus)
            logger.info(
                "Trained vectorizer with {0} features".format(
                    len(vectorizer.get_feature_names_out())
                )
            )
            logger.info("Building bag-of-words model")
            bow = vectorizer.transform(documents_corpus)
            logger.info("Done")

            self.url_ids = ids
            self.bow_sparse_matrix = bow
            self.feature_names = (
                vectorizer.get_feature_names_out()
            )  # mapping from url_id to url
            self.vocabulary = vectorizer.vocabulary_  # mapping from url to url_id
            self.shape = self.bow_sparse_matrix.shape

    def get_index(self, doc_id):
        return self.id2index[doc_id]

    def get_doc_id(self, index):
        return self.index2id[index]

    def get_feature_id(self, feature_name):
        return self.vocabulary.get(feature_name)

    def get_feature_name(self, feature_id):
        return self.feature_names[feature_id]

    def toarray(self):
        return self.bow_sparse_matrix.toarray()

    def to_uci(self, model_name="bow", save_folder=""):
        import codecs
        import os.path

        if self.bow_sparse_matrix is None:
            logger.error("Model is None.")
            return
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        filenames = model_name
        docword_name = os.path.join(save_folder, "docword." + filenames + ".txt")
        vocab_name = os.path.join(save_folder, "vocab." + filenames + ".txt")
        with (
            codecs.open(docword_name, "w", encoding="utf-8") as docword_f,
            codecs.open(vocab_name, "w", encoding="utf-8") as vocab_f,
        ):
            urls_count = self.shape[0]
            words_count = self.shape[1]
            # Fill vocab_f file
            logger.info("Start filling {0}".format(vocab_name))
            for i in range(words_count):
                vocab_f.write(self.get_feature_name(i) + "\n")
            logger.info("Done.")
            # Fill docword_f file
            logger.info("Start filling {0}".format(docword_name))
            docword_f.write(str(urls_count) + "\n")
            docword_f.write(str(words_count) + "\n")
            docword_f.write(str(self.bow_sparse_matrix.nnz) + "\n")
            # nnz_position = docword_f.tell() # We fill this line later with nnz_counter.
            # nnz_counter = 0 # The number of nonzero counts in the bag-of-words.
            nnz_x, nnz_y = self.bow_sparse_matrix.nonzero()
            for x, y in zip(nnz_x, nnz_y):
                # nnz_counter += len(url_sparse_vector)
                docword_f.write(
                    str(x + 1)
                    + " "
                    + str(y + 1)
                    + " "
                    + str(self.bow_sparse_matrix[x, y])
                    + "\n"
                )
            logger.info("Done.")


@click.command()
@click.option(
    "--collection-path",
    required=True,
    type=str,
    help="Path to folder with papers in PDF format",
)
def run(collection_path: str):
    collection_path = Path(collection_path)
    if not collection_path.exists():
        raise ValueError("Error! Collection doesn't exist!")
    logger.info(f"Found a collection at {collection_path}")
    documents = {}

    for i, file in enumerate(collection_path.rglob("*.pdf")):
        logger.info(f"Reading a document at {file}")
        file_text = pdf_to_text(file)
        file_text = preprocess_text(file_text)
        documents[i] = file_text

    bow_model = BagOfWordsModel(id_document_dict=documents, model=TfidfVectorizer)
    bow_model.to_uci(model_name="tfidf", save_folder=collection_path)


if __name__ == "__main__":
    run()

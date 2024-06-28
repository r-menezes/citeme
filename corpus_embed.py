import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Main Class
class CorpusEmbed(object):
    def __init__(self, model='allenai-specter', load_file=None, sentence_df=None, sentence_column='sentence', id_column='id'):

        assert (load_file is not None) != (sentence_df is not None), "Either a file or a DataFrame must be provided"

        self.id_column = id_column
        self.sentence_column = sentence_column

        # load the model
        print("Loading model...")
        if type(model) == str:
            self.model = SentenceTransformer(model)
        elif type(model) == SentenceTransformer:
            self.model = model
        else:
            raise ValueError("model must be convertable to a SentenceTransformer object")

        if load_file is not None:
            self.load_embeddings(load_file)
        else:
            self.make_embeddings(sentence_df, sentence_column, id_column)


    def make_embeddings(self, sentence_df, sentence_column='sentence', id_column='id'):

        assert type(sentence_df) == pd.DataFrame, "sentence_df must be a pandas DataFrame"
        assert sentence_column in sentence_df.columns, "The sentence_column must be present in the DataFrame"
        assert id_column in sentence_df.columns, "The id_column must be present in the DataFrame"

        # store metadata dataframe
        self.metadata = sentence_df

        # construct and store embeddings
        print("Computing embeddings...")
        corpus = self.metadata[sentence_column].tolist()
        _embeddings = self.model.encode(corpus, convert_to_tensor=True)

        # structured array with id and embeddings for faster search
        print("Storing embeddings...")

        # TODO: implement a better index than just the idx of the row
        # type_emb = _embeddings[0].dtype
        # type_id = sentence_df[id_column].dtype
        # self.embeddings = np.array(
        #     [(sentence_df[id_column][i], _embeddings[i]) for i in range(len(_embeddings))],
        #     dtype=[(id_column, type_id), ('embed', type_emb)]
        # )

        self.embeddings = _embeddings

        print("Done!")


    def save_embeddings(self, output_folder=".", output_file="embeddings"):
        
        print(f"Saving embeddings to {output_folder}/{output_file}.npy and metadata to {output_folder}/{output_file}_metadata.parquet")
        
        np.save(output_folder + "/" + output_file + ".npy", self.embeddings)
        self.metadata.to_parquet(output_folder + "/" + output_file + "_metadata.parquet")
        
        print("Done!")


    def load_embeddings(self, load_file):
        import os

        assert load_file[-4:] == ".npy", "The file storing the embeddings must have a .npy extension."
        assert os.path.exists(load_file), "The path for the file storing the embeddings does not exist."
        assert os.path.exists(load_file[:-4] + "_metadata.parquet"), "The path for the metadata file does not exist."

        print("Loading embeddings...")

        self.embeddings = np.load(load_file)
        self.metadata = pd.read_parquet(load_file[:-4] + "_metadata.parquet")

        print("Done!")


    def extend_embeddings(self, sentence, sentence_id, metadata):
        
        assert type(sentence) == str, "sentence must be a string"
        
        # update metadata
        # if some metadata is missing, fill it with NAs
        # BUG: unchecked
        metadata[self.id_column] = sentence_id
        metadata[self.sentence_column] = sentence
        self.metadata = self.metadata.append(metadata, ignore_index=True)

        # update embeddings
        sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)
        self.embeddings = np.append(self.embeddings, np.array([(sentence_id, sentence_embedding)], dtype=self.embeddings.dtype))


    def search_reference(self, phrase):
        query_embedding = self.model.encode(phrase, convert_to_tensor=True)
        search_hits = util.semantic_search(query_embedding, self.embeddings)
        search_hits = search_hits[0]  # Get the hits for the first query

        columns = self.metadata.keys()

        # Print the results
        print("\n\nPhrase:", phrase)
        print("Related papers:")
        for hit in search_hits[:7]:
            # hit_id = self.embeddings[hit["corpus_id"]][self.id_column]
            # related_paper = self.metadata[self.metadata[self.id_column] == hit_id]
            # related_paper = self.metadata.iloc[hit["corpus_id"]]

            # print all available metadata in a neat way
            print(f"{'-'*10}")
            # for col in related_paper.columns:
                # print(f"{col}: {related_paper[col].values[0]}")
            for key in columns:
                print(f"{key}: {self.metadata[key].iloc[hit['corpus_id']]}")

            print(f"Similarity: {hit['score']}")
            print(f"{'-'*10}")

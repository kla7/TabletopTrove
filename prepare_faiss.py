import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import json


def prepare_faiss(sample_data=False, sample_base_size=700, sample_expansion_size=300):
    """
    Prepare faiss index
    :param sample_data: Determine whether to use a sample of the dataset or the full dataset
    :param sample_base_size: The number of base board games to include in the sample
    :param sample_expansion_size: The number of board game expansions to include in the sample
    """
    with open('metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    df = pd.DataFrame(metadata)

    if sample_data:
        df_base = df[df['type'] == 'boardgame']
        df_expansion = df[df['type'] == 'boardgameexpansion']

        df_base_sampled = df_base.sample(n=sample_base_size, random_state=1)
        df_expansion_sampled = df_expansion.sample(n=sample_expansion_size, random_state=1)

        df = pd.concat([df_base_sampled, df_expansion_sampled])
        df = df.reset_index(drop=True)
        print(f"Using subset: {df.shape[0]} rows")

        df.to_json('sampled_metadata.json', orient='records', force_ascii=False, indent=2)
    else:
        df = df.reset_index(drop=True)
        print(f"Using full dataset: {df.shape[0]} rows")

    df = df.fillna("").astype(str)

    df['min_players_desc'] = df['min_players'].apply(lambda x: players_desc(x, 'Minimum'))
    df['max_players_desc'] = df['max_players'].apply(lambda x: players_desc(x, 'Maximum'))
    df['min_playtime_desc'] = df['min_playtime'].apply(lambda x: playtime_desc(x, 'Minimum playtime'))
    df['max_playtime_desc'] = df['max_playtime'].apply(lambda x: playtime_desc(x, 'Maximum playtime'))
    df['avg_playtime_desc'] = df['avg_playtime'].apply(lambda x: playtime_desc(x, 'Average playtime'))
    df['min_age_desc'] = df['min_age'].apply(lambda x: f'Minimum age: Unknown' if x == '0' else f'Minimum age: {x}')
    df['type_desc'] = df['type'].apply(lambda x: 'Base game' if x == 'boardgame' else 'Expansion')
    df['expansion_desc'] = df['expansion'].apply(lambda x: f'Expansion of {x}' if x != 'None' else 'Not an expansion')
    df['avg_rating_desc'] = df['avg_rating'].apply(rating_desc)

    primary_cols = ['id', 'name', 'description', 'category', 'mechanic']
    desc_cols = [col for col in df.columns if col.endswith('_desc')]
    other_cols = [col for col in df.columns if (col not in primary_cols and col not in desc_cols)]
    df = df[primary_cols + desc_cols + other_cols]

    if sample_data:
        df.to_json('desc_sampled_metadata.json', orient='records', force_ascii=False, indent=2)
    else:
        df.to_json('desc_metadata.json', orient='records', force_ascii=False, indent=2)

    df["embedding"] = df[df.columns].agg(" | ".join, axis=1)

    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    embeddings = []

    for text in tqdm(df["embedding"].tolist(), desc="Encoding embeddings", unit="row"):
        embedding = embedder.encode([text], convert_to_numpy=True)
        embeddings.append(embedding[0])

    embeddings = np.array(embeddings, dtype='float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    if sample_data:
        np.save("sampled_embeddings.npy", embeddings)
    else:
        np.save("embeddings.npy", embeddings)

    print("Embeddings saved.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    if sample_data:
        faiss.write_index(index, "sampled_faiss.index")
    else:
        faiss.write_index(index, "faiss.index")

    print("FAISS index saved.")


def players_desc(players: str, label: str) -> str:
    """
    Helper function to convert a number of players into a more informative string.
    :param players: A number of players.
    :param label: The particular label of interest, i.e., 'minimum' or 'maximum'.
    :return: A more readable and comprehensible description for the columns denoting a number of players.
    """
    if players == '0':
        return f'{label}: Unknown'
    else:
        return f'{label} {players} players'


def playtime_desc(playtime: str, label: str) -> str:
    """
    Helper function to convert an amount of playtime into a more informative string.
    :param playtime: An amount of playtime.
    :param label: The particular label of interest, i.e., 'minimum', 'maximum', or 'average'.
    :return: A more readable and comprehensible description for the columns denoting an amount of playtime.
    """
    if playtime == '0':
        return f'{label}: Unknown'
    else:
        return f'{label}: {playtime} minutes'


def rating_desc(rating: str) -> str:
    """
    Helper function to convert an average rating to a more informative string via binning, where 0-5 denotes
    a 'low rated game', 5-7 denotes an 'average rated game', and 7-10 denotes a 'highly rated game'.
    :param rating: An average rating.
    :return: A more readable and comprehensible description for the average rating column to support search.
    """
    rating = float(rating)
    if rating >= 7:
        label = 'Highly rated game'
    elif rating >= 5:
        label = 'Average rated game'
    else:
        label = 'Low rated game'
    return f'{label} ({rating:.1f})'


if __name__ == "__main__":
    prepare_faiss(sample_data=True)
    # prepare_faiss(sample_data=False)

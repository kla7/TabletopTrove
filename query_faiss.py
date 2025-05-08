import json
import faiss
import numpy as np


def load_faiss_index(path):
    return faiss.read_index(path)


def load_embeddings(path):
    return np.load(path)


def load_metadata(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def encode_query(query, model):
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)


def search_index(query_embedding, index, k=5):
    distances, indices = index.search(query_embedding, k)
    return distances, indices


def retrieve_results(distances, indices, metadata):
    results = []
    for idx, i in enumerate(indices[0]):
        if i < len(metadata):
            results.append((distances[0][idx], metadata[i]))
        else:
            print(f'Index {i} out of bounds for metadata size {len(metadata)}')
    return results


def search_boardgames(query, model, index, metadata, k=10, base_only=False, min_players=None, max_players=None,
                      min_playtime=None, max_playtime=None, min_rating=None, max_rating=None):
    query_embedding = encode_query(query, model)
    distances, indices = search_index(query_embedding, index, k)
    results = retrieve_results(distances, indices, metadata)

    filtered = []

    for _, item in results:
        if item['min_players'] > item['max_players']:
            item['max_players'] = item['min_players']

        if item['min_playtime'] > item['max_playtime']:
            item['max_playtime'] = item['min_playtime']

        if base_only and item['type'] != 'boardgame':
            continue
        if item['max_players'] < min_players or item['min_players'] > max_players:
            continue
        if item['max_playtime'] < min_playtime or item['min_playtime'] > max_playtime:
            continue
        if item['avg_rating'] < min_rating or item['avg_rating'] > max_rating:
            continue

        filtered.append(item)

    return filtered

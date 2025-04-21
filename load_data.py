import sqlite3
import pandas as pd
import json
import html
import re
from tqdm import tqdm

con = sqlite3.connect('database.sqlite')

selected_columns = [
    'game.id',
    'game.type',  # board game or expansion
    'details.description',
    'details.maxplayers',
    'details.maxplaytime',
    'details.minage',
    'details.minplayers',
    'details.minplaytime',
    'details.name',
    'details.playingtime',  # average playtime
    'attributes.boardgamecategory',
    'attributes.boardgameexpansion',  # base game for expansion
    'attributes.boardgamemechanic',
    'stats.average'  # average rating 1-10 on BGG
]

search_columns = ", ".join([f'"{column}"' for column in selected_columns])

df_filtered = pd.read_sql_query(f"SELECT {search_columns} FROM BoardGames", con)


def clean_text(text: str, is_description: bool = False, fix_commas: bool = False) -> str:
    """
    Replaces escaped characters with their proper unicode characters and removes newline characters.
    If the text is from a description column, removes html comments.
    If the text is from a column containing lists delimited by commas, ensures there is a space after each comma.
    :param text: The text to be cleaned.
    :param is_description: Whether the text is from a description column.
    :param fix_commas: Whether the text is from a column containing lists delimited by commas.
    :return: Cleaned text.
    """
    text = html.unescape(text)

    if is_description:
        text = re.sub(r'<!--.*?--!>', '', text, flags=re.DOTALL)

    if fix_commas:
        text = re.sub(r'\s*,\s*', ', ', text)

    return text.replace('\n', ' ').strip()


for col in df_filtered.columns:
    is_description = col == 'details.description'

    fix_commas = col in [
        'attributes.boardgamecategory',
        'attributes.boardgamemechanic',
        'attributes.boardgameexpansion'
    ]

    df_filtered[col] = df_filtered[col].astype(str).apply(
        lambda x: clean_text(x, is_description=is_description, fix_commas=fix_commas)
    )

int_cols = [
    'details.minplayers',
    'details.maxplayers',
    'details.minplaytime',
    'details.maxplaytime',
    'details.playingtime',
    'details.minage'
]

for col in int_cols:
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce').fillna(0).astype(int)

df_filtered['stats.average'] = pd.to_numeric(df_filtered['stats.average'], errors='coerce').fillna(0).round(2)

metadata = []

for _, row in tqdm(df_filtered.iterrows(), desc='Loading data', total=len(df_filtered), unit='row'):
    metadata.append({
        'id': row['game.id'],
        'name': row['details.name'],
        'description': row['details.description'],
        'category': row['attributes.boardgamecategory'],
        'mechanic': row['attributes.boardgamemechanic'],
        'min_players': row['details.minplayers'],
        'max_players': row['details.maxplayers'],
        'min_playtime': row['details.minplaytime'],
        'max_playtime': row['details.maxplaytime'],
        'avg_playtime': row['details.playingtime'],
        'min_age': row['details.minage'],
        'type': row['game.type'],
        'expansion': row['attributes.boardgameexpansion'],
        'avg_rating': row['stats.average']
    })

with open('metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

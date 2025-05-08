import streamlit as st
import query_faiss as qf
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import login
import os

torch.classes.__path__ = []

login(os.environ.get('LLAMA-3.2'))


@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@st.cache_resource
def get_index():
    # return qf.load_faiss_index('sampled_faiss.index')
    return qf.load_faiss_index('faiss.index')


@st.cache_data
def get_metadata():
    # data = qf.load_metadata('desc_sampled_metadata.json')
    data = qf.load_metadata('desc_metadata.json')
    for game in data:
        game['min_players'] = int(game['min_players']) if game['min_players'].isdigit() else None
        game['max_players'] = int(game['max_players']) if game['max_players'].isdigit() else None
        game['min_playtime'] = int(game['min_playtime']) if game['min_playtime'].isdigit() else None
        game['max_playtime'] = int(game['max_playtime']) if game['max_playtime'].isdigit() else None
        game['min_age'] = int(game['min_age']) if game['min_age'].isdigit() else None
        game['avg_rating'] = float(game['avg_rating'])
    return data


@st.cache_resource
def load_generator():
    return pipeline(
        'text-generation',
        model='meta-llama/Llama-3.2-1B-Instruct',
        device_map='cuda'
    )


def generate_answer(query: str, retrieved_passages: list[str]) -> str:
    combined_prompt = ("**System prompt:** You are a helpful assistant and board game expert. "
                       "Use the relevant passages below to recommend ONE game based on the query. "
                       "Provide a clear response that summarizes the passage and game description "
                       "for the selected game. Make sure to include the number of players "
                       "it supports and how long typical playtime is, if known. "
                       "End your answer with a complete sentence. "
                       "Do not repeat the system prompt. "
                       "Do not use external sources.\n\n")

    combined_prompt += (f'**Query:** {query}\n\n'
                        f'**Relevant passages:**\n\n')

    for idx, passage in enumerate(retrieved_passages):
        combined_prompt += f'[Game #{idx + 1}]\n\n{passage}\n\n'

    combined_prompt += (f'**Based on the relevant passages, select ONE game to recommend '
                        f'and provide a comprehensive description for the selected game. '
                        f'Include information about the number of players and typical playtime if known.**')

    eos_token_id = st.session_state.generator_pipeline.tokenizer.eos_token_id

    output = st.session_state.generator_pipeline(
        combined_prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.3,
        pad_token_id=eos_token_id,
        eos_token_id=eos_token_id
    )[0]['generated_text']

    answer = output.replace(combined_prompt, '').strip()
    return answer


model = load_model()
index = get_index()
metadata = get_metadata()

if 'generator_pipeline' not in st.session_state:
    st.session_state.generator_pipeline = load_generator()

st.title('Tabletop Trove')

all_min_players = [game['min_players'] for game in metadata if game['min_players'] is not None]
all_max_players = [game['max_players'] for game in metadata if game['max_players'] is not None]
all_min_playtime = [game['min_playtime'] for game in metadata if game['min_playtime'] is not None]
all_max_playtime = [game['max_playtime'] for game in metadata if game['max_playtime'] is not None]
all_avg_ratings = [item['avg_rating'] for item in metadata if item['avg_rating'] is not None]

query = st.text_input('What kind of board game are you looking for?', key='query')

# -- SEARCH FILTERS --

default_filters = {
    'players': (min(all_min_players), min(max(all_max_players), 12)),
    'playtime': (min(all_min_playtime), min(max(all_max_playtime), 360)),
    'rating': (0.0, 10.0),
    'base_only': False,
    'include_large_players': False,
    'include_large_playtime': False
}

if 'apply_filters' not in st.session_state:
    st.session_state.apply_filters = default_filters.copy()
    for k, v in default_filters.items():
        st.session_state[k] = v

if st.session_state.get('reset_filters'):
    st.session_state.apply_filters = default_filters.copy()
    for k, v in default_filters.items():
        st.session_state[k] = v
    st.session_state.reset_filters = False
    st.rerun()

filters = st.session_state.apply_filters

with st.expander('Filter options'):
    base_only = st.checkbox(
        'Only show base games (no expansions)',
        key='base_only'
    )
    include_large_players = st.checkbox(
        'Include games for 12+ players',
        key='include_large_players'
    )
    include_large_playtime = st.checkbox(
        'Include games with playtime over 360 minutes (6 hours)',
        key='include_large_playtime'
    )

    if st.session_state.include_large_players:
        max_players_cap = max(all_max_players)
    else:
        max_players_cap = min(max(all_max_players), 12)

    if st.session_state.include_large_playtime:
        max_playtime_cap = max(all_max_playtime)
    else:
        max_playtime_cap = min(max(all_max_playtime), 360)

    st.slider(
        'Number of Players',
        min_value=min(all_min_players),
        max_value=max_players_cap,  # max is 11299
        step=1,
        key='players'
    )

    st.slider(
        'Playtime (minutes)',
        min_value=min(all_min_playtime),
        max_value=max_playtime_cap,  # max is 60120
        step=10,
        key='playtime'
    )

    st.slider(
        'Average rating',
        min_value=0.0,
        max_value=10.0,
        step=1.0,
        key='rating'
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Apply filters'):
            st.session_state.apply_filters = {
                'players': st.session_state.players,
                'playtime': st.session_state.playtime,
                'rating': st.session_state.rating,
                'base_only': st.session_state.base_only,
                'include_large_players': st.session_state.include_large_players,
                'include_large_playtime': st.session_state.include_large_playtime
            }

    with col2:
        if st.button('Reset filters'):
            st.session_state.reset_filters = True
            st.rerun()

filters = st.session_state.apply_filters

min_players, max_players = filters['players']
min_playtime, max_playtime = filters['playtime']
min_rating, max_rating = filters['rating']
base_only = filters['base_only']
include_large_players = filters['include_large_players']
include_large_playtime = filters['include_large_playtime']

# -- QUERY FAISS --

if query:
    filtered_results = qf.search_boardgames(
        query,
        model,
        index,
        metadata,
        k=20,
        base_only=base_only,
        min_players=min_players,
        max_players=max_players,
        min_playtime=min_playtime,
        max_playtime=max_playtime,
        min_rating=min_rating,
        max_rating=max_rating
    )

    passages = []

    st.info(f'**Results found**: {len(filtered_results)}')

    for game in filtered_results:
        passage = f'**{game["name"]}**: '

        if game['category'] != 'None':
            passage += f'It falls under the following categories: {game["category"]}. '
        if game['mechanic'] != 'None':
            passage += f'It has the following mechanics: {game["mechanic"]}. '

        if game['min_players'] == game['max_players'] and game['min_players'] == 0:
            pass
        elif game['min_players'] != 0 and game['max_players'] == 0:
            passage += f'It supports {game["min_players"]} players. '
        elif game['min_players'] == 0 and game['max_players'] != 0:
            passage += f'It supports {game["max_players"]} players. '
        elif game['min_players'] == game['max_players']:
            passage += f'It supports {game["min_players"]} players. '
        else:
            passage += f'It supports {game["min_players"]}-{game["max_players"]} players. '

        if game['min_playtime'] == game['max_playtime'] and game['min_playtime'] == 0:
            pass
        elif game['min_playtime'] != 0 and game['max_playtime'] == 0:
            passage += f'Typical playtime is {game["min_playtime"]} minutes. '
        elif game['min_playtime'] == 0 and game['max_playtime'] != 0:
            passage += f'Typical playtime is {game["max_playtime"]} minutes. '
        elif game['min_playtime'] == game['max_playtime']:
            passage += f'Typical playtime is {game["min_playtime"]} minutes. '
        else:
            passage += f'Typical playtime is {game["min_playtime"]}-{game["max_playtime"]} minutes. '

        if game['min_age'] > 0:
            passage += f'Recommended for ages {game["min_age"]} and up. '

        if game['type'] == 'boardgameexpansion':
            passage += f'This is an expansion: {game["expansion_desc"]}. '

        if game['avg_rating'] > 0.0:
            passage += f'Average rating: {game["avg_rating_desc"]}. '

        passage += f'{game["description"]}'

        passages.append(passage)

# -- LLM --

    if st.button('Generate answer', key='generate'):
        with st.spinner('Generating answer...'):
            answer = generate_answer(query, passages)
            st.markdown(f'### Answer:\n\n{answer}')

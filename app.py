import streamlit as st
import query_faiss as qf
import torch
from sentence_transformers import SentenceTransformer

torch.classes.__path__ = []


@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@st.cache_resource
def get_index():
    return qf.load_faiss_index('sampled_faiss.index')


@st.cache_data
def get_metadata():
    data = qf.load_metadata('desc_sampled_metadata.json')
    for game in data:
        game['min_players'] = int(game['min_players']) if game['min_players'].isdigit() else None
        game['max_players'] = int(game['max_players']) if game['max_players'].isdigit() else None
        game['min_playtime'] = int(game['min_playtime']) if game['min_playtime'].isdigit() else None
        game['max_playtime'] = int(game['max_playtime']) if game['max_playtime'].isdigit() else None
        game['min_age'] = int(game['min_age']) if game['min_age'].isdigit() else None
        game['avg_rating'] = float(game['avg_rating'])
    return data


model = load_model()
index = get_index()
metadata = get_metadata()

st.title('Tabletop Trove')
query = st.text_input('Enter your search query')

# SEARCH FILTERS

all_min_players = [game['min_players'] for game in metadata if game['min_players'] is not None]
all_max_players = [game['max_players'] for game in metadata if game['max_players'] is not None]
all_min_playtime = [game['min_playtime'] for game in metadata if game['min_playtime'] is not None]
all_max_playtime = [game['max_playtime'] for game in metadata if game['max_playtime'] is not None]
all_avg_ratings = [item['avg_rating'] for item in metadata if item['avg_rating'] is not None]

default_filters = {
    'players': (min(all_min_players), min(max(all_min_players), 12)),
    'playtime': (min(all_min_playtime), min(max(all_min_playtime), 360)),
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
        # value=filters['rating'],
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

if query:
    query_embedding = qf.encode_query(query, model)
    distances, indices = qf.search_index(query_embedding, index, k=10)
    results = qf.retrieve_results(distances, indices, metadata)

    filtered_results = qf.search_boardgames(
        query,
        model,
        index,
        metadata,
        k=10,
        base_only=base_only,
        min_players=min_players,
        max_players=max_players,
        min_playtime=min_playtime,
        max_playtime=max_playtime,
        min_rating=min_rating,
        max_rating=max_rating
    )

    for i, game in enumerate(filtered_results, 1):
        st.markdown(f'### {i}. {game["name"]}')
        st.markdown(f'**Description:** {game["description"]}')
        st.markdown(f'**Category:** {game["category"]}')
        st.markdown(f'**Mechanic:** {game["mechanic"]}')

        if game['min_players'] == game['max_players'] and game['min_players'] == 0:
            st.markdown(f'**Players:** Unknown')
        elif game['min_players'] != 0 and game['max_players'] == 0:
            st.markdown(f'**Players:** {game["min_players"]}')
        elif game['min_players'] == 0 and game['max_players'] != 0:
            st.markdown(f'**Players:** {game["max_players"]}')
        elif game['min_players'] == game['max_players']:
            st.markdown(f'**Players:** {game["min_players"]}')
        else:
            st.markdown(f'**Players:** {game["min_players"]}-{game["max_players"]}')

        if game['min_playtime'] == game['max_playtime'] and game['min_playtime'] == 0:
            st.markdown(f'**Playtime:** Unknown')
        elif game['min_playtime'] != 0 and game['max_players'] == 0:
            st.markdown(f'**Playtime:** {game["min_playtime"]} minutes')
        elif game['min_playtime'] == 0 and game['max_playtime'] != 0:
            st.markdown(f'**Playtime:** {game["max_playtime"]} minutes')
        elif game['min_playtime'] == game['max_playtime']:
            st.markdown(f'**Playtime:** {game["min_playtime"]} minutes')
        else:
            st.markdown(f'**Playtime:** {game["min_playtime"]}-{game["max_playtime"]} minutes')

        if game['min_age'] == 0:
            st.markdown(f'**Age:** Unknown')
        else:
            st.markdown(f'**Age:** {game["min_age"]}+')

        st.markdown(f'**Type:** {game["type_desc"]}')

        if game['type'] == 'boardgameexpansion':
            st.markdown(f'**Expansion:** {game["expansion_desc"]}')

        if game['avg_rating'] == 0.0:
            st.markdown(f'**Average rating:** None')
        else:
            st.markdown(f'**Average rating:** {game["avg_rating_desc"]}')

        st.markdown('---')

import pandas as pd
import polars as pl
import numpy as np
import plotly.express as px
import os
import streamlit as st

os.chdir(os.path.dirname(os.path.abspath(__file__)))



# Define the data
@st.cache_data
def load_in_data():
    people = pl.read_parquet("closest_people_df.parquet")
    places = pl.read_parquet("closest_places_df.parquet")
    topics = pl.read_parquet("closest_topics_df.parquet")


    return people, places, topics

@st.cache_data
def make_df_display():
    df_display = pl\
        .read_csv('derived_data.csv')\
        .rename({
            'Internal ID': 'internal_id',
            'Document Type': 'document_type',
            'Parent ID': 'parent_id',
            'Order': 'order',
            'Parent Name': 'parent_name',
            'UUID': 'uuid',
            'Name': 'name',
            'Website URL': 'website_url',
            'Short URL': 'short_url',
            'Image URL': 'image_url',
            'Original Transcript': 'original_transcript',
            'Text Only Transcript': 'text_only_transcript',
            'People': 'people',
            'Places': 'places',
            'First Date': 'first_date',
            'Dates': 'dates',
            'Topics': 'topics'})\
        .select(['internal_id', 'text_only_transcript', 'people', 'places', 'topics'])\
        .with_columns([
            pl.col("people").fill_null(""),
            pl.col("places").fill_null(""),
            pl.col("topics").fill_null("")])\
        .with_columns([
            pl\
                .col("text_only_transcript")\
                .str\
                .replace_all(r"\||\[\[|\]\]|\\r|\\t|\\n", " "),
            pl\
                .col("people")\
                .str\
                .replace_all(r"\||\[\[|\]\]|\\r|\\t|\\n", ", "),
            pl\
                .col("places")\
                .str\
                .replace_all(r"\||\[\[|\]\]|\\r|\\t|\\n", ", "),
            pl\
                .col("topics")\
                .str\
                .replace_all(r"\||\[\[|\]\]|\\r|\\t|\\n", ", ")])
    
    return df_display

@st.cache_data
def get_emotions():
    return pd.read_csv("emotions.csv")\
        .rename(columns={"Unnamed: 0": "emotions"})\
        .set_index("emotions")

def grab_from_internal_id(internal_id, column_name):
    return df_display\
        .filter(pl.col("internal_id") == internal_id)\
        [column_name]\
        .to_list()[0]

with st.status("Loading and calculating..."):
    st.write("Defining data...")
    df_display = make_df_display()
    people, places, topics = load_in_data()
    emotions = get_emotions()
    st.write("Finished!")

def return_text(df, match_number, column_name):
    return df_display\
        .filter(pl.col("internal_id") == match_number)\
        .select([column_name])\
        .item()

def make_radarchart(id):

    user_data = emotions.filter([str(id)])

    return px\
        .line_polar(user_data, 
                    r=user_data.values.flatten(), 
                    theta=user_data.index, 
                    line_close=True, 
                    line_shape='linear')\
        .update_traces(fill='toself', 
                    marker=dict(size=10, color='LightSkyBlue'),
                    line=dict(color='RoyalBlue', width=2))\
        .update_layout(title=f'Emotional Radar Chart for<br>Journal Entry {id}<br><br>',
                    width=450, 
                    height=450,
                    polar=dict(
                            radialaxis=dict(visible=True,
                                            showticklabels = False),
                            angularaxis=dict(showline=False, showticklabels=True)),
                    font=dict(size=12, color='RebeccaPurple'))

st.sidebar.title("Choose a Journal Entry")

# Get the user's choice
input_number = st.sidebar.selectbox(
    "Which journal entry would you like to view?",
    df_display["internal_id"].to_list())

emotion_graph = make_radarchart(input_number)
st.sidebar.plotly_chart(emotion_graph)

# Display the journal entry
side_col1, side_col2, side_col3 = st.sidebar.columns(3)

with side_col1:
    st.sidebar.write("Topics:")
    st.sidebar.write(f":blue[{grab_from_internal_id(input_number, 'topics')}]")

with side_col2:
    st.sidebar.write("People:")
    st.sidebar.write(f":red[{grab_from_internal_id(input_number, 'people')}]")

with side_col3:
    st.sidebar.write("Places:")
    st.sidebar.write(f":green[{grab_from_internal_id(input_number, 'places')}]")

st.sidebar.write("Transcript:")
st.sidebar.write(grab_from_internal_id(input_number, "text_only_transcript"))


# ######################
# # Main Page
# ######################

st.title("Woodruff Similarity Algorithm")

tab1, tab2, tab3 = st.tabs(["People", "Topics", "Places"])

with tab1:

    ###
    st.header("People Match 1")
    ###
    
    match1_people = people\
        .filter(pl.col("internal_id") == input_number)\
        .select(["closest_1"])[0]\
        .item()
    
    st.write(f"Internal ID: {match1_people}")

    col1_tab1, col2_tab1, col3_tab1 = st.columns(3)

    with col1_tab1:
        st.write("Topics:")
        st.write(f":blue[{return_text(people, match1_people, 'topics')}]")

    with col2_tab1:
        st.write("People:")
        st.write(f":red[{return_text(people, match1_people, 'people')}]")

    with col3_tab1:
        st.write("Places:")
        st.write(f":green[{return_text(people, match1_people, 'places')}]")

    st.write(df_display\
        .filter(pl.col("internal_id") == match1_people)\
        .select(["text_only_transcript"])[0]\
        .item())

    ###

    st.header("People Match 2")

    match2_people = people\
        .filter(pl.col("internal_id") == input_number)\
        .select(["closest_2"])[0]\
        .item()
    
    st.write(f"Internal ID: {match2_people}")

    col1_tab1, col2_tab1, col3_tab1 = st.columns(3)

    with col1_tab1:
        st.write("Topics:")
        st.write(f":blue[{return_text(people, match2_people, 'topics')}]")

    with col2_tab1:
        st.write("People:")
        st.write(f":red[{return_text(people, match2_people, 'people')}]")

    with col3_tab1:
        st.write("Places:")
        st.write(f":green[{return_text(people, match2_people, 'places')}]")

    st.write(df_display\
        .filter(pl.col("internal_id") == match2_people)\
        .select(["text_only_transcript"])[0]\
        .item())
    
    ###

    st.header("People Match 3")

    match3_people = people\
        .filter(pl.col("internal_id") == input_number)\
        .select(["closest_3"])[0]\
        .item()
    
    st.write(f"Internal ID: {match3_people}")

    col1_tab1, col2_tab1, col3_tab1 = st.columns(3)

    with col1_tab1:
        st.write("Topics:")
        st.write(f":blue[{return_text(people, match3_people, 'topics')}]")

    with col2_tab1:
        st.write("People:")
        st.write(f":red[{return_text(people, match3_people, 'people')}]")

    with col3_tab1:
        st.write("Places:")
        st.write(f":green[{return_text(people, match3_people, 'places')}]")

    st.write(df_display\
        .filter(pl.col("internal_id") == match3_people)\
        .select(["text_only_transcript"])[0]\
        .item())

with tab2:

    ###
    st.header("Topics Match 1")
    ###
    
    match1_topics = topics\
        .filter(pl.col("internal_id") == input_number)\
        .select(["closest_1"])[0]\
        .item()
    
    st.write(f"Internal ID: {match1_topics}")

    col1_tab1, col2_tab1, col3_tab1 = st.columns(3)

    with col1_tab1:
        st.write("Topics:")
        st.write(f":blue[{return_text(topics, match1_topics, 'topics')}]")

    with col2_tab1:
        st.write("People:")
        st.write(f":red[{return_text(topics, match1_topics, 'people')}]")

    with col3_tab1:
        st.write("Places:")
        st.write(f":green[{return_text(topics, match1_topics, 'places')}]")

    st.write(df_display\
        .filter(pl.col("internal_id") == match1_topics)\
        .select(["text_only_transcript"])[0]\
        .item())

    ###

    st.header("Topics Match 2")

    match2_topics = topics\
        .filter(pl.col("internal_id") == input_number)\
        .select(["closest_2"])[0]\
        .item()
    
    st.write(f"Internal ID: {match2_topics}")

    col1_tab1, col2_tab1, col3_tab1 = st.columns(3)

    with col1_tab1:
        st.write("Topics:")
        st.write(f":blue[{return_text(topics, match2_topics, 'topics')}]")

    with col2_tab1:
        st.write("People:")
        st.write(f":red[{return_text(topics, match2_topics, 'people')}]")

    with col3_tab1:
        st.write("Places:")
        st.write(f":green[{return_text(topics, match2_topics, 'places')}]")

    st.write(df_display\
        .filter(pl.col("internal_id") == match2_topics)\
        .select(["text_only_transcript"])[0]\
        .item())
    
    ###

    st.header("Topics Match 3")

    match3_topics = topics\
        .filter(pl.col("internal_id") == input_number)\
        .select(["closest_3"])[0]\
        .item()
    
    st.write(f"Internal ID: {match3_topics}")

    col1_tab1, col2_tab1, col3_tab1 = st.columns(3)

    with col1_tab1:
        st.write("Topics:")
        st.write(f":blue[{return_text(topics, match3_topics, 'topics')}]")

    with col2_tab1:
        st.write("People:")
        st.write(f":red[{return_text(topics, match3_topics, 'people')}]")

    with col3_tab1:
        st.write("Places:")
        st.write(f":green[{return_text(topics, match3_topics, 'places')}]")

    st.write(df_display\
        .filter(pl.col("internal_id") == match3_topics)\
        .select(["text_only_transcript"])[0]\
        .item())

with tab3:

    ###
    st.header("Places Match 1")
    ###
    
    match1_places = places\
        .filter(pl.col("internal_id") == input_number)\
        .select(["closest_1"])[0]\
        .item()
    
    st.write(f"Internal ID: {match1_places}")

    col1_tab1, col2_tab1, col3_tab1 = st.columns(3)

    with col1_tab1:
        st.write("Topics:")
        st.write(f":blue[{return_text(places, match1_places, 'topics')}]")

    with col2_tab1:
        st.write("People:")
        st.write(f":red[{return_text(places, match1_places, 'people')}]")

    with col3_tab1:
        st.write("Places:")
        st.write(f":green[{return_text(places, match1_places, 'places')}]")

    st.write(df_display\
        .filter(pl.col("internal_id") == match1_places)\
        .select(["text_only_transcript"])[0]\
        .item())

    ###

    st.header("Places Match 2")

    match2_places = places\
        .filter(pl.col("internal_id") == input_number)\
        .select(["closest_2"])[0]\
        .item()
    
    st.write(f"Internal ID: {match2_places}")

    col1_tab1, col2_tab1, col3_tab1 = st.columns(3)

    with col1_tab1:
        st.write("Topics:")
        st.write(f":blue[{return_text(places, match2_places, 'topics')}]")

    with col2_tab1:
        st.write("People:")
        st.write(f":red[{return_text([places], match2_places, 'people')}]")

    with col3_tab1:
        st.write("Places:")
        st.write(f":green[{return_text(places, match2_places, 'places')}]")

    st.write(df_display\
        .filter(pl.col("internal_id") == match2_places)\
        .select(["text_only_transcript"])[0]\
        .item())
    
    ###

    st.header("Places Match 3")

    match3_places = places\
        .filter(pl.col("internal_id") == input_number)\
        .select(["closest_3"])[0]\
        .item()
    
    st.write(f"Internal ID: {match3_places}")

    col1_tab1, col2_tab1, col3_tab1 = st.columns(3)

    with col1_tab1:
        st.write("Topics:")
        st.write(f":blue[{return_text(places, match3_places, 'topics')}]")

    with col2_tab1:
        st.write("People:")
        st.write(f":red[{return_text(places, match3_places, 'people')}]")

    with col3_tab1:
        st.write("Places:")
        st.write(f":green[{return_text(places, match3_places, 'places')}]")

    st.write(df_display\
        .filter(pl.col("internal_id") == match3_places)\
        .select(["text_only_transcript"])[0]\
        .item())

import pandas as pd
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

closest_people = pl.read_parquet("closest_people.parquet")
closest_places = pl.read_parquet("closest_places.parquet")
closest_topics = pl.read_parquet("closest_topics.parquet")

unwanted_match_words = [ 
    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
    "\\|", "\\[\\[", "\\]\\]", "\\[", "\\]", "\\\\r", "\\\\t", "\\\\n"
    ]
           
nogo = "|".join(unwanted_match_words)
           
df = pl\
    .read_csv('../../derived_data.csv')\
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
        pl.col("text_only_transcript").fill_null(""),
        pl.col("people").fill_null(""),
        pl.col("places").fill_null(""),
        pl.col("topics").fill_null("")])\
    .with_columns([
        pl\
            .col("text_only_transcript")\
            .str\
            .replace_all(rf"{nogo}", " "),

        pl\
            .col("people")\
            .str\
            .replace_all(r"\||\[\[|\]\]|\\r|\\t|\\n", " ")\
            .alias("text_people"),
        pl\
            .col("places")\
            .str\
            .replace_all(r"\||\[\[|\]\]|\\r|\\t|\\n", " ")\
            .alias("text_places"),
        pl\
            .col("topics")\
            .str\
            .replace_all(r"\||\[\[|\]\]|\\r|\\t|\\n", " ")\
            .alias("text_topics")])

def sim_maker(df, var):

    """
    This makes a similarity matrix for a given col (var) in the dataframe.
    """

    def create_tfidf_matrix(data, stop_words='english'):
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = tfidf_vectorizer.fit_transform(data)
        return tfidf_matrix

    tfidf_matrix_var = create_tfidf_matrix(df[var])
    var_similarity = cosine_similarity(tfidf_matrix_var)

    return var_similarity

ndarray_wwtext = sim_maker(df, 'text_only_transcript')

def closest_indices_df_duds(my_df):

    return my_df\
        .melt(id_vars=["internal_id"])\
        .sort("value", descending=True)\
        .group_by("internal_id")\
        .head(4)\
        .group_by("internal_id")\
        .apply(lambda group: group\
            .with_columns(pl.col("value")\
            .rank(method='ordinal', descending=True)\
            .alias("rank")))\
        .pivot(index='internal_id',columns="rank", values="variable")\
        .rename({'1': 'closest_0', '2': 'closest_1', '3': 'closest_2', '4': 'closest_3'})

def fix_duds(ranks):

    duds = ranks.filter(
            (pl.col("closest_0") == "1") & 
            (pl.col("closest_1") == "2") & 
            (pl.col("closest_2") == "3") & 
            (pl.col("closest_3") == "4"))["internal_id"]\
            .cast(str)\
            .to_list()
    
    backup = pl.DataFrame(ndarray_wwtext)
    backup.columns = df["internal_id"].cast(str)

    reduced_raw = backup\
        .select(duds)\
        .with_columns([df["internal_id"].alias("internal_id")])\
        .filter(pl.col("internal_id").is_in(list(pd.Series(duds).astype('int'))))\
    
    duds_rank = closest_indices_df_duds(reduced_raw)

    ranks = ranks.filter(~pl.col("internal_id").is_in(list(pd.Series(duds).astype('int'))))
    
    return pl.concat([ranks,duds_rank], how='vertical').sort("internal_id")

fix_duds(closest_people).write_parquet("closest_people_df.parquet")
fix_duds(closest_places).write_parquet("closest_places_df.parquet")
fix_duds(closest_topics).write_parquet("closest_topics_df.parquet")

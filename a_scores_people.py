import pandas as pd
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_path = 'derived_data.csv'

print("1: Loading data")
unwanted_match_words = [
    
    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
    "\\|", "\\[\\[", "\\]\\]", "\\[", "\\]", "\\\\r", "\\\\t", "\\\\n"
   
    ]
           
nogo = "|".join(unwanted_match_words)
           
df = pl\
    .read_csv(data_path)\
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
    Creates a similarity matrix for a given column (var) in the dataframe, where each row in the column represents
    a journal entry. The function uses Term Frequency-Inverse Document Frequency (TF-IDF) to transform the textual data
    into a numerical format that captures the importance of words in the context of the whole dataset. This transformed
    data is then used to calculate the cosine similarity between each pair of entries, providing a measure of similarity
    based on the text content.

    Args:
        df (DataFrame): The dataframe containing the journal entries.
        var (str): The name of the column for which the similarity matrix is to be created.

    Returns:
        ndarray: A square matrix where each element [i, j] represents the cosine similarity between the text content
                 of the i-th and j-th journal entries.
    """



    def create_tfidf_matrix(data, stop_words='english'):
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = tfidf_vectorizer.fit_transform(data)
        return tfidf_matrix

    tfidf_matrix_var = create_tfidf_matrix(df[var])
    var_similarity = cosine_similarity(tfidf_matrix_var)

    return var_similarity

print("2: Calculating Similarity Matrices")
ndarray_wwtext = sim_maker(df, 'text_only_transcript')

print("2a: People")
pl_people = pl.DataFrame(sim_maker(df, 'text_people'))

print("3: Calculating Percentiles")
def get_percentiles_fromdf_tolist_byrow(my_df, quantile=.75):

    """
    Calculates the specified percentile for each column in the given DataFrame, with each column representing a different 
    journal entry. The function reshapes the DataFrame and computes percentiles for each column (journal entry) to
    identify a threshold value. This is useful for filtering or thresholding similarity scores.

    Args:
        my_df (DataFrame): The DataFrame containing similarity scores or other numerical values for each journal entry.
        quantile (float, optional): The percentile to calculate for each column. Defaults to 0.75.

    Returns:
        list: A list of percentile values for each column in the DataFrame.

    Explanation:
        - The columns of the DataFrame are set to the 'internal_id' of each journal entry for identification.
        - The DataFrame is reshaped using the 'melt' function, which transforms it into a long format, making 
          each row represent a single score/value along with its corresponding journal entry ID.
        - The function then groups the reshaped data by the journal entry IDs and calculates the specified 
          percentile for the scores/values in each group.
        - These percentile values are then sorted by the journal entry IDs and returned as a list.
    """

    my_df.columns = df["internal_id"].cast(str) 

    percentiles = my_df\
            .with_row_count(name="temp_index", offset=1)\
            .melt(id_vars=["temp_index"])\
            .with_columns(pl.col("variable").cast(int))\
            .group_by("variable")\
            .agg(pl.col("value").quantile(quantile).alias("percentile_n"))\
            .sort("variable")["percentile_n"]\
            .to_list()

    return percentiles

print("3a: People")
percentiles_people = get_percentiles_fromdf_tolist_byrow(pl_people)

print("4a: Thresholding People")
pl_people_thresh = pl_people * (pl_people > pl.Series(percentiles_people))

print("5: Calculating Closest Indices")
def closest_indices_df(my_df, ids = df["internal_id"]):
    """
    Identifies the top matches (closest indices) for each journal entry based on their similarity scores.

    Args:
        my_df (DataFrame): The DataFrame containing similarity scores, where each column corresponds to a journal entry.
        ids (Series): A series of internal IDs corresponding to each journal entry, used for labeling.

    Returns:
        DataFrame: A DataFrame where each row corresponds to a journal entry, and columns indicate the top matches.

    Explanation:
        - First, the columns of the DataFrame are set to the 'internal_id' of each journal entry.
        - The DataFrame is reshaped (melted) into a long format where each row represents a similarity score between two entries.
        - The data is sorted by the similarity score in descending order.
        - The data is then grouped by each journal entry's internal ID.
        - For each group, the top 4 highest similarity scores are kept (top matches).
        - Within each group, the similarity scores are ranked.
        - The DataFrame is then pivoted to make each row represent a journal entry, and columns represent its top matches.
        - The columns are renamed to indicate the rank of each match (closest_0, closest_1, etc.).
    """

    my_df.columns = ids.cast(str) 

    return my_df\
        .with_columns([df["internal_id"].alias("internal_id")])\
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
    
    # return ranks.concat(duds_rank, how='vertical').sort("internal_id")
    return pl.concat([ranks,duds_rank], how='vertical').sort("internal_id")

print("5a: People")
closest_people = closest_indices_df(pl_people_thresh)

closest_people.write_parquet("closest_people.parquet")

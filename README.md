

# Purpose of this Repository

The purpose of this repository is to input a document called, `derived_data.csv` which is a CSV file of the Wilford Woodruff Journals and to output 3 parquet files called, 

 - `closest_people_df.parquet`,
 - `closest_places_df.parquet`, and
 - `closest_topics_df.parquet`.

 # Instructions on how to complete this purpose:

1. Ensure that the following files: (all 3 `a_scores_<variable>.py`), `b_scores.py`, `c_scores.py`, and `derived_data.csv` files are all adjacent to each other.

2. Ensure that the data_path variable in all the a files are correct and lead to `derived_data.csv`.

3. Ensure that you have enough processing power to run these files as they are extremely computationally demanding for a small computer.

4. **Run code** on `a_scores_people.py`, then after that has finished, **run code** on `a_scores_places.py`, and after that finishes, **run code** on `a_scores_topics.py`. The `a_scores_<variable>.py` files all output a file called `closest_<variable>.parquet` that is used in `b_scores.py`.

5. **Run code** on `b_scores.py`, and then on `c_scores.py`


## Under

This section explains the mathematical and algorithmic processes used in the script to generate recommendations based on similarities in people tags from the Wilford Woodruff Papers dataset.

### File: `a_score.py`: Data Preprocessing

1. **Data Loading**: The script begins by loading the journal entries from 'derived_data.csv', focusing on columns like 'text_only_transcript', 'people', 'places', and 'topics'.

2. **Data Cleaning**: Unwanted words and special characters (e.g., days of the week, months, and punctuation) are removed to ensure the quality of the text for analysis.

### Similarity Calculation

```{python}
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
```

3. **Term Frequency-Inverse Document Frequency (TF-IDF)**: The script employs TF-IDF to transform the textual data of each journal entry into a numerical format. This step helps in understanding the importance of words in the context of the whole dataset.

4. **Cosine Similarity**: With the TF-IDF matrix, the script calculates the cosine similarity between each pair of entries. This similarity measure is based on the angle between the TF-IDF vectors, thus capturing the likeness in content.

### Thresholding and Ranking

```{python}
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
```

5. **Percentile Calculation (People, Places, Topics)**: For either the People, Places, or Topics similarity matrix, the script calculates the 75th percentile. This percentile acts as a threshold to filter out lower similarity scores, focusing only on the most relevant matches.

6. **Threshold Application**: Entries with similarity scores below the 75th percentile threshold are discarded. This step refines the selection to only the most similar entries.

7. **Closest Matches Identification**: The script then identifies the top matches (closest indices) for each journal entry by text. This is achieved by ranking the entries based on their text similarity scores and selecting the top 4.

### Output
```{python}
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
```

8. **Output Generation**: The final output of this file is a matrix or a set of matrices, indicating the closest journal entries for each entry in the dataset. These results are saved in files (e.g., 'closest_people.parquet') for further analysis or direct use in the recommendation system.

### `b_scores.py` and `c_scores.py`: Backup Handling

- **Handling Insufficient Data**: In cases where the primary method does not yield sufficient results (duds), the script has a backup mechanism. It reprocesses the entries using raw similarity scores from the text content, ensuring that every entry gets a set of recommendations. (When a journal entry has no people/ places/ topics tags, the script uses the raw text similarity scores to generate recommendations)

By leveraging these mathematical and algorithmic approaches, the script efficiently processes the Wilford Woodruff Papers, uncovering meaningful connections across journal entries based on people, places, and topics.


# Streamlit App folder

This repository contains a folder called `streamlit_app`. This folder contains a file called `working_app.py` which is a python script to run a streamlit app that is an exploratory data analysis tool. To run this tool, right click the file, `working_app.py`, and press "Copy as Path". In the Command Prompt, type `streamlit run "<working_app.py path>"` where `<working_app.py path>` is the path to working_app.py in quotes. This will open the application.
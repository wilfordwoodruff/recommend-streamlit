## How It Works

This section explains the mathematical and algorithmic processes used in the script to generate recommendations based on similarities in people tags from the Wilford Woodruff Papers dataset.

### Data Preprocessing

1. **Data Loading**: The script begins by loading the journal entries from 'derived_data.csv', focusing on columns like 'text_only_transcript', 'people', 'places', and 'topics'.

2. **Data Cleaning**: Unwanted words and special characters (e.g., days of the week, months, and punctuation) are removed to ensure the quality of the text for analysis.

### Similarity Calculation

3. **Term Frequency-Inverse Document Frequency (TF-IDF)**: The script employs TF-IDF to transform the textual data of each journal entry into a numerical format. This step helps in understanding the importance of words in the context of the whole dataset.

4. **Cosine Similarity**: With the TF-IDF matrix, the script calculates the cosine similarity between each pair of entries. This similarity measure is based on the angle between the TF-IDF vectors, thus capturing the likeness in content.

### Thresholding and Ranking

5. **Percentile Calculation (People, Places, Topics)**: For either the People, Places, or Topics similarity matrix, the script calculates the 75th percentile. This percentile acts as a threshold to filter out lower similarity scores, focusing only on the most relevant matches.

6. **Threshold Application**: Entries with similarity scores below the 75th percentile threshold are discarded. This step refines the selection to only the most similar entries.

7. **Closest Matches Identification**: The script then identifies the top matches (closest indices) for each journal entry by text. This is achieved by ranking the entries based on their text similarity scores and selecting the top 4.

### Output

8. **Output Generation**: The final output is a matrix or a set of matrices, indicating the closest journal entries for each entry in the dataset. These results are saved in files (e.g., 'closest_people.parquet') for further analysis or direct use in the recommendation system.

### Backup Handling

- **Handling Insufficient Data**: In cases where the primary method does not yield sufficient results (duds), the script has a backup mechanism. It reprocesses the entries using raw similarity scores from the text content, ensuring that every entry gets a set of recommendations. (When a journal entry has no people/ places/ topics tags, the script uses the raw text similarity scores to generate recommendations)

By leveraging these mathematical and algorithmic approaches, the script efficiently processes the Wilford Woodruff Papers, uncovering meaningful connections across journal entries based on people, places, and topics.

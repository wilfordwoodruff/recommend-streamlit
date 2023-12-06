import polars as pl
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

people = pl.read_parquet("closest_people_df.parquet")
places = pl.read_parquet("closest_places_df.parquet")
topics = pl.read_parquet("closest_topics_df.parquet")

people = people\
    .with_columns([
        pl.col("internal_id").cast(pl.Int32),
        pl.col("closest_0").cast(pl.Int32),
        pl.col("closest_1").cast(pl.Int32),
        pl.col("closest_2").cast(pl.Int32),
        pl.col("closest_3").cast(pl.Int32)])

places = places\
    .with_columns([
        pl.col("internal_id").cast(pl.Int32),
        pl.col("closest_0").cast(pl.Int32),
        pl.col("closest_1").cast(pl.Int32),
        pl.col("closest_2").cast(pl.Int32),
        pl.col("closest_3").cast(pl.Int32)])

topics = topics\
    .with_columns([
        pl.col("internal_id").cast(pl.Int32),
        pl.col("closest_0").cast(pl.Int32),
        pl.col("closest_1").cast(pl.Int32),
        pl.col("closest_2").cast(pl.Int32),
        pl.col("closest_3").cast(pl.Int32)])

def fix_duplicates(people):

    fix1_people = people\
        .filter(pl.col("internal_id") != pl.col("closest_0"))\
        .rename({"closest_0": "closest_1", 
                "closest_1": "closest_0"})\
        .select(["internal_id", "closest_0", "closest_1", "closest_2", "closest_3"])\
        .filter(pl.col("internal_id") == pl.col("closest_0"))

    fix2_people = people\
        .filter(pl.col("internal_id") != pl.col("closest_0"))\
        .rename({"closest_0": "closest_1", 
                "closest_1": "closest_0"})\
        .select(["internal_id", "closest_0", "closest_1", "closest_2", "closest_3"])\
        .filter(pl.col("internal_id") != pl.col("closest_0"))\
        .rename({"closest_0": "closest_2", 
                "closest_2": "closest_0"})\
        .select(["internal_id", "closest_0", "closest_1", "closest_2", "closest_3"])\
        .filter(pl.col("internal_id") == pl.col("closest_0"))

    fix3_people = people\
        .filter(pl.col("internal_id") != pl.col("closest_0"))\
        .rename({"closest_0": "closest_1", 
                "closest_1": "closest_0"})\
        .select(["internal_id", "closest_0", "closest_1", "closest_2", "closest_3"])\
        .filter(pl.col("internal_id") != pl.col("closest_0"))\
        .rename({"closest_0": "closest_2", 
                "closest_2": "closest_0"})\
        .select(["internal_id", "closest_0", "closest_1", "closest_2", "closest_3"])\
        .filter(pl.col("internal_id") != pl.col("closest_0"))\
        .rename({"closest_0": "closest_3", 
                "closest_3": "closest_0"})\
        .select(["internal_id", "closest_0", "closest_1", "closest_2", "closest_3"])\
        .filter(pl.col("internal_id") == pl.col("closest_0"))

    fixes = pl.concat([fix1_people, fix2_people, fix3_people], how='vertical')

    people_yabueno = people.filter(~pl.col("internal_id").is_in(fixes["internal_id"]))

    people_fixed = pl.concat([people_yabueno, fixes], how='vertical').sort("internal_id")

    people_fixed.filter(pl.col("internal_id") != pl.col("closest_0"))\
        .rename(
            {"closest_0": "closest_1", 
            "closest_1": "closest_2",
            "closest_2": "closest_3",
            "closest_3": "closest_4"
            })\
        .with_columns(pl.col("internal_id").alias("closest_0"))\
        .select(["internal_id", "closest_0", "closest_1", "closest_2", "closest_3"])

    return people_fixed

people_fixed = fix_duplicates(people)
places_fixed = fix_duplicates(places)
topics_fixed = fix_duplicates(topics)

people_fixed.write_parquet("closest_people_df.parquet")
places_fixed.write_parquet("closest_places_df.parquet")
topics_fixed.write_parquet("closest_topics_df.parquet")

# people_fixed.write_csv("closest_people_df.csv")

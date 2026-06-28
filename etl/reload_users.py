from pyspark.sql import SparkSession, functions as F
from db.mongo_client import get_db

CLEAN_EVENTS_PATH = "data/processed/events_clean.parquet"
USER_RECENT_PATH = "data/processed/user_recent_events.parquet"

TOP_N_USERS = 50000


def get_spark():
    return (
        SparkSession.builder
        .appName("ReloadUsers")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def select_top_users(spark):
    events = spark.read.parquet(CLEAN_EVENTS_PATH)
    activity = events.groupBy("user_id").agg(F.count("*").alias("event_count"))
    top_users = (
        activity.orderBy(F.col("event_count").desc())
        .limit(TOP_N_USERS)
        .select("user_id")
    )
    return top_users


if __name__ == "__main__":
    spark = get_spark()

    print("Selecting top active users...")
    top_users = select_top_users(spark)
    top_users.cache()
    print("Top users selected:", top_users.count())

    user_recent = spark.read.parquet(USER_RECENT_PATH)
    filtered = user_recent.join(top_users, on="user_id", how="inner")

    print("Collecting user documents...")
    user_docs = [row.asDict(recursive=True) for row in filtered.collect()]
    print("User docs ready:", len(user_docs))

    db = get_db()
    print("Writing to Mongo (users)...")
    db.users.delete_many({})
    if user_docs:
        db.users.insert_many(user_docs, ordered=False)

    print("Done. Users in Mongo:", len(user_docs))
from pyspark.sql import SparkSession, functions as F
from db.mongo_client import get_db

CLEAN_EVENTS_PATH = "data/processed/events_clean.parquet"
USER_RECENT_PATH = "data/processed/user_recent_events.parquet"

TOP_N_USERS = 50000
MAX_PRODUCTS = 30000


def get_spark():
    return (
        SparkSession.builder
        .appName("LoadToMongo")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def select_top_users(spark):
    events = spark.read.parquet(CLEAN_EVENTS_PATH)

    # rank users by total event count, take the most active ones
    activity = events.groupBy("user_id").agg(F.count("*").alias("event_count"))
    top_users = (
        activity.orderBy(F.col("event_count").desc())
        .limit(TOP_N_USERS)
        .select("user_id")
    )
    return top_users


def select_top_products(events, top_users, max_products=MAX_PRODUCTS):
    # only products that the selected top users actually interacted with
    user_events = events.join(top_users, on="user_id", how="inner")

    product_activity = (
        user_events.groupBy("product_id", "category_code", "brand")
        .agg(
            F.count("*").alias("interaction_count"),
            F.avg("price").alias("avg_price"),
        )
    )

    top_products = (
        product_activity.orderBy(F.col("interaction_count").desc())
        .limit(max_products)
    )
    return top_products


def load_users(spark, top_users):
    user_recent = spark.read.parquet(USER_RECENT_PATH)
    filtered = user_recent.join(top_users, on="user_id", how="inner")
    return [row.asDict(recursive=True) for row in filtered.collect()]


def load_products(top_products):
    return [row.asDict(recursive=True) for row in top_products.collect()]


if __name__ == "__main__":
    spark = get_spark()

    print("Selecting top active users...")
    top_users = select_top_users(spark)
    top_users.cache()
    print("Top users selected:", top_users.count())

    events = spark.read.parquet(CLEAN_EVENTS_PATH)

    print("Selecting top products from those users' activity...")
    top_products = select_top_products(events, top_users)
    top_products.cache()
    print("Top products selected:", top_products.count())

    print("Collecting user documents...")
    user_docs = load_users(spark, top_users)
    print("User docs ready:", len(user_docs))

    print("Collecting product documents...")
    product_docs = load_products(top_products)
    print("Product docs ready:", len(product_docs))

    db = get_db()

    print("Writing to Mongo (users)...")
    db.users.delete_many({})
    if user_docs:
        db.users.insert_many(user_docs, ordered=False)

    print("Writing to Mongo (products)...")
    db.products.delete_many({})
    if product_docs:
        db.products.insert_many(product_docs, ordered=False)

    print("Done. Users:", db.users.count_documents({}), "Products:", db.products.count_documents({}))
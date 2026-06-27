from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

RAW_PATH = "data/raw/2019-Oct.csv.gz"
CLEAN_EVENTS_PATH = "data/processed/events_clean.parquet"
USER_RECENT_PATH = "data/processed/user_recent_events.parquet"

MAX_RECENT_EVENTS = 20


def get_spark():
    return (
        SparkSession.builder
        .appName("EcommerceClean")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def load_raw(spark, path=RAW_PATH):
    return spark.read.csv(path, header=True, inferSchema=True)


def clean_events(df):
    # drop rows missing core identifiers
    df = df.dropna(subset=["user_id", "product_id", "event_type", "event_time"])

    # fill missing descriptive fields instead of dropping (price/brand/category_code can be legitimately missing)
    df = df.fillna({"brand": "unknown", "category_code": "unknown"})

    # drop negative or null prices, they're not usable for recs
    df = df.filter((F.col("price").isNotNull()) & (F.col("price") >= 0))

    # dedup exact duplicate events
    df = df.dropDuplicates(["user_id", "product_id", "event_type", "event_time", "user_session"])

    # normalize category_code casing/whitespace
    df = df.withColumn("category_code", F.trim(F.lower(F.col("category_code"))))
    df = df.withColumn("brand", F.trim(F.lower(F.col("brand"))))

    return df


def build_user_recent_events(df, max_events=MAX_RECENT_EVENTS):
    # window: rank each user's events by most recent first
    w = Window.partitionBy("user_id").orderBy(F.col("event_time").desc())

    ranked = df.withColumn("rn", F.row_number().over(w))
    recent = ranked.filter(F.col("rn") <= max_events).drop("rn")

    # collapse into one row per user with a list of recent event structs
    user_recent = (
        recent.groupBy("user_id")
        .agg(
            F.collect_list(
                F.struct(
                    "product_id", "event_type", "category_code", "brand", "price", "event_time"
                )
            ).alias("recent_events")
        )
    )
    return user_recent


if __name__ == "__main__":
    spark = get_spark()

    print("Loading raw data...")
    raw = load_raw(spark)

    print("Cleaning events...")
    clean = clean_events(raw)
    clean.cache()
    print("Clean row count:", clean.count())

    print("Building per-user recent events...")
    user_recent = build_user_recent_events(clean)
    print("Unique users:", user_recent.count())

    print("Writing cleaned events to parquet...")
    clean.write.mode("overwrite").parquet(CLEAN_EVENTS_PATH)

    print("Writing user recent-events to parquet...")
    user_recent.write.mode("overwrite").parquet(USER_RECENT_PATH)

    print("Done.")
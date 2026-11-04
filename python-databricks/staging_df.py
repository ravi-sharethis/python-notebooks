def _path_exists(path):
    # Imported as a module, so the notebook's global `dbutils` isn't visible here.
    from pyspark.dbutils import DBUtils

    dbutils = DBUtils(SparkSession.getActiveSession())
    try:
        dbutils.fs.ls(path)
        print (f"{path} exists")
        return True
    except Exception:
        return False

import logging
from functools import wraps
from typing import List, Literal, Optional

from delta import DeltaTable
from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


def spark_read_parquet_or_delta(
    path: str, spark: Optional[SparkSession] = None
) -> DataFrame:
    if spark is None:
        spark = SparkSession.builder.appName("SparkReader").getOrCreate()

    if not DeltaTable.isDeltaTable(spark, path):
        return spark.read.parquet(path)
    return spark.read.format("delta").load(path)


def check_if_dataframe_exists(
    path: str, spark: SparkSession, write_format: Literal["parquet, delta"] = "delta"
) -> bool:
    if write_format not in ["parquet", "delta"]:
        raise ValueError("Expected 'delta' or 'parquet'")

    try:
        df = spark_read_parquet_or_delta(path, spark)
        # Spark Connect (serverless / shared clusters) reads lazily: a missing path
        # only fails once the plan is analyzed. Touch the schema so it fails here.
        _ = df.schema
        return True
    except Exception:
        return False


def stage_dataframe(
    partitionby: Optional[List[str]] = None,
    mode: Optional[Literal["overwrite", "append"]] = "overwrite",
    compression: Optional[Literal["snappy", "gzip"]] = "snappy",
    write_format: Optional[Literal["parquet", "delta"]] = "parquet",
    repartition: Optional[int] = None,
    unpersist_result_after_write: bool = False,
):
    """
    Stage a Spark DataFrame and return the saved data.

    - overwrite: reuse existing readable data without executing the function.
    - append: execute and append, then return the full saved dataset.
    - No write_path: execute without writing (spark is still required).
    - unpersist_result_after_write: release the produced DataFrame after
      a write attempt, whether the write succeeds or fails.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            write_path = kwargs.get("write_path")
            spark = kwargs.get("spark")

            # Find Spark session from args or self._spark.
            if not spark:
                for arg in args:
                    if isinstance(arg, SparkSession):
                        spark = arg
                        break
                    if hasattr(arg, "_spark") and isinstance(arg._spark, SparkSession):
                        spark = arg._spark
                        break
            if not spark:
                raise RuntimeError(
                    "[stage_dataframe] SparkSession not found. "
                    "Pass it explicitly or ensure the class has `self._spark`."
                )

            def write_produced_dataframe(produced_df, save_mode):
                primary_error = None
                try:
                    write_df = produced_df
                    if repartition:
                        write_df = produced_df.repartition(repartition)
                        logger.debug(
                            f"=-=-=- Repartitioned DataFrame to {repartition} partition(s)."
                        )

                    writer = (
                        write_df.write.format(write_format)
                        .mode(save_mode)
                        .option("compression", compression)
                    )
                    if partitionby:
                        writer = writer.partitionBy(*partitionby)
                    writer.save(write_path)
                except BaseException as exc:
                    primary_error = exc
                    raise
                finally:
                    if unpersist_result_after_write:
                        try:
                            produced_df.unpersist()
                        except Exception:
                            if primary_error is None:
                                raise
                            logger.exception(
                                "Failed to unpersist a staged DataFrame after "
                                "the primary write operation failed."
                            )

            if write_path:
                path_exists = check_if_dataframe_exists(
                    write_path, spark, write_format=write_format
                )

                if path_exists:
                    if mode == "overwrite":
                        logger.info(
                            f"=-=-=- File exists at {write_path}. Skipping execution and reading existing DataFrame."
                        )
                        return spark_read_parquet_or_delta(write_path, spark)

                    elif mode == "append":
                        logger.info(
                            f"=-=-=- File exists at {write_path}. Running {func.__name__} and appending results."
                        )
                        produced_df = func(*args, **kwargs)
                        write_produced_dataframe(produced_df, "append")
                        return spark_read_parquet_or_delta(write_path, spark)

                logger.debug(
                    f"=-=-=- Path does not exist. Running {func.__name__} and writing to {write_path}."
                )
                produced_df = func(*args, **kwargs)
                write_produced_dataframe(produced_df, mode)
                logger.info(f"Data successfully written to {write_path}.")
                return spark_read_parquet_or_delta(write_path, spark)

            logger.debug(
                f"=-=-=- No write_path provided. Running {func.__name__} without writing."
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator

""" sample usage 

    # spark & write_path are required function signatures

    @stage_dataframe(write_format="parquet")
    def pairs_fn(left, right, spark=None, write_path=f"{local_path}/pairs"):
        return (
            left.join(right, on="key")
            .filter(F.col("node_a") < F.col("node_b"))
            .select(
                "node_a",
                "node_b",
                F.lit(via_label).alias("via_type"),
                F.col("key").cast("string").alias("via_key"),
            )
        )
    
    pairs = pairs_fn(
	left, right,
        spark=spark,
        write_path=f"{local_path}/pairs",
    )
"""


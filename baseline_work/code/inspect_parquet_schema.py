import requests
import io
import pyarrow.parquet as pq

# The parquet file can be read via HTTP or huggingface_hub
from huggingface_hub import hf_hub_download

print("Downloading single small parquet shard header or table from nebula/OpenSDI_test...")
path = hf_hub_download(
    repo_id="nebula/OpenSDI_test",
    filename="data/sd15-00000-of-00019.parquet",
    repo_type="dataset",
    cache_dir="I:/icvgip workshop/baselines/model_cache"
)
print("Downloaded shard to:", path)
parquet_file = pq.ParquetFile(path)
print("Schema:")
print(parquet_file.schema)
print("Num rows in this shard:", parquet_file.metadata.num_rows)

table = parquet_file.read_row_group(0, columns=["key", "label"])
print("Sample rows:")
for i in range(min(5, table.num_rows)):
    print(f"Row {i}: key={table['key'][i]}, label={table['label'][i]}")


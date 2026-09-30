from io import StringIO
import pandas as pd

# Toy SCADA export, as we don't have a real SCADA system to use at the moment...
csv_file = StringIO(
    "time,tag,value\n"
    "2026-09-29T08:00:00Z,temperature,72.5\n"
    "2026-09-29T08:00:01Z,temperature,73.0\n"
)

df = pd.read_csv(csv_file, parse_dates=["time"])
print(df["value"].mean())

# Toy WMS export: same concept, different delimiter (a "tab" instead of a comma).
tsv_file = StringIO("sku\tqty\nA100\t12\nB200\t7\n")
stock = pd.read_csv(tsv_file, sep="\t", dtype={"sku": str})  # notice that we still use read_csv, but specify a different separator
print(stock["qty"].sum())

# Toy legacy ERP export: two fields, each five characters wide.
fixed_file = StringIO("A100 00012\nB200 00007\n")
stock = pd.read_fwf(  # read fixed-width formatted lines into DataFrame
    fixed_file,
    widths=[5, 5],
    names=["sku", "qty"],
    dtype={"sku": str},
)
print(stock.to_dict("records"))

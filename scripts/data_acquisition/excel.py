from io import BytesIO
import pandas as pd

# Create a genuine XLSX workbook containing toy stock data.
workbook = BytesIO()
pd.DataFrame({
    "sku": ["A100", "B200"],
    "qty": [12, 7],
}).to_excel(
    workbook,
    sheet_name="Stock",
    index=False,
    engine="openpyxl", # tell pandas to use this package to read the Excel
)

workbook.seek(0)  # technicality: reset the file pointer to the beginning of the file, so that we can read it back

# Read it as a table.
stock = pd.read_excel(
    workbook,
    sheet_name="Stock",
    engine="openpyxl",
    dtype={"sku": str},
)
print(stock["qty"].sum())  # 19

# Read individual worksheet cells instead.
from openpyxl import load_workbook

workbook.seek(0)
wb = load_workbook(workbook, read_only=True, data_only=True)
print(wb["Stock"]["A2"].value)  # A100
wb.close()
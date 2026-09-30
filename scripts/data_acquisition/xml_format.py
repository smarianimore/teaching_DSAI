from io import StringIO
import xml.etree.ElementTree as ET

xml_file = StringIO("""
<orders>
  <order id="PO1">
    <sku>A100</sku>
    <qty>12</qty>
  </order>
  <order id="PO2">
    <sku>B200</sku>
    <qty>7</qty>
  </order>
</orders>
""")

root = ET.parse(xml_file).getroot()

orders = [
    {
        "id": order.attrib["id"],
        "sku": order.findtext("sku"),
        "qty": int(order.findtext("qty")),
    }
    for order in root.findall("order")
]

print(sum(order["qty"] for order in orders))  # 19
import httpx
import xml.etree.ElementTree as ET

soap_response = """
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">
  <s:Body>
    <GetStockResponse xmlns="urn:toy">
      <Quantity>12</Quantity>
    </GetStockResponse>
  </s:Body>
</s:Envelope>
"""

transport = httpx.MockTransport(
    lambda request: httpx.Response(
        200, text=soap_response,
        headers={"Content-Type": "text/xml"},
    )
)

request_xml = """
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">
  <s:Body><GetStock xmlns="urn:toy"><Sku>A100</Sku></GetStock></s:Body>
</s:Envelope>
"""

with httpx.Client(transport=transport) as client:
    response = client.post(
        "https://erp.example/soap",
        content=request_xml,
        headers={
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": '"urn:toy/GetStock"',
        },
    )
    response.raise_for_status()

root = ET.fromstring(response.content)
print(int(root.findtext(".//{urn:toy}Quantity")))  # 12
from io import StringIO
import json

# Toy CRM export: one JSON document.
crm_file = StringIO("""
{
  "accounts": [
    {"id": "C001", "name": "Acme", "revenue": 1200},
    {"id": "C002", "name": "Beta", "revenue": 800}
  ]
}
""")

data = json.load(crm_file)
print(sum(a["revenue"] for a in data["accounts"]))  # 2000

# Toy IoT log: one JSON object per line.
event_file = StringIO(
    '{"sensor":"T1","value":72.5}\n'
    '{"sensor":"T1","value":73.0}\n'
)

values = [
    json.loads(line)["value"]
    for line in event_file
    if line.strip()
]
print(sum(values) / len(values))  # 72.75
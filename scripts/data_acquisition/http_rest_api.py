import httpx

def toy_crm(request):  # simulates a CRM server API
    if request.url.path == "/accounts":
        return httpx.Response(200, json={
            "value": [
                {"id": "C001", "name": "Acme", "revenue": 1200},
                {"id": "C002", "name": "Beta", "revenue": 800},
            ]
        })
    return httpx.Response(404)

with httpx.Client(
    transport=httpx.MockTransport(toy_crm),  # fake transport layer (e.g. TCP)
    base_url="https://crm.example",
) as client:
    response = client.get("/accounts")
    response.raise_for_status()
    accounts = response.json()["value"]

print(sum(account["revenue"] for account in accounts))  # 2000
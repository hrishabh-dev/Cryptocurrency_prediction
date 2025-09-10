import requests

URL = "http://127.0.0.1:5000/predict_api"

# 50 diverse test cases
test_cases = [
    {"24h_volume": 5000000, "mkt_cap": 300000000, "1h": 0.01, "price": 5.5},
    {"24h_volume": 12000000, "mkt_cap": 950000000, "1h": 0.01, "price": 35.6},
    {"24h_volume": 25000000, "mkt_cap": 1200000000, "1h": -0.02, "price": 12.85},
    {"24h_volume": 18000000, "mkt_cap": 750000000, "1h": 0.0, "price": 49.22},
    {"24h_volume": 30000000, "mkt_cap": 2000000000, "1h": -0.03, "price": 120.5},
    {"24h_volume": 60000000, "mkt_cap": 500000000, "1h": 0.04, "price": 8.9},
    {"24h_volume": 100000000, "mkt_cap": 1500000000, "1h": 0.01, "price": 45.3},
    {"24h_volume": 200000000, "mkt_cap": 800000000, "1h": 0.05, "price": 22.1},
    {"24h_volume": 400000000, "mkt_cap": 1200000000, "1h": -0.01, "price": 55.2},
    {"24h_volume": 750000000, "mkt_cap": 1200000000, "1h": 0.02, "price": 41.25},
    {"24h_volume": 950000000, "mkt_cap": 1400000000, "1h": -0.01, "price": 56.78},
    {"24h_volume": 1250000000, "mkt_cap": 2000000000, "1h": 0.0, "price": 15.42},
    {"24h_volume": 850000000, "mkt_cap": 900000000, "1h": 0.03, "price": 67.1},
    {"24h_volume": 1500000000, "mkt_cap": 2500000000, "1h": 0.02, "price": 100.0},
    {"24h_volume": 3000000000, "mkt_cap": 4000000000, "1h": 0.05, "price": 250.5},
    {"24h_volume": 5000000000, "mkt_cap": 8000000000, "1h": 0.01, "price": 310.2},
    {"24h_volume": 7000000, "mkt_cap": 200000000, "1h": -0.05, "price": 3.25},
    {"24h_volume": 9000000, "mkt_cap": 300000000, "1h": 0.02, "price": 6.7},
    {"24h_volume": 15000000, "mkt_cap": 400000000, "1h": 0.01, "price": 9.8},
    {"24h_volume": 35000000, "mkt_cap": 600000000, "1h": -0.02, "price": 15.9},
    {"24h_volume": 100000000, "mkt_cap": 1200000000, "1h": 0.03, "price": 45.7},
    {"24h_volume": 200000000, "mkt_cap": 2200000000, "1h": 0.01, "price": 78.5},
    {"24h_volume": 500000000, "mkt_cap": 3000000000, "1h": -0.04, "price": 99.2},
    {"24h_volume": 800000000, "mkt_cap": 3500000000, "1h": 0.05, "price": 120.0},
    {"24h_volume": 1000000000, "mkt_cap": 5000000000, "1h": -0.01, "price": 250.2},
    {"24h_volume": 2500000, "mkt_cap": 200000000, "1h": 0.02, "price": 2.5},
    {"24h_volume": 4000000, "mkt_cap": 150000000, "1h": 0.0, "price": 1.2},
    {"24h_volume": 10000000, "mkt_cap": 450000000, "1h": -0.03, "price": 7.1},
    {"24h_volume": 20000000, "mkt_cap": 900000000, "1h": 0.04, "price": 12.3},
    {"24h_volume": 50000000, "mkt_cap": 1800000000, "1h": 0.01, "price": 33.4},
    {"24h_volume": 100000000, "mkt_cap": 2500000000, "1h": -0.02, "price": 56.0},
    {"24h_volume": 250000000, "mkt_cap": 3000000000, "1h": 0.05, "price": 87.4},
    {"24h_volume": 400000000, "mkt_cap": 4500000000, "1h": -0.01, "price": 110.6},
    {"24h_volume": 600000000, "mkt_cap": 5000000000, "1h": 0.03, "price": 140.9},
    {"24h_volume": 1200000000, "mkt_cap": 7000000000, "1h": -0.02, "price": 175.3},
    {"24h_volume": 1800000000, "mkt_cap": 8500000000, "1h": 0.01, "price": 210.0},
    {"24h_volume": 2500000000, "mkt_cap": 10000000000, "1h": 0.0, "price": 280.7},
    {"24h_volume": 3200000000, "mkt_cap": 12000000000, "1h": 0.05, "price": 310.8},
    {"24h_volume": 4000000000, "mkt_cap": 14000000000, "1h": -0.03, "price": 350.5},
    {"24h_volume": 4500000000, "mkt_cap": 16000000000, "1h": 0.02, "price": 375.9},
    {"24h_volume": 5200000000, "mkt_cap": 18000000000, "1h": -0.04, "price": 400.0},
    {"24h_volume": 6000000000, "mkt_cap": 20000000000, "1h": 0.01, "price": 450.0},
    {"24h_volume": 7000000000, "mkt_cap": 22000000000, "1h": 0.02, "price": 500.0},
    {"24h_volume": 8000000000, "mkt_cap": 25000000000, "1h": 0.0, "price": 550.0},
    {"24h_volume": 9000000000, "mkt_cap": 28000000000, "1h": -0.05, "price": 600.0},
    {"24h_volume": 10000000000, "mkt_cap": 30000000000, "1h": 0.03, "price": 650.0},
    {"24h_volume": 11000000000, "mkt_cap": 32000000000, "1h": 0.01, "price": 700.0},
    {"24h_volume": 12000000000, "mkt_cap": 35000000000, "1h": -0.02, "price": 750.0},
    {"24h_volume": 13000000000, "mkt_cap": 37000000000, "1h": 0.04, "price": 800.0},
    {"24h_volume": 14000000000, "mkt_cap": 40000000000, "1h": 0.02, "price": 850.0},
]

for i, case in enumerate(test_cases, 1):
    response = requests.post(URL, json=case)
    print(f"Test Case {i}: {case}")
    if response.ok:
        text = response.text
        start = text.find("Predicted liquidity")
        snippet = text[start:start+120] if start != -1 else text[:120]
        print("➡", snippet, "\n")
    else:
        print("❌ Request failed\n")

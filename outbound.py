import os
import requests
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()
API_KEY = os.getenv("EXOTEL_API_KEY")
API_TOKEN = os.getenv("EXOTEL_API_TOKEN")
ACCOUNT_SID = os.getenv("EXOTEL_ACCOUNT_SID")
EXOPHONE = os.getenv("EXOTEL_EXOPHONE")

# 1. Who do you want to call? (Must be a registered number if on Exotel Trial)
CUSTOMER_NUMBER = "+919277199206" # Replace with your number!

# 2. What is your Voicebot Flow App ID?
# Go to Exotel Dashboard -> App Bazaar -> Your Voicebot Flow. 
# Look at the URL in your browser, the number at the end is the APP_ID (e.g., 123456)
APP_ID = "1320322"

# Exotel API URL
url = f"https://api.exotel.com/v1/Accounts/{ACCOUNT_SID}/Calls/connect.json"

payload = {
    "From": CUSTOMER_NUMBER,
    "CallerId": EXOPHONE,
    "Url": f"http://my.exotel.com/{ACCOUNT_SID}/exoml/start_voice/{APP_ID}"
}

print(f"📞 Calling {CUSTOMER_NUMBER}...")
resp = requests.post(url, data=payload, auth=(API_KEY, API_TOKEN))

if resp.status_code == 200:
    print("✅ Call initiated successfully!")
    print(resp.json())
else:
    print(f"❌ Failed to initiate call (Code: {resp.status_code})")
    print(resp.text)

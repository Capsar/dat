import os
import requests

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ['TELEGRAM_API_KEY']
CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

api_url = f'https://api.telegram.org/bot{API_KEY}/sendMessage'


def send_telegram_message(message: str) -> bool:
    params = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(api_url, json=params)
    return response.status_code == 200

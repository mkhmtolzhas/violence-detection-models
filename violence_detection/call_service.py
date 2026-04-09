import http.client
import json

from violence_detection.settings import settings


def normalize_phone_number(phone_number: str) -> str:
    normalized_phone_number = phone_number.strip()
    if normalized_phone_number and not normalized_phone_number.startswith("+"):
        normalized_phone_number = f"+{normalized_phone_number}"
    return normalized_phone_number


def call_to_number(phone_number: str, text: str | None = None):
    if not phone_number:
        raise RuntimeError("ALERT_PHONE_NUMBER is not set")

    conn = http.client.HTTPSConnection(settings.api_url)
    payload = json.dumps(
        {
            "messages": [
                {
                    "destinations": [{"to": normalize_phone_number(phone_number)}],
                    "from": settings.voice_call_from,
                    "language": "en",
                    "text": text or settings.voice_call_text,
                    "voice": {
                        "name": "Joanna",
                        "gender": "female",
                    },
                }
            ]
        }
    )
    headers = {
        "Authorization": f"App {settings.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    conn.request("POST", "/tts/3/advanced", payload, headers)
    res = conn.getresponse()
    data = res.read()
    response_text = data.decode("utf-8")
    print(response_text)

    if res.status >= 400:
        raise RuntimeError(f"Voice call failed with status {res.status}: {response_text}")

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from ingest.common import Record, write_records

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MIN_WORDS = 150
QUOTE_MARKERS = ("\nOn ", "\n> ", "\n-----Original Message-----", "\nFrom: ")


def strip_quoted_reply(body: str) -> str:
    cut_points = [body.find(marker) for marker in QUOTE_MARKERS if marker in body]
    cut_points = [p for p in cut_points if p > 0]
    if cut_points:
        body = body[: min(cut_points)]
    return body.strip()


def decode_body(payload: dict) -> str:
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        for part in payload["parts"]:
            text = decode_body(part)
            if text:
                return text
        return ""
    data = payload.get("body", {}).get("data")
    if not data:
        return ""
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")


def message_to_record(msg: dict) -> Record | None:
    headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
    subject = headers.get("Subject", "(no subject)")
    date = headers.get("Date", "")
    body = strip_quoted_reply(decode_body(msg["payload"]))
    if len(body.split()) < MIN_WORDS:
        return None
    return Record(source="sent_mail", title=subject, reference=f"gmail:{msg['id']}", date=date, text=body)


def get_credentials(token_path: str, client_secret_path: str):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)
        Path(token_path).write_text(creds.to_json())
    return creds


def ingest_gmail(token_path: str, client_secret_path: str, out_path: str, max_messages: int = 2000) -> int:
    from googleapiclient.discovery import build

    creds = get_credentials(token_path, client_secret_path)
    service = build("gmail", "v1", credentials=creds)

    records = []
    request = service.users().messages().list(userId="me", q="in:sent", maxResults=min(500, max_messages))
    while request is not None and len(records) < max_messages:
        response = request.execute()
        for item in response.get("messages", []):
            msg = service.users().messages().get(userId="me", id=item["id"], format="full").execute()
            record = message_to_record(msg)
            if record:
                records.append(record)
        request = service.users().messages().list_next(request, response)

    write_records(records, out_path)
    return len(records)


if __name__ == "__main__":
    token_path = sys.argv[1] if len(sys.argv) > 1 else "data/gmail_token.json"
    client_secret_path = sys.argv[2] if len(sys.argv) > 2 else "client_secret.json"
    out = sys.argv[3] if len(sys.argv) > 3 else "data/sent_mail.json"
    count = ingest_gmail(token_path, client_secret_path, out)
    print(f"Ingested {count} sent emails to {out}")

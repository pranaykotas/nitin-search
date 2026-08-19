import base64

from ingest.gmail import decode_body, message_to_record, strip_quoted_reply


def test_strip_quoted_reply_cuts_at_on_wrote():
    body = "My actual reply here.\n\nOn Mon, Jan 1, 2024, Someone <x@example.com> wrote:\n> quoted text"
    assert strip_quoted_reply(body) == "My actual reply here."


def test_strip_quoted_reply_cuts_at_original_message_marker():
    body = "My reply.\n\n-----Original Message-----\nFrom: someone"
    assert strip_quoted_reply(body) == "My reply."


def test_strip_quoted_reply_returns_full_body_when_no_marker():
    body = "Just a plain reply with no quoting."
    assert strip_quoted_reply(body) == body


def test_decode_body_handles_plain_text_part():
    text = "Hello, this is the email body."
    encoded = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
    payload = {"parts": [{"mimeType": "text/plain", "body": {"data": encoded}}]}
    assert decode_body(payload) == text


def test_decode_body_handles_single_part_message():
    text = "Simple non-multipart body."
    encoded = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
    payload = {"body": {"data": encoded}}
    assert decode_body(payload) == text


def test_message_to_record_filters_short_messages():
    short_text = "Too short."
    encoded = base64.urlsafe_b64encode(short_text.encode("utf-8")).decode("ascii")
    msg = {
        "id": "abc123",
        "payload": {
            "headers": [{"name": "Subject", "value": "Re: quick note"}, {"name": "Date", "value": "Mon, 1 Jan 2024 10:00:00 +0000"}],
            "body": {"data": encoded},
        },
    }
    assert message_to_record(msg) is None


def test_message_to_record_builds_record_for_substantive_message():
    long_text = " ".join(["word"] * 200)
    encoded = base64.urlsafe_b64encode(long_text.encode("utf-8")).decode("ascii")
    msg = {
        "id": "abc123",
        "payload": {
            "headers": [{"name": "Subject", "value": "Thoughts on federalism"}, {"name": "Date", "value": "Mon, 1 Jan 2024 10:00:00 +0000"}],
            "body": {"data": encoded},
        },
    }
    record = message_to_record(msg)
    assert record is not None
    assert record.source == "sent_mail"
    assert record.title == "Thoughts on federalism"
    assert record.reference == "gmail:abc123"


def test_strip_quoted_reply_quote_marker_at_index_zero():
    # Critical bug 1: quote marker at index 0 should result in empty string,
    # not return the full quoted body unstripped
    quoted_text = " ".join(["quote"] * 200)
    body = f"\nOn Mon, Jan 1, 2024, Someone wrote:\n{quoted_text}"
    result = strip_quoted_reply(body)
    # Should be empty or near-empty, not the full quoted text
    assert len(result.split()) < 150


def test_decode_body_rejects_html_only_message():
    # Critical bug 2: HTML-only messages should not be decoded
    html_text = "<html><body>This is <b>HTML</b> content</body></html>"
    encoded = base64.urlsafe_b64encode(html_text.encode("utf-8")).decode("ascii")
    payload = {"mimeType": "text/html", "body": {"data": encoded}}
    result = decode_body(payload)
    # Should return empty string, not the raw HTML
    assert result == ""


def test_strip_quoted_reply_detects_forwarded_message_marker():
    # Important 3: Gmail forwarded message marker should be detected
    body = "My response here.\n\n---------- Forwarded message ---------\nFrom: someone\n> forwarded content"
    result = strip_quoted_reply(body)
    assert result == "My response here."


def test_decode_body_handles_invalid_base64():
    # Important 4: malformed base64 should not crash, should return empty string
    payload = {"body": {"data": "!!!invalid base64!!!"}}
    result = decode_body(payload)
    # Should return empty string, not raise an exception
    assert result == ""

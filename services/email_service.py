"""
Email delivery service using SMTP.
"""
from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Iterable
import base64

from config.settings import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    SMTP_FROM_EMAIL,
    SMTP_USE_TLS,
    GMAIL_CLIENT_ID,
    GMAIL_CLIENT_SECRET,
    GMAIL_REFRESH_TOKEN,
    GMAIL_SENDER_EMAIL,
)

try:
    from google.oauth2.credentials import Credentials  # type: ignore
    from google.auth.transport.requests import Request  # type: ignore
    from googleapiclient.discovery import build  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Credentials = None  # type: ignore
    Request = None  # type: ignore
    build = None  # type: ignore


def _is_configured() -> bool:
    return bool(SMTP_HOST and SMTP_FROM_EMAIL)


def _is_gmail_configured() -> bool:
    return all(
        [
            GMAIL_CLIENT_ID,
            GMAIL_CLIENT_SECRET,
            GMAIL_REFRESH_TOKEN,
            GMAIL_SENDER_EMAIL,
            Credentials,
            Request,
            build,
        ]
    )


def _send_via_gmail(
    recipients: list[str],
    subject: str,
    html_body: str,
    text_body: str | None = None,
) -> bool:
    if not _is_gmail_configured():
        return False
    try:
        creds = Credentials(
            token=None,
            refresh_token=GMAIL_REFRESH_TOKEN,
            client_id=GMAIL_CLIENT_ID,
            client_secret=GMAIL_CLIENT_SECRET,
            token_uri="https://oauth2.googleapis.com/token",
        )
        creds.refresh(Request())
        service = build("gmail", "v1", credentials=creds)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = GMAIL_SENDER_EMAIL
        msg["To"] = ", ".join(recipients)

        if text_body:
            msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw_message}).execute()

        try:
            service.close()
        except Exception:
            pass
        return True
    except Exception as exc:
        print(f"EMAIL: Gmail send failed, falling back to SMTP if configured. Error: {exc}")
        return False


def _send_via_smtp(
    recipients: list[str],
    subject: str,
    html_body: str,
    text_body: str | None = None,
) -> bool:
    if not _is_configured():
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM_EMAIL
        msg["To"] = ", ".join(recipients)

        if text_body:
            msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            if SMTP_USE_TLS:
                server.starttls()
            if SMTP_USERNAME and SMTP_PASSWORD:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, recipients, msg.as_string())
        return True
    except Exception as exc:
        print(f"EMAIL: Failed to send via SMTP to {recipients}: {exc}")
        return False


def send_email(
    recipients: Iterable[str],
    subject: str,
    html_body: str,
    text_body: str | None = None,
) -> bool:
    """
    Send an email to the provided recipients.

    Args:
        recipients: iterable of recipient email addresses
        subject: subject line
        html_body: HTML message body
        text_body: optional plain-text fallback body

    Returns:
        True if the email was dispatched (or configuration missing), False on error.
    """
    recipients = [addr.strip() for addr in recipients if addr and addr.strip()]
    if not recipients:
        return False

    if _is_gmail_configured():
        if _send_via_gmail(recipients, subject, html_body, text_body):
            return True

    if _send_via_smtp(recipients, subject, html_body, text_body):
        return True

    print("EMAIL: No email transport configured; skipping send.")
    return False


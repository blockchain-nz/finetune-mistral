# email_sender.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")

def send_notification(subject, html_body):
    if not RESEND_API_KEY:
        raise ValueError("Missing RESEND_API_KEY")

    response = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
        json={
            "from": EMAIL_FROM,
            "to": [EMAIL_TO],
            "subject": subject,
            "html": html_body,
        },
    )

    if response.status_code == 200:
        print("✅ Email sent successfully.")
    else:
        print(f"❌ Failed to send email: {response.status_code}, {response.text}")

if __name__ == "__main__":
    subject = "Email Notification"
    html_body = "<p>This is a test email.</p>"
    send_notification(subject, html_body)
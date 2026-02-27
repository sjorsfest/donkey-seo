"""Email verification message rendering and delivery."""

from __future__ import annotations

import html
import logging

from app.config import settings
from app.integrations.resend_email import ResendEmailClient

logger = logging.getLogger(__name__)


def _first_name(full_name: str | None) -> str:
    normalized = (full_name or "").strip()
    if not normalized:
        return "there"
    return normalized.split(" ", 1)[0]


def render_verification_email_html(*, first_name: str, verification_url: str) -> str:
    """Render a DonkeySEO-themed verification email HTML."""
    safe_name = html.escape(first_name)
    safe_url = html.escape(verification_url)
    return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Nunito:wght@400;600;700;800&display=swap');
    </style>
  </head>
  <body style="margin:0;padding:0;font-family:'Nunito','Segoe UI',sans-serif;background:#70ac96;">
    <table width="100%" cellpadding="0" cellspacing="0" style="padding:36px 16px;background:#70ac96;">
      <tr>
        <td align="center">
          <table width="100%" cellpadding="0" cellspacing="0" style="max-width:520px;background:#ffffff;border:2px solid #0f3d3e;border-radius:20px;box-shadow:4px 4px 0 #0f3d3e;">
            <tr>
              <td style="padding:28px 30px 14px;background:#ffd641;border-bottom:2px solid #0f3d3e;border-radius:18px 18px 0 0;">
                <p style="margin:0;font-family:'Fredoka','Nunito',sans-serif;font-weight:700;font-size:26px;line-height:1.1;color:#0f3d3e;">
                  DonkeySEO
                </p>
                <p style="margin:8px 0 0;font-size:14px;line-height:20px;color:#0f3d3e;">
                  Verify your email to finish setting up your account.
                </p>
              </td>
            </tr>
            <tr>
              <td style="padding:28px 30px 24px;">
                <h2 style="margin:0 0 14px;font-family:'Fredoka','Nunito',sans-serif;font-size:24px;font-weight:600;color:#0f3d3e;">
                  Hey {safe_name},
                </h2>
                <p style="margin:0 0 22px;font-size:16px;line-height:26px;color:#1f2937;">
                  Confirm your email address by clicking the button below.
                </p>
                <table width="100%" cellpadding="0" cellspacing="0">
                  <tr>
                    <td align="center" style="padding:6px 0 24px;">
                      <a href="{safe_url}" style="display:inline-block;padding:13px 28px;background:#86c4ad;color:#0f3d3e;font-family:'Nunito','Segoe UI',sans-serif;font-size:16px;font-weight:800;text-decoration:none;border-radius:12px;border:2px solid #0f3d3e;box-shadow:3px 3px 0 #0f3d3e;">
                        Verify Email
                      </a>
                    </td>
                  </tr>
                </table>
                <p style="margin:0 0 12px;font-size:13px;line-height:20px;color:#4b5563;">
                  If you did not create this account, you can ignore this email.
                </p>
                <p style="margin:0;font-size:12px;line-height:19px;color:#6b7280;word-break:break-all;">
                  Or copy and paste this link:<br>
                  <a href="{safe_url}" style="color:#0f3d3e;">{safe_url}</a>
                </p>
              </td>
            </tr>
            <tr>
              <td style="padding:16px 30px;background:#fff9e1;border-top:1px solid #d1d5db;border-radius:0 0 18px 18px;">
                <p style="margin:0;font-size:12px;line-height:18px;color:#4b5563;">
                  donkeyseo.io
                </p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


def render_email_verification_result_html(*, success: bool, message: str) -> str:
    """Render a minimal browser page after verification link click."""
    title = "Email verified" if success else "Verification failed"
    badge_color = "#86c4ad" if success else "#ef4444"
    safe_message = html.escape(message)
    return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">
  </head>
  <body style="margin:0;padding:0;font-family:'Nunito','Segoe UI',sans-serif;background:#70ac96;">
    <table width="100%" height="100%" cellpadding="0" cellspacing="0" style="min-height:100vh;padding:16px;">
      <tr>
        <td align="center" valign="middle">
          <div style="max-width:520px;background:#ffffff;border:2px solid #0f3d3e;border-radius:18px;box-shadow:4px 4px 0 #0f3d3e;padding:28px;">
            <p style="display:inline-block;margin:0 0 12px;padding:4px 10px;border-radius:999px;background:{badge_color};color:#ffffff;font-size:12px;font-weight:700;">
              DonkeySEO
            </p>
            <h1 style="margin:0 0 12px;font-family:'Fredoka','Nunito',sans-serif;font-size:28px;color:#0f3d3e;">{title}</h1>
            <p style="margin:0;font-size:16px;line-height:24px;color:#1f2937;">{safe_message}</p>
          </div>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


async def send_verification_email(
    *,
    email: str,
    name: str | None,
    verification_url: str,
) -> None:
    """Send verification email through Resend."""
    if not settings.resend_api_key:
        logger.info(
            "Skipping verification email send because RESEND_API_KEY is not configured",
            extra={"email": email},
        )
        return

    html_body = render_verification_email_html(
        first_name=_first_name(name),
        verification_url=verification_url,
    )
    async with ResendEmailClient() as client:
        await client.send_email(
            from_email=settings.resend_from_email,
            to_email=email,
            subject="Verify your email - DonkeySEO",
            html=html_body,
        )
    logger.info("Verification email sent", extra={"email": email})

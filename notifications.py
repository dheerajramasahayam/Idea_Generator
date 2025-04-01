import smtplib
import logging
from email.message import EmailMessage
import config # For email settings

def send_summary_email(subject, body):
    """Sends an email summary using configured SMTP settings."""
    if not config.ENABLE_EMAIL_NOTIFICATIONS:
        logging.info("Email notifications are disabled. Skipping email send.")
        return

    # Validate required settings are present
    required_settings = [
        config.SMTP_SERVER, config.SMTP_PORT, config.SMTP_USER,
        config.SMTP_PASSWORD, config.EMAIL_SENDER, config.EMAIL_RECIPIENT
    ]
    if not all(required_settings):
        logging.error("Cannot send email. Missing one or more required SMTP settings in config/.env")
        return

    logging.info(f"Attempting to send email summary to {config.EMAIL_RECIPIENT}...")

    msg = EmailMessage()
    msg['Subject'] = subject
    # Ensure sender format is correct (e.g., "Display Name <email@addr>")
    # If only email is provided in .env, use it for both parts
    sender_formatted = config.EMAIL_SENDER
    if '<' not in sender_formatted:
         sender_formatted = f"{config.EMAIL_SENDER} <{config.EMAIL_SENDER}>"

    msg['From'] = sender_formatted
    msg['To'] = config.EMAIL_RECIPIENT
    msg.set_content(body)

    try:
        # Connect, login, send, quit
        # Use SMTP_SSL for port 465, otherwise assume STARTTLS for 587/other
        if config.SMTP_PORT == 465:
            with smtplib.SMTP_SSL(config.SMTP_SERVER, config.SMTP_PORT) as server:
                server.login(config.SMTP_USER, config.SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                server.starttls() # Secure connection
                server.login(config.SMTP_USER, config.SMTP_PASSWORD)
                server.send_message(msg)
        logging.info("Email summary sent successfully.")

    except smtplib.SMTPAuthenticationError:
        logging.error("SMTP Authentication Error: Check SMTP_USER and SMTP_PASSWORD.")
    except smtplib.SMTPConnectError:
        logging.error(f"SMTP Connection Error: Could not connect to {config.SMTP_SERVER}:{config.SMTP_PORT}.")
    except smtplib.SMTPSenderRefused:
         logging.error(f"SMTP Sender Refused: Server didn't accept FROM address '{config.EMAIL_SENDER}'.")
    except smtplib.SMTPRecipientsRefused:
         logging.error(f"SMTP Recipient Refused: Server didn't accept TO address '{config.EMAIL_RECIPIENT}'.")
    except Exception as e:
        logging.error(f"Failed to send email summary: {e}", exc_info=True)

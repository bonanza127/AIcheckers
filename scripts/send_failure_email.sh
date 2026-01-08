#!/bin/bash
# Send email notification when aicheckers-backend fails
# Usage: send_failure_email.sh <service_name>

SERVICE_NAME="${1:-aicheckers-backend}"
TO_EMAIL="contact@aicheckers.net"
FROM_EMAIL="noreply@aicheckers.net"
SUBJECT="[ALERT] ${SERVICE_NAME} has failed"

# Get service status
STATUS=$(systemctl --user status "${SERVICE_NAME}" 2>&1 | head -20)

# Email body
BODY="The AIcheckers backend service has failed.

Service: ${SERVICE_NAME}
Time: $(date)
Hostname: $(hostname)

Status:
${STATUS}

Please check the server immediately.

---
This is an automated notification from AIcheckers monitoring system.
"

# Send email using mail command (requires mailutils)
if command -v mail &> /dev/null; then
    echo "${BODY}" | mail -s "${SUBJECT}" "${TO_EMAIL}"
    echo "[$(date)] Email sent to ${TO_EMAIL}"
else
    echo "[$(date)] WARNING: 'mail' command not found. Cannot send email."
    echo "[$(date)] Install mailutils: sudo apt-get install mailutils"
fi

# Integrations & Webhooks

GitHub integration:
- You can connect a GitHub repository to a DevBoard project.
- Supported actions: create issues from GitHub issues, sync labels.

Webhooks:
- Webhook events: issue.created, issue.updated, comment.created.
- Signature header: X-DevBoard-Signature (HMAC SHA-256).
- Retry policy: up to 5 retries with exponential backoff.

Rate limits:
- Webhook delivery is not rate-limited, but API calls are rate-limited (see Troubleshooting).

# Troubleshooting

Common error codes:
- DB-401: access token expired or invalid. Log in again.
- DB-403: account locked due to too many failed attempts.
- DB-409: conflict. Example: workspace slug already exists.
- DB-429: too many requests (rate limit exceeded).
- DB-500: server error. Try again later.

API rate limits:
- 60 requests per minute per token.
- If you hit DB-429, retry after 60 seconds.

Webhook debugging:
- Verify the X-DevBoard-Signature using your webhook secret.
- Confirm your endpoint returns HTTP 2xx within 3 seconds.

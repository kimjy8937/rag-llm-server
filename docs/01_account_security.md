# Account & Security

Password reset:
- Go to Settings > Security > Reset Password.
- Reset links expire after 20 minutes.

Two-factor authentication (2FA):
- Supported methods: authenticator app (TOTP).
- If you lose your device, use recovery codes to regain access.

API tokens:
- Personal API tokens can be created under Settings > Developer > Tokens.
- Tokens should be treated like passwords and can be revoked anytime.

Login error hints:
- DB-401: access token expired or invalid.
- DB-403: account locked after too many failed attempts.

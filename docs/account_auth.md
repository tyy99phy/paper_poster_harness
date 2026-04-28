# Local account login

The clean repo includes a minimal local login bootstrap. It does not depend on
any external workflow framework or separate account manager.

Run:

```bash
poster-harness login
```

or during configuration initialization:

```bash
poster-harness init-config --out poster_harness.yaml --login
```

The command opens the user's browser, waits for a localhost OAuth callback,
exchanges the code, and writes an account-auth JSON under:

```text
~/.config/poster-harness/auth
```

Default output filename:

```text
chatgpt-your_email@example.com.json
```

The JSON contains the fields needed by this harness:

```json
{
  "type": "chatgpt_account",
  "email": "your_email@example.com",
  "account_id": "...",
  "access_token": "...",
  "refresh_token": "...",
  "id_token": "..."
}
```

Tokens are not printed. The auth file is written with owner-only permissions
where the OS supports it.

If a user wants to place the file elsewhere:

```bash
poster-harness login --auth-file /absolute/path/to/account-auth.json
export POSTER_HARNESS_AUTH_FILE=/absolute/path/to/account-auth.json
```

Or in YAML:

```yaml
llm:
  backend: chatgpt_account
  account:
    auth_file: /absolute/path/to/account-auth.json

image_generation:
  backend: chatgpt_account
  account:
    auth_file: /absolute/path/to/account-auth.json
```

If the access token is close to expiry and a refresh token is present, the
harness refreshes the auth JSON in place before making requests.

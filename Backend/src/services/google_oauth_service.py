"""Google OAuth service — handles the Authorization Code flow with Google.

Uses google-auth-oauthlib's Flow class to generate the authorization URL,
exchange the authorization code for tokens, and verify/ decode the ID token.
"""

from typing import Optional

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow

from src.core.config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI

# Scopes requested during Google OAuth
_OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


def _create_flow(state: Optional[str] = None) -> Flow:
    """Create a google-auth-oauthlib Flow configured with our client settings.

    Args:
        state: Optional CSRF state string. If provided, the flow will use
               it instead of generating a random one (needed on the callback
               side to reuse the flow created during login).

    Returns:
        A configured Flow instance.
    """
    client_config = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [GOOGLE_REDIRECT_URI],
        }
    }
    flow = Flow.from_client_config(
        client_config,
        scopes=_OAUTH_SCOPES,
        state=state,
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    return flow


def get_authorization_url() -> tuple[str, str]:
    """Generate the Google OAuth authorization URL and CSRF state token.

    Returns:
        (authorization_url, state) tuple.
    """
    flow = _create_flow()
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="select_account",
    )
    return authorization_url, state


def exchange_code_for_tokens(code: str, state: str) -> dict:
    """Exchange an authorization code for Google tokens.

    Args:
        code: The authorization code from Google's callback.
        state: The CSRF state token that was returned by get_authorization_url().

    Returns:
        The decoded ID token info dict, containing at minimum: sub, email, name.

    Raises:
        google.auth.exceptions.GoogleAuthError: If token exchange or verification fails.
    """
    flow = _create_flow(state=state)
    flow.fetch_token(code=code)
    raw_id_token = flow.credentials.id_token
    decoded = id_token.verify_oauth2_token(
        raw_id_token, google_requests.Request(), audience=GOOGLE_CLIENT_ID
    )
    return decoded
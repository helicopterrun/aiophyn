"""Test for Kohler authentication error handling."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiophyn.partners.kohler import KOHLER_API
from aiophyn.errors import KohlerB2CError, KohlerTokenError


def create_mock_response(status=200, json_data=None, text_data=None, headers=None):
    """Create a mock aiohttp response."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.headers = headers or {}

    if json_data is not None:
        mock_response.json = AsyncMock(return_value=json_data)
    else:
        mock_response.json = AsyncMock(
            side_effect=json.JSONDecodeError("msg", "doc", 0)
        )

    if text_data is not None:
        mock_response.text = AsyncMock(return_value=text_data)
    else:
        mock_response.text = AsyncMock(return_value="")

    return mock_response


@pytest.mark.asyncio
async def test_b2c_login_http_error_on_initial_request():
    """Test that B2C login handles HTTP errors on initial request."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()

    # Mock failed initial request
    mock_response = create_mock_response(status=500, text_data="Internal Server Error")
    api._session.get = AsyncMock(return_value=mock_response)

    with pytest.raises(KohlerB2CError) as exc_info:
        await api.b2c_login()

    assert "HTTP 500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_b2c_login_missing_state_properties():
    """Test that B2C login handles missing StateProperties."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()

    # Mock response without StateProperties
    mock_response = create_mock_response(
        status=200, text_data="<html>No state properties here</html>"
    )
    api._session.get = AsyncMock(return_value=mock_response)

    with pytest.raises(KohlerB2CError) as exc_info:
        await api.b2c_login()

    assert "StateProperties" in str(exc_info.value)


@pytest.mark.asyncio
async def test_b2c_login_missing_csrf_token():
    """Test that B2C login handles missing CSRF token."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()

    # Mock response with StateProperties but no CSRF cookie
    mock_response = create_mock_response(
        status=200, text_data='"StateProperties=ABC123"'
    )
    api._session.get = AsyncMock(return_value=mock_response)

    # Mock empty cookie jar
    mock_cookie_jar = MagicMock()
    mock_cookie_jar.filter_cookies.return_value = {}
    api._session.cookie_jar = mock_cookie_jar

    with pytest.raises(KohlerB2CError) as exc_info:
        await api.b2c_login()

    assert "CSRF token" in str(exc_info.value)


@pytest.mark.asyncio
async def test_b2c_login_missing_location_header():
    """Test that B2C login handles missing Location header."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()

    # Setup mock responses
    responses = [
        create_mock_response(status=200, text_data='"StateProperties=ABC123"'),
        create_mock_response(status=200),  # POST response
        create_mock_response(status=302, headers={}),  # GET without Location header
    ]
    api._session.get = AsyncMock(side_effect=responses[::2])
    api._session.post = AsyncMock(return_value=responses[1])

    # Mock CSRF cookie
    mock_cookie = MagicMock()
    mock_cookie.value = "csrf_token_value"
    mock_cookie_jar = MagicMock()
    mock_cookie_jar.filter_cookies.return_value = {"x-ms-cpim-csrf": mock_cookie}
    api._session.cookie_jar = mock_cookie_jar

    with pytest.raises(KohlerB2CError) as exc_info:
        await api.b2c_login()

    assert "Location header" in str(exc_info.value)


@pytest.mark.asyncio
async def test_b2c_login_missing_auth_code():
    """Test that B2C login handles missing authorization code in Location."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()

    # Setup mock responses
    initial_response = create_mock_response(
        status=200, text_data='"StateProperties=ABC123"'
    )
    login_post_response = create_mock_response(status=200)
    redirect_response = create_mock_response(
        status=302,
        headers={
            "Location": "https://redirect.com?error=something"
        },  # No code parameter
    )

    # Track call count for proper mocking sequence
    get_call_count = [0]

    def mock_get(*args, **kwargs):
        get_call_count[0] += 1
        if get_call_count[0] == 1:
            return initial_response
        else:
            return redirect_response

    api._session.get = AsyncMock(side_effect=mock_get)
    api._session.post = AsyncMock(return_value=login_post_response)

    # Mock CSRF cookie
    mock_cookie = MagicMock()
    mock_cookie.value = "csrf_token_value"
    mock_cookie_jar = MagicMock()
    mock_cookie_jar.filter_cookies.return_value = {"x-ms-cpim-csrf": mock_cookie}
    api._session.cookie_jar = mock_cookie_jar

    with pytest.raises(KohlerB2CError) as exc_info:
        await api.b2c_login()

    # The error should mention authorization code
    assert "authorization code" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_b2c_login_invalid_json_in_token_response():
    """Test that B2C login handles invalid JSON in token response."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()

    # Setup mock responses
    responses = [
        create_mock_response(status=200, text_data='"StateProperties=ABC123"'),
        create_mock_response(status=200),  # Login POST
        create_mock_response(
            status=302, headers={"Location": "https://redirect.com?code=AUTH_CODE"}
        ),
        create_mock_response(status=200, text_data="Not valid JSON"),  # Token POST
    ]
    api._session.get = AsyncMock(side_effect=responses[::2])
    api._session.post = AsyncMock(side_effect=responses[1::2])

    # Mock CSRF cookie
    mock_cookie = MagicMock()
    mock_cookie.value = "csrf_token_value"
    mock_cookie_jar = MagicMock()
    mock_cookie_jar.filter_cookies.return_value = {"x-ms-cpim-csrf": mock_cookie}
    api._session.cookie_jar = mock_cookie_jar

    with pytest.raises(KohlerB2CError) as exc_info:
        await api.b2c_login()

    assert "Invalid JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_b2c_login_missing_client_info():
    """Test that B2C login handles missing client_info."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()
    api._session.close = AsyncMock()

    # Setup mock responses
    responses = [
        create_mock_response(status=200, text_data='"StateProperties=ABC123"'),
        create_mock_response(status=200),  # Login POST
        create_mock_response(
            status=302, headers={"Location": "https://redirect.com?code=AUTH_CODE"}
        ),
        create_mock_response(
            status=200,
            json_data={
                "access_token": "token123",
                "expires_in": 3600,
            },  # Missing client_info
        ),
    ]
    api._session.get = AsyncMock(side_effect=responses[::2])
    api._session.post = AsyncMock(side_effect=responses[1::2])

    # Mock CSRF cookie
    mock_cookie = MagicMock()
    mock_cookie.value = "csrf_token_value"
    mock_cookie_jar = MagicMock()
    mock_cookie_jar.filter_cookies.return_value = {"x-ms-cpim-csrf": mock_cookie}
    api._session.cookie_jar = mock_cookie_jar

    with pytest.raises(KohlerB2CError) as exc_info:
        await api.b2c_login()

    assert "client_info" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_phyn_token_http_error():
    """Test that get_phyn_token handles HTTP errors."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()
    api._user_id = "user123"
    api._token = "token123"

    # Mock failed request
    mock_response = create_mock_response(status=401, text_data="Unauthorized")
    api._session.get = AsyncMock(return_value=mock_response)

    with pytest.raises(KohlerTokenError) as exc_info:
        await api.get_phyn_token()

    assert "HTTP 401" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_phyn_token_error_msg_in_response():
    """Test that get_phyn_token handles error_msg in response."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()
    api._session.close = AsyncMock()
    api._user_id = "user123"
    api._token = "token123"

    # Mock response with error_msg
    mock_response = create_mock_response(
        status=200, json_data={"error_msg": "Invalid partner credentials"}
    )
    api._session.get = AsyncMock(return_value=mock_response)

    with pytest.raises(KohlerTokenError) as exc_info:
        await api.get_phyn_token()

    assert "Invalid partner credentials" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_phyn_token_missing_cognito():
    """Test that get_phyn_token handles missing cognito info."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()
    api._session.close = AsyncMock()
    api._user_id = "user123"
    api._token = "token123"

    # Mock response without cognito
    mock_response = create_mock_response(
        status=200, json_data={"pws_api": {"app_api_key": "key123"}}  # Missing cognito
    )
    api._session.get = AsyncMock(return_value=mock_response)

    with pytest.raises(KohlerTokenError) as exc_info:
        await api.get_phyn_token()

    assert "cognito" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_phyn_token_invalid_json():
    """Test that get_phyn_token handles invalid JSON."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()
    api._user_id = "user123"
    api._token = "token123"

    # Mock response with invalid JSON
    mock_response = create_mock_response(status=200, text_data="Not JSON")
    api._session.get = AsyncMock(return_value=mock_response)

    with pytest.raises(KohlerTokenError) as exc_info:
        await api.get_phyn_token()

    assert "Invalid JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_phyn_token_missing_token_field():
    """Test that get_phyn_token handles missing token field."""
    api = KOHLER_API("test@example.com", "password")
    api._session = MagicMock()
    api._session.close = AsyncMock()
    api._user_id = "user123"
    api._token = "token123"

    # Mock both responses but token response is missing the token field
    responses = [
        create_mock_response(
            status=200,
            json_data={
                "cognito": {"region": "us-east-1"},
                "pws_api": {"app_api_key": "key123"},
            },
        ),
        create_mock_response(
            status=200, json_data={"other_field": "value"}  # Missing token
        ),
    ]
    api._session.get = AsyncMock(side_effect=responses)

    with pytest.raises(KohlerTokenError) as exc_info:
        await api.get_phyn_token()

    assert "Token not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_token_to_password_invalid_token():
    """Test that token_to_password handles invalid token format."""
    api = KOHLER_API("test@example.com", "password")
    api._mobile_data = {
        "partner": {"comm_id": "dGVzdGRhdGF0ZXN0ZGF0YQ=="}  # Valid base64
    }

    # Invalid base64 token - use something that will fail base64 decode
    import base64

    with pytest.raises(KohlerTokenError) as exc_info:
        # Pass a token that has invalid characters
        await api.token_to_password("\x00\x01\x02")

    assert (
        "decode" in str(exc_info.value).lower()
        or "decrypt" in str(exc_info.value).lower()
    )


@pytest.mark.asyncio
async def test_token_to_password_missing_partner_data():
    """Test that token_to_password handles missing partner data."""
    api = KOHLER_API("test@example.com", "password")
    api._mobile_data = {}  # Missing partner

    # Use a valid base64 token so we get to the partner check
    with pytest.raises(KohlerTokenError) as exc_info:
        await api.token_to_password("dGVzdA")  # Valid base64

    assert (
        "partner" in str(exc_info.value).lower()
        or "comm_id" in str(exc_info.value).lower()
    )


@pytest.mark.asyncio
async def test_custom_exceptions_are_subclass_of_phyn_error():
    """Test that custom exceptions can be caught as PhynError."""
    from aiophyn.errors import PhynError

    try:
        raise KohlerB2CError("Test error")
    except PhynError as e:
        assert isinstance(e, KohlerB2CError)
        assert isinstance(e, PhynError)


@pytest.mark.asyncio
async def test_custom_exceptions_are_specific():
    """Test that custom exceptions can be caught specifically."""
    try:
        raise KohlerTokenError("Token error")
    except KohlerB2CError:
        pytest.fail("Should not catch KohlerB2CError")
    except KohlerTokenError as e:
        assert "Token error" in str(e)

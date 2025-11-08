"""Test for concurrent token refresh race condition fix."""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiophyn.api import API


def create_mock_session():
    """Create a properly mocked aiohttp ClientSession."""
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value={"success": True})
    mock_response.raise_for_status = MagicMock()

    mock_request_context = AsyncMock()
    mock_request_context.__aenter__.return_value = mock_response
    mock_request_context.__aexit__.return_value = None

    mock_session = MagicMock()
    mock_session.request.return_value = mock_request_context
    mock_session.close = AsyncMock()
    mock_session.closed = False

    return mock_session


@pytest.mark.asyncio
async def test_concurrent_token_refresh_no_race_condition():
    """Test that concurrent requests don't trigger duplicate authentication."""
    # Create API instance with mocked authentication
    with patch("aiophyn.api.boto3"), patch("aiophyn.api.AWSSRP"):
        api = API("test@example.com", "password", phyn_brand="phyn")

        # Set up initial token that's already expired
        api._token = "old_token"
        api._token_expiration = datetime.now() - timedelta(seconds=1)

        # Mock the authenticate method to track calls
        auth_call_count = 0

        async def mock_authenticate():
            nonlocal auth_call_count
            auth_call_count += 1
            # Simulate some delay in authentication
            await asyncio.sleep(0.1)
            # Set new token and expiration
            api._token = f"new_token_{auth_call_count}"
            api._token_expiration = datetime.now() + timedelta(hours=1)
            api._id_token = "id_token"
            api._refresh_token = "refresh_token"

        api.async_authenticate = mock_authenticate

        # Mock the actual HTTP request
        with patch("aiophyn.api.ClientSession") as mock_session_class:
            mock_session = create_mock_session()
            mock_session_class.return_value = mock_session

            # Simulate 5 concurrent requests that all detect expired token
            tasks = [
                api._request("GET", "https://api.example.com/test") for _ in range(5)
            ]

            # Run all requests concurrently
            results = await asyncio.gather(*tasks)

            # Verify results
            assert len(results) == 5
            assert all(result == {"success": True} for result in results)

            # CRITICAL: Only one authentication should have been called
            # despite 5 concurrent requests detecting the expired token
            assert auth_call_count == 1, f"Expected 1 auth call, got {auth_call_count}"


@pytest.mark.asyncio
async def test_lock_prevents_race_condition():
    """Test that the lock prevents multiple threads from authenticating simultaneously."""
    with patch("aiophyn.api.boto3"), patch("aiophyn.api.AWSSRP"):
        api = API("test@example.com", "password", phyn_brand="phyn")

        # Set expired token
        api._token = "old_token"
        api._token_expiration = datetime.now() - timedelta(seconds=1)

        # Track when authentication starts and ends
        auth_start_times = []
        auth_end_times = []

        async def mock_authenticate():
            auth_start_times.append(datetime.now())
            await asyncio.sleep(0.05)  # Simulate auth delay
            api._token = "new_token"
            api._token_expiration = datetime.now() + timedelta(hours=1)
            api._id_token = "id_token"
            api._refresh_token = "refresh_token"
            auth_end_times.append(datetime.now())

        api.async_authenticate = mock_authenticate

        # Mock HTTP request
        with patch("aiophyn.api.ClientSession") as mock_session_class:
            mock_session = create_mock_session()
            mock_session_class.return_value = mock_session

            # Launch multiple concurrent requests
            tasks = [
                api._request("GET", "https://api.example.com/test") for _ in range(3)
            ]
            await asyncio.gather(*tasks)

            # With proper locking, only one authentication should occur
            assert len(auth_start_times) == 1
            assert len(auth_end_times) == 1


@pytest.mark.asyncio
async def test_second_request_waits_for_first():
    """Test that second request waits for first authentication to complete."""
    with patch("aiophyn.api.boto3"), patch("aiophyn.api.AWSSRP"):
        api = API("test@example.com", "password", phyn_brand="phyn")

        # Set expired token
        api._token = "old_token"
        api._token_expiration = datetime.now() - timedelta(seconds=1)

        auth_completed = False

        async def mock_authenticate():
            nonlocal auth_completed
            await asyncio.sleep(0.1)
            api._token = "new_token"
            api._token_expiration = datetime.now() + timedelta(hours=1)
            api._id_token = "id_token"
            api._refresh_token = "refresh_token"
            auth_completed = True

        api.async_authenticate = mock_authenticate

        # Mock HTTP request
        with patch("aiophyn.api.ClientSession") as mock_session_class:
            mock_session = create_mock_session()
            mock_session_class.return_value = mock_session

            # First request
            task1 = asyncio.create_task(
                api._request("GET", "https://api.example.com/test1")
            )

            # Give first request a tiny head start
            await asyncio.sleep(0.01)

            # Second request should wait for first to complete auth
            task2 = asyncio.create_task(
                api._request("GET", "https://api.example.com/test2")
            )

            results = await asyncio.gather(task1, task2)

            # Both should succeed
            assert len(results) == 2
            assert all(result == {"success": True} for result in results)
            assert auth_completed

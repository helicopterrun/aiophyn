"""Test MQTT reconnection handling and state management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest
from aiophyn.mqtt import MQTTClient


@pytest.mark.asyncio
async def test_reconnect_evt_cleared_on_exception():
    """Test that reconnect_evt is properly cleared when exception occurs during reconnection."""
    # Create a mock API
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    # Create MQTT client
    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        # Track number of connection attempts
        call_count = [0]

        async def mock_get_mqtt_info():
            call_count[0] += 1
            raise Exception("Connection failed")

        client.get_mqtt_info = mock_get_mqtt_info

        # Mock sleep to avoid waiting - use a mock that doesn't call itself
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Run reconnection - will hit max attempts
            await client._do_reconnect(first=True)

        # Verify reconnect_evt is cleared even after exceptions
        assert (
            not client.reconnect_evt.is_set()
        ), "reconnect_evt should be cleared after exception"
        # Verify it attempted reconnection 20 times (max_attempts)
        assert (
            call_count[0] == 20
        ), f"Should have attempted to connect 20 times, got {call_count[0]}"


@pytest.mark.asyncio
async def test_max_reconnection_attempts():
    """Test that reconnection stops after max_attempts (20)."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        # Track number of connection attempts
        attempts = []

        async def mock_get_mqtt_info():
            attempts.append(1)
            raise Exception("Connection failed")

        client.get_mqtt_info = mock_get_mqtt_info

        # Mock sleep to avoid waiting
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Run reconnect - should stop after max attempts
            await client._do_reconnect(first=True)

        # Should have attempted exactly 20 times (max_attempts)
        assert len(attempts) == 20, f"Expected 20 attempts, got {len(attempts)}"

        # Verify reconnect_evt is cleared after max attempts
        assert (
            not client.reconnect_evt.is_set()
        ), "reconnect_evt should be cleared after max attempts"


@pytest.mark.asyncio
async def test_reconnect_timeout_handling():
    """Test that get_mqtt_info timeout is properly handled."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        # Mock get_mqtt_info to timeout
        async def mock_get_mqtt_info():
            await asyncio.sleep(10)  # Longer than the 5 second timeout
            return "host", "/path"

        client.get_mqtt_info = mock_get_mqtt_info

        # Attempt reconnection - should timeout and retry
        task = asyncio.create_task(client._do_reconnect(first=True))

        # Wait a bit for the timeout to occur
        await asyncio.sleep(6)

        # Cancel the task to prevent running all 20 attempts
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify reconnect_evt is cleared after cancellation
        assert (
            not client.reconnect_evt.is_set()
        ), "reconnect_evt should be cleared after cancellation"


@pytest.mark.asyncio
async def test_disconnect_reason_code_validation():
    """Test that _on_disconnect properly validates reason_code."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client") as mock_mqtt:
        client = MQTTClient(mock_api)
        client.client.is_connected = Mock(return_value=True)

        # Test with None reason_code
        client._on_disconnect(None, None, None, None)
        # Should not crash and should clear connect_evt
        assert not client.connect_evt.is_set()

        # Test with invalid reason_code
        client.connect_evt.set()
        client._on_disconnect(None, None, 9999, None)
        # Should not crash and should clear connect_evt
        assert not client.connect_evt.is_set()

        # Test with valid reason_code
        client.connect_evt.set()
        client._on_disconnect(None, None, 0, None)
        # Should not crash and should clear connect_evt
        assert not client.connect_evt.is_set()


@pytest.mark.asyncio
async def test_reconnect_success_exits_loop():
    """Test that successful reconnection exits the retry loop."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        # Track number of connection attempts
        attempts = []

        async def mock_get_mqtt_info():
            attempts.append(1)
            # Succeed on second attempt
            if len(attempts) == 2:
                return "host.example.com", "/mqtt"
            raise Exception("Connection failed")

        client.get_mqtt_info = mock_get_mqtt_info

        # Mock successful connection
        async def mock_executor(*args):
            pass

        client.event_loop.run_in_executor = mock_executor
        client.client.ws_set_options = Mock()

        # Mock connect_evt to simulate successful connection
        async def mock_wait_for(event_wait, timeout):
            # Simulate successful connection on second attempt
            if len(attempts) >= 2:
                client.connect_evt.set()
            return True

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            await client._do_reconnect(first=True)

        # Should have attempted exactly 2 times (first fails, second succeeds)
        assert len(attempts) == 2, f"Expected 2 attempts, got {len(attempts)}"

        # Verify reconnect_evt is cleared after success
        assert (
            not client.reconnect_evt.is_set()
        ), "reconnect_evt should be cleared after success"


@pytest.mark.asyncio
async def test_concurrent_reconnect_prevented():
    """Test that concurrent reconnection attempts are prevented."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        # Set reconnect_evt to simulate ongoing reconnection
        client.reconnect_evt.set()

        # Attempt another reconnection - should return immediately
        await client._do_reconnect(first=True)

        # Verify reconnect_evt is still set (wasn't cleared because we returned early)
        assert (
            client.reconnect_evt.is_set()
        ), "reconnect_evt should remain set for first caller"

        # Clean up
        client.reconnect_evt.clear()


@pytest.mark.asyncio
async def test_exponential_backoff():
    """Test that reconnection uses exponential backoff."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        sleep_times = []

        async def mock_sleep(duration):
            sleep_times.append(duration)

        async def mock_get_mqtt_info():
            raise Exception("Connection failed")

        client.get_mqtt_info = mock_get_mqtt_info

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await client._do_reconnect(first=True)

        # Check that backoff increases over time
        # First few attempts should use 2 second delay (but we skip first=True)
        # After 3 attempts, should use 10 second delay
        # After 6 attempts, should use 60 second delay
        assert (
            len(sleep_times) >= 10
        ), f"Should have multiple sleep calls, got {len(sleep_times)}"

        # Verify exponential backoff pattern
        # Attempts 1-3 should have t=2
        # Attempts 4-6 should have t=10
        # Attempts 7+ should have t=60
        for i, sleep_time in enumerate(sleep_times[:3]):
            assert (
                sleep_time == 2.0
            ), f"Attempt {i+2} should use 2s backoff, got {sleep_time}"

        if len(sleep_times) > 3:
            for i, sleep_time in enumerate(sleep_times[3:6], start=3):
                assert (
                    sleep_time == 10.0
                ), f"Attempt {i+2} should use 10s backoff, got {sleep_time}"

        if len(sleep_times) > 6:
            for i, sleep_time in enumerate(sleep_times[6:], start=6):
                assert (
                    sleep_time == 60.0
                ), f"Attempt {i+2} should use 60s backoff, got {sleep_time}"

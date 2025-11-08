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


@pytest.mark.asyncio
async def test_subscription_with_ack_success():
    """Test that _subscribe_with_ack successfully subscribes and waits for acknowledgment."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client") as mock_mqtt_class:
        mock_client_instance = MagicMock()
        mock_mqtt_class.return_value = mock_client_instance

        client = MQTTClient(mock_api)

        # Mock successful subscription
        mock_client_instance.subscribe.return_value = (
            0,
            123,
        )  # MQTT_ERR_SUCCESS, msg_id

        # Create a task to simulate the acknowledgment
        async def simulate_ack():
            await asyncio.sleep(0.1)
            # Simulate _on_subscribe being called
            client._on_subscribe(None, None, 123, [0], None)

        # Start the acknowledgment simulation
        ack_task = asyncio.create_task(simulate_ack())

        # Call _subscribe_with_ack
        res, msg_id = await client._subscribe_with_ack("test/topic", timeout=2)

        # Wait for the acknowledgment task to complete
        await ack_task

        # Verify success
        assert res == 0, "Should return MQTT_ERR_SUCCESS"
        assert msg_id == 123, "Should return the message ID"
        assert "test/topic" in client.topics, "Topic should be added to topics list"


@pytest.mark.asyncio
async def test_subscription_with_ack_timeout():
    """Test that _subscribe_with_ack raises RequestError on timeout."""
    from aiophyn.errors import RequestError

    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client") as mock_mqtt_class:
        mock_client_instance = MagicMock()
        mock_mqtt_class.return_value = mock_client_instance

        client = MQTTClient(mock_api)

        # Mock successful subscription initiation but no acknowledgment
        mock_client_instance.subscribe.return_value = (0, 456)

        # Try to subscribe with a short timeout - should timeout
        with pytest.raises(RequestError, match="Subscription ACK timeout"):
            await client._subscribe_with_ack("test/topic", timeout=0.5)


@pytest.mark.asyncio
async def test_subscription_with_ack_failure():
    """Test that _subscribe_with_ack handles subscription failures."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client") as mock_mqtt_class:
        mock_client_instance = MagicMock()
        mock_mqtt_class.return_value = mock_client_instance

        client = MQTTClient(mock_api)

        # Mock failed subscription (non-zero error code)
        mock_client_instance.subscribe.return_value = (1, 789)  # MQTT error code 1

        # Call _subscribe_with_ack - should return error code
        res, msg_id = await client._subscribe_with_ack("test/topic", timeout=2)

        assert res == 1, "Should return the error code"
        assert msg_id == 789, "Should return the message ID"


@pytest.mark.asyncio
async def test_reconnect_cleans_stale_pending_acks():
    """Test that reconnection cleans up stale pending_acks."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        # Add some stale pending_acks
        client.pending_acks[111] = "old/topic1"
        client.pending_acks[222] = "old/topic2"

        # Add a topic to re-subscribe to
        client.topics = ["test/topic"]

        async def mock_get_mqtt_info():
            return "host.example.com", "/mqtt"

        client.get_mqtt_info = mock_get_mqtt_info

        # Mock successful connection
        async def mock_executor(*args):
            pass

        client.event_loop.run_in_executor = mock_executor
        client.client.ws_set_options = Mock()
        client.client.subscribe = Mock(return_value=(0, 333))

        # Mock connect_evt to simulate successful connection
        async def mock_wait_for(event_wait, timeout):
            client.connect_evt.set()
            return True

        # Mock _subscribe_with_ack to simulate successful subscription
        async def mock_subscribe_with_ack(topic, timeout=5):
            # Simulate cleaning of stale acks in _do_reconnect
            return (0, 333)

        client._subscribe_with_ack = mock_subscribe_with_ack

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            await client._do_reconnect(first=True)

        # Verify stale acks were cleaned (they should have been cleared during reconnect)
        # The new subscription shouldn't add to stale acks since we mocked it
        assert 111 not in client.pending_acks, "Stale ack 111 should be cleaned"
        assert 222 not in client.pending_acks, "Stale ack 222 should be cleaned"


@pytest.mark.asyncio
async def test_reconnect_verifies_subscription_success():
    """Test that reconnection verifies all subscriptions succeeded."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        # Add multiple topics to re-subscribe to
        client.topics = ["topic1", "topic2", "topic3"]

        async def mock_get_mqtt_info():
            return "host.example.com", "/mqtt"

        client.get_mqtt_info = mock_get_mqtt_info

        # Mock successful connection
        async def mock_executor(*args):
            pass

        client.event_loop.run_in_executor = mock_executor
        client.client.ws_set_options = Mock()

        # Mock connect_evt to simulate successful connection
        async def mock_wait_for(event_wait, timeout):
            client.connect_evt.set()
            return True

        # Track which topics were subscribed
        subscribed_topics = []

        # Mock _subscribe_with_ack - topic2 fails
        async def mock_subscribe_with_ack(topic, timeout=5):
            subscribed_topics.append(topic)
            if topic == "topic2":
                from aiophyn.errors import RequestError

                raise RequestError(f"Subscription failed for {topic}")
            return (0, len(subscribed_topics))

        client._subscribe_with_ack = mock_subscribe_with_ack

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            await client._do_reconnect(first=True)

        # Verify all topics were attempted
        assert set(subscribed_topics) == {
            "topic1",
            "topic2",
            "topic3",
        }, "Should attempt to subscribe to all topics"


@pytest.mark.asyncio
async def test_reconnect_handles_partial_subscription_failure():
    """Test that reconnection continues even with partial subscription failures."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        # Add multiple topics
        client.topics = ["topic1", "topic2", "topic3", "topic4"]

        async def mock_get_mqtt_info():
            return "host.example.com", "/mqtt"

        client.get_mqtt_info = mock_get_mqtt_info

        # Mock successful connection
        async def mock_executor(*args):
            pass

        client.event_loop.run_in_executor = mock_executor
        client.client.ws_set_options = Mock()

        # Mock connect_evt
        async def mock_wait_for(event_wait, timeout):
            client.connect_evt.set()
            return True

        # Mock _subscribe_with_ack - some fail, some succeed
        async def mock_subscribe_with_ack(topic, timeout=5):
            if topic in ["topic2", "topic4"]:
                # Return error code for these topics
                return (1, 999)  # Error code 1
            return (0, 123)  # Success for others

        client._subscribe_with_ack = mock_subscribe_with_ack

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            # Should complete successfully despite partial failures
            await client._do_reconnect(first=True)

        # Verify reconnect completed (reconnect_evt cleared)
        assert (
            not client.reconnect_evt.is_set()
        ), "Reconnect should complete even with partial subscription failures"


@pytest.mark.asyncio
async def test_on_subscribe_handles_tuple_format():
    """Test that _on_subscribe correctly handles tuple format with event."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        # Create an event for acknowledgment
        ack_evt = asyncio.Event()

        # Add pending ack with tuple format (topic, event)
        client.pending_acks[123] = ("test/topic", ack_evt)

        # Call _on_subscribe
        client._on_subscribe(None, None, 123, [0], None)

        # Verify event was set
        assert ack_evt.is_set(), "Event should be set"

        # Verify topic was added
        assert "test/topic" in client.topics, "Topic should be added"

        # Verify pending_ack was removed
        assert 123 not in client.pending_acks, "Pending ack should be removed"


@pytest.mark.asyncio
async def test_on_subscribe_handles_string_format():
    """Test that _on_subscribe still handles old string format for backward compatibility."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        # Add pending ack with old string format
        client.pending_acks[456] = "another/topic"

        # Call _on_subscribe
        client._on_subscribe(None, None, 456, [0], None)

        # Verify topic was added
        assert "another/topic" in client.topics, "Topic should be added"

        # Verify pending_ack was removed
        assert 456 not in client.pending_acks, "Pending ack should be removed"


@pytest.mark.asyncio
async def test_on_subscribe_avoids_duplicate_topics():
    """Test that _on_subscribe doesn't add duplicate topics."""
    mock_api = MagicMock()
    mock_api.username = "test@example.com"

    with patch("aiophyn.mqtt.paho_mqtt.Client"):
        client = MQTTClient(mock_api)

        # Pre-add the topic
        client.topics = ["duplicate/topic"]

        # Add pending ack for same topic
        client.pending_acks[789] = "duplicate/topic"

        # Call _on_subscribe
        client._on_subscribe(None, None, 789, [0], None)

        # Verify topic appears only once
        assert (
            client.topics.count("duplicate/topic") == 1
        ), "Topic should not be duplicated"

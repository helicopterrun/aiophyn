"""Test for device operation timeouts."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiophyn.device import Device
from aiophyn.errors import RequestError


@pytest.mark.asyncio
async def test_get_state_timeout():
    """Test that get_state times out correctly."""
    # Create a mock request that takes longer than the timeout
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {"state": "online"}

    device = Device(slow_request)
    
    # Should timeout after 1 second
    with pytest.raises(RequestError) as exc_info:
        await device.get_state("test-device", timeout=1)
    
    assert "Timeout getting state for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_state_succeeds_within_timeout():
    """Test that get_state succeeds when response is within timeout."""
    # Create a mock request that completes quickly
    async def fast_request(*args, **kwargs):
        await asyncio.sleep(0.1)
        return {"state": "online"}

    device = Device(fast_request)
    
    # Should succeed with 2 second timeout
    result = await device.get_state("test-device", timeout=2)
    assert result == {"state": "online"}


@pytest.mark.asyncio
async def test_get_consumption_timeout():
    """Test that get_consumption times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {"consumption": 100}

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.get_consumption("test-device", "2023/01/01", timeout=1)
    
    assert "Timeout getting consumption for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_water_statistics_timeout():
    """Test that get_water_statistics times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return [{"stat": "data"}]

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.get_water_statistics("test-device", 1000, 2000, timeout=1)
    
    assert "Timeout getting water statistics for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_open_valve_timeout():
    """Test that open_valve times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {}

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.open_valve("test-device", timeout=1)
    
    assert "Timeout opening valve for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_close_valve_timeout():
    """Test that close_valve times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {}

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.close_valve("test-device", timeout=1)
    
    assert "Timeout closing valve for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_away_mode_timeout():
    """Test that get_away_mode times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {"away_mode": True}

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.get_away_mode("test-device", timeout=1)
    
    assert "Timeout getting away mode for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_enable_away_mode_timeout():
    """Test that enable_away_mode times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {}

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.enable_away_mode("test-device", timeout=1)
    
    assert "Timeout enabling away mode for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_disable_away_mode_timeout():
    """Test that disable_away_mode times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {}

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.disable_away_mode("test-device", timeout=1)
    
    assert "Timeout disabling away mode for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_autoshutoff_status_timeout():
    """Test that get_autoshuftoff_status times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {"status": "enabled"}

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.get_autoshuftoff_status("test-device", timeout=1)
    
    assert "Timeout getting autoshutoff status for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_device_preferences_timeout():
    """Test that get_device_preferences times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return [{"preference": "value"}]

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.get_device_preferences("test-device", timeout=1)
    
    assert "Timeout getting device preferences for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_health_tests_timeout():
    """Test that get_health_tests times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return [{"test": "result"}]

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.get_health_tests("test-device", timeout=1)
    
    assert "Timeout getting health tests for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_latest_firmware_info_timeout():
    """Test that get_latest_firmware_info times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {"version": "1.0.0"}

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.get_latest_firmware_info("test-device", timeout=1)
    
    assert "Timeout getting latest firmware info for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_leak_test_timeout():
    """Test that run_leak_test times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {}

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.run_leak_test("test-device", timeout=1)
    
    assert "Timeout running leak test for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_set_autoshutoff_enabled_timeout():
    """Test that set_autoshutoff_enabled times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {}

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.set_autoshutoff_enabled("test-device", True, timeout=1)
    
    assert "Timeout setting autoshutoff for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_set_device_preferences_timeout():
    """Test that set_device_preferences times out correctly."""
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(2)
        return {}

    device = Device(slow_request)
    
    with pytest.raises(RequestError) as exc_info:
        await device.set_device_preferences("test-device", [{"name": "test", "value": "value"}], timeout=1)
    
    assert "Timeout setting device preferences for device test-device after 1s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_default_timeout_is_10_seconds():
    """Test that the default timeout is 10 seconds."""
    call_count = 0
    
    async def request_that_tracks_timeout(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Simulate a request that takes 11 seconds
        await asyncio.sleep(11)
        return {}
    
    device = Device(request_that_tracks_timeout)
    
    # Should timeout with default 10 second timeout
    with pytest.raises(RequestError):
        await device.get_state("test-device")
    
    assert call_count == 1


@pytest.mark.asyncio
async def test_concurrent_operations_with_one_slow():
    """Test that one slow operation doesn't block other operations."""
    fast_call_completed = False
    slow_call_started = False
    
    async def fast_request(*args, **kwargs):
        nonlocal fast_call_completed
        await asyncio.sleep(0.1)
        fast_call_completed = True
        return {"state": "online"}
    
    async def slow_request(*args, **kwargs):
        nonlocal slow_call_started
        slow_call_started = True
        await asyncio.sleep(5)
        return {"state": "online"}
    
    fast_device = Device(fast_request)
    slow_device = Device(slow_request)
    
    # Start both operations concurrently
    slow_task = asyncio.create_task(slow_device.get_state("slow-device", timeout=1))
    fast_task = asyncio.create_task(fast_device.get_state("fast-device", timeout=2))
    
    # Wait for fast task to complete
    result = await fast_task
    assert result == {"state": "online"}
    assert fast_call_completed
    assert slow_call_started
    
    # The slow task should timeout
    with pytest.raises(RequestError):
        await slow_task


@pytest.mark.asyncio
async def test_custom_timeout_values():
    """Test that custom timeout values work correctly."""
    async def request_with_delay(delay):
        async def _request(*args, **kwargs):
            await asyncio.sleep(delay)
            return {"success": True}
        return _request
    
    # Test with 2 second timeout - should succeed with 1 second delay
    device = Device(await request_with_delay(1))
    result = await device.get_state("test-device", timeout=2)
    assert result == {"success": True}
    
    # Test with 1 second timeout - should fail with 2 second delay
    device = Device(await request_with_delay(2))
    with pytest.raises(RequestError):
        await device.get_state("test-device", timeout=1)

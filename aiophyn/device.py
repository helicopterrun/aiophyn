"""Define /devices endpoints."""
import asyncio
from typing import Awaitable, Any, Callable, Optional

from .const import API_BASE
from .errors import RequestError


class Device:
    """Define an object to handle the endpoints."""

    def __init__(self, request: Callable[..., Awaitable]) -> None:
        """Initialize."""
        self._request: Callable[..., Awaitable] = request

    async def get_state(self, device_id: str, timeout: int = 10) -> dict:
        """Return state of a device.

        :param device_id: Unique identifier for the device
        :type device_id: ``str``
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: ``int``
        :rtype: ``dict``
        :raises RequestError: If request fails or times out
        """
        try:
            async with asyncio.timeout(timeout):
                return await self._request("get", f"{API_BASE}/devices/{device_id}/state")
        except TimeoutError:
            raise RequestError(f"Timeout getting state for device {device_id} after {timeout}s")

    async def get_consumption(
        self,
        device_id: str,
        duration: str,
        precision: int = 6,
        details: Optional[str] = False,
        event_count: Optional[str] = False,
        comparison: Optional[str] = False,
        timeout: int = 10,
    ) -> dict:
        """Return water consumption of a device.

        :param device_id: Unique identifier for the device
        :type device_id: ``str``
        :param duration: Date string formatted as 'YYYY/MM/DD', 'YYYY/MM', or 'YYYY'
        :type duration: ``str``
        :param precision: Decimal places of measurement precision
        :type precision: ``int``
        :param details: Include detailed breakdown of consumption
        :type details: ``bool``
        :param event_count: Include the event count
        :type event_count: ``bool``
        :param comparison: Include comparison data
        :type comparison: ``bool``
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: ``int``
        :rtype: ``dict``
        :raises RequestError: If request fails or times out
        """

        params = {
            "device_id": device_id,
            "duration": duration,
            "precision": precision,
        }

        if details:
            params["details"] = "Y"

        if event_count:
            params["event_count"] = "Y"

        if comparison:
            params["comparison"] = "Y"

        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "get", f"{API_BASE}/devices/{device_id}/consumption/details", params=params
                )
        except TimeoutError:
            raise RequestError(f"Timeout getting consumption for device {device_id} after {timeout}s")

    async def get_water_statistics(self, device_id: str, from_ts, to_ts, timeout: int = 10):
        """Get statistics about a PW1 sensor

        :param device_id: Unique identifier for the device
        :type device_id: str
        :param from_ts: Lower bound timestamp. This is a timestamp with thousands as integer
        :type from_ts: int
        :param to_ts: Upper bound timestamp. This is a timestamp with thousands as integer
        :type to_ts: int
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: int
        :return: List of dictionaries of results. 
        :rtype: List[dict[str, Any]]
        :raises RequestError: If request fails or times out
        """
        params = {
            "from_ts": from_ts,
            "to_ts": to_ts
        }

        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "get", f"{API_BASE}/devices/{device_id}/water_statistics/history/", params=params
                )
        except TimeoutError:
            raise RequestError(f"Timeout getting water statistics for device {device_id} after {timeout}s")

    async def open_valve(self, device_id: str, timeout: int = 10) -> None:
        """Open a device shutoff valve.

        :param device_id: Unique identifier for the device
        :type device_id: ``str``
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: ``int``
        :rtype: ``dict``
        :raises RequestError: If request fails or times out
        """
        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "post",
                    f"{API_BASE}/devices/{device_id}/sov/Open",
                )
        except TimeoutError:
            raise RequestError(f"Timeout opening valve for device {device_id} after {timeout}s")

    async def close_valve(self, device_id: str, timeout: int = 10) -> None:
        """Close a device shutoff valve.

        :param device_id: Unique identifier for the device
        :type device_id: ``str``
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: ``int``
        :rtype: ``dict``
        :raises RequestError: If request fails or times out
        """
        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "post",
                    f"{API_BASE}/devices/{device_id}/sov/Close",
                )
        except TimeoutError:
            raise RequestError(f"Timeout closing valve for device {device_id} after {timeout}s")

    async def get_away_mode(self, device_id: str, timeout: int = 10) -> dict:
        """Return away mode status of a device.

        :param device_id: Unique identifier for the device
        :type device_id: ``str``
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: ``int``
        :rtype: ``dict``
        :raises RequestError: If request fails or times out
        """
        try:
            async with asyncio.timeout(timeout):
                return await self._request("get", f"{API_BASE}/preferences/device/{device_id}/leak_sensitivity_away_mode")
        except TimeoutError:
            raise RequestError(f"Timeout getting away mode for device {device_id} after {timeout}s")


    async def enable_away_mode(self, device_id: str, timeout: int = 10) -> None:
        """Enable the device's away mode.

        :param device_id: Unique identifier for the device
        :type device_id: ``str``
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: ``int``
        :rtype: ``dict``
        :raises RequestError: If request fails or times out
        """
        data = [
            {
                "name": "leak_sensitivity_away_mode",
                "value": "true",
                "device_id": device_id,
            }
        ]
        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "post", f"{API_BASE}/preferences/device/{device_id}", json=data
                )
        except TimeoutError:
            raise RequestError(f"Timeout enabling away mode for device {device_id} after {timeout}s")

    async def disable_away_mode(self, device_id: str, timeout: int = 10) -> None:
        """Disable the device's away mode.

        :param device_id: Unique identifier for the device
        :type device_id: ``str``
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: ``int``
        :rtype: ``dict``
        :raises RequestError: If request fails or times out
        """
        data = [
            {
                "name": "leak_sensitivity_away_mode",
                "value": "false",
                "device_id": device_id,
            }
        ]
        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "post", f"{API_BASE}/preferences/device/{device_id}", json=data
                )
        except TimeoutError:
            raise RequestError(f"Timeout disabling away mode for device {device_id} after {timeout}s")
    
    async def get_autoshuftoff_status(self, device_id: str, timeout: int = 10) -> dict:
        """Get phyn device preferences.

        :param device_id: Unique identifier for the device
        :type device_id: str
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: int
        :return: List of dicts with the following keys: created_ts, device_id, name, updated_ts, value
        :rtype: dict
        :raises RequestError: If request fails or times out
        """
        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "get", f"{API_BASE}/devices/{device_id}/auto_shutoff"
                )
        except TimeoutError:
            raise RequestError(f"Timeout getting autoshutoff status for device {device_id} after {timeout}s")
    

    async def get_device_preferences(self, device_id: str, timeout: int = 10) -> dict:
        """Get phyn device preferences.

        :param device_id: Unique identifier for the device
        :type device_id: str
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: int
        :return: List of dicts with the following keys: created_ts, device_id, name, updated_ts, value
        :rtype: dict
        :raises RequestError: If request fails or times out
        """
        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "get", f"{API_BASE}/preferences/device/{device_id}"
                )
        except TimeoutError:
            raise RequestError(f"Timeout getting device preferences for device {device_id} after {timeout}s")
    
    async def get_health_tests(self, device_id: str, timeout: int = 10) -> dict:
        """Get phyn device preferences.

        :param device_id: Unique identifier for the device
        :type device_id: str
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: int
        :return: List of dicts with the following keys
        :rtype: dict
        :raises RequestError: If request fails or times out
        """
        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "get", f"{API_BASE}/devices/{device_id}/health_tests?list_type=grouped"
                )
        except TimeoutError:
            raise RequestError(f"Timeout getting health tests for device {device_id} after {timeout}s")
    
    async def get_latest_firmware_info(self, device_id: str, timeout: int = 10) -> dict:
        """Get Latest Firmware Information

        :param device_id: Unique identifier for the device
        :type device_id: str
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: int
        :return: Returns dict with fw_img_name, fw_version, product_code
        :rtype: dict
        :raises RequestError: If request fails or times out
        """
        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "get", f"{API_BASE}/firmware/latestVersion/v2?device_id={device_id}"
                )
        except TimeoutError:
            raise RequestError(f"Timeout getting latest firmware info for device {device_id} after {timeout}s")

    async def run_leak_test(self, device_id: str, extended_test: bool = False, timeout: int = 10):
        """Run a leak test

        :param device_id: Unique identifier for the device
        :type device_id: str
        :param extended_test: True if the test be extended, defaults to False
        :type extended_test: bool, optional
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: int
        :raises RequestError: If request fails or times out
        """
        data = {
            "initiator": "App",
            "test_duration": "e" if extended_test is True else "s"
        }
        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "post", f"{API_BASE}/devices/{device_id}/health_tests", json=data
                )
        except TimeoutError:
            raise RequestError(f"Timeout running leak test for device {device_id} after {timeout}s")

    async def set_autoshutoff_enabled(self, device_id: str, shutoff_on: bool, time: int | None = None, timeout: int = 10) -> None:
        """Set autoshutoff enabled

        :param device_id: Unique identifier for the device
        :type device_id: str
        :param shutoff_on: Turn autoshutoff on (True) or off (False). If false, also turn off for amount of time
        :type shutoff_on: bool
        :param time: Time for shutoff in seconds if disabling (30, 3600, 21600, 86400), or blank for indefinite
        :type time: int | None
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: int
        :param data: List of dicts which have the keys: device_id, name, value
        :type data: List[dict]
        :raises RequestError: If request fails or times out
        """
        url = f"{API_BASE}/devices/{device_id}/auto_shutoff/status/"
        if shutoff_on == True:
            url += "Enable"
        else:
            url += "Disable"
            if time != None:
                url += "/%s" % time
        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "post", url
                )
        except TimeoutError:
            raise RequestError(f"Timeout setting autoshutoff for device {device_id} after {timeout}s")

    async def set_device_preferences(self, device_id: str, data: list[dict], timeout: int = 10) -> None:
        """Set device preferences

        :param device_id: Unique identifier for the device
        :type device_id: str
        :param data: List of dicts which have the keys: device_id, name, value
        :type data: List[dict]
        :param timeout: Request timeout in seconds, defaults to 10
        :type timeout: int
        :raises RequestError: If request fails or times out
        """
        try:
            async with asyncio.timeout(timeout):
                return await self._request(
                    "post", f"{API_BASE}/preferences/device/{device_id}", json=data
                )
        except TimeoutError:
            raise RequestError(f"Timeout setting device preferences for device {device_id} after {timeout}s")

"""Module providing a MQTT provider"""

import asyncio
import logging

from typing import Any, Dict, Union, Optional

import inspect
import json
import time
import urllib
import ssl
import socket
import re
import socks
import paho.mqtt.client as paho_mqtt

from .const import API_BASE
from .errors import RequestError

_LOGGER = logging.getLogger(__name__)


class AIOHelper:
    """Helper class for Asynchronous IO"""

    def __init__(self, client: paho_mqtt.Client) -> None:
        self.loop = asyncio.get_running_loop()
        self.client = client
        self.client.on_socket_open = self._on_socket_open
        self.client.on_socket_close = self._on_socket_close
        self.client._on_socket_register_write = self._on_socket_register_write
        self.client._on_socket_unregister_write = self._on_socket_unregister_write
        self.misc_task: Optional[asyncio.Task] = None

    def _on_socket_open(
        self, client: paho_mqtt.Client, userdata: Any, sock: socket.socket
    ) -> None:
        # pylint: disable=unused-argument
        _LOGGER.info("MQTT Socket Opened")
        self.loop.add_reader(sock, client.loop_read)
        self.misc_task = self.loop.create_task(self.misc_loop())

    def _on_socket_close(
        self, client: paho_mqtt.Client, userdata: Any, sock: socket.socket
    ) -> None:
        # pylint: disable=unused-argument
        _LOGGER.info("MQTT Socket Closed")
        self.loop.remove_reader(sock)
        if self.misc_task is not None:
            self.misc_task.cancel()

    def _on_socket_register_write(
        self, client: paho_mqtt.Client, userdata: Any, sock: socket.socket
    ) -> None:
        # pylint: disable=unused-argument
        self.loop.add_writer(sock, client.loop_write)

    def _on_socket_unregister_write(
        self, client: paho_mqtt.Client, userdata: Any, sock: socket.socket
    ) -> None:
        # pylint: disable=unused-argument
        self.loop.remove_writer(sock)

    async def misc_loop(self) -> None:
        """Loop for MQTT"""
        while self.client.loop_misc() == paho_mqtt.MQTT_ERR_SUCCESS:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
        _LOGGER.info("MQTT Misc Loop Complete")


class Timer:
    """Class to run a job with a timeout"""

    def __init__(self, callback):
        _LOGGER.info("Creating timer")
        self._timeout = 0
        self._callback = callback
        self._task = None

    async def _job(self, timeout):
        """Run the job with a timeout"""
        await asyncio.sleep(timeout)
        _LOGGER.debug("Executing timer callback")
        if inspect.iscoroutinefunction(self._callback):
            await self._callback()
        else:
            self._callback()

    def cancel(self):
        """Cancel a timer task"""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def start(self, timeout):
        """Start a timer task"""
        if self._task is not None:
            self._task.cancel()
        _LOGGER.debug("Starting timer job for %s seconds", timeout)
        self._task = asyncio.create_task(self._job(timeout))


class MQTTClient:
    """AIO MQTT client"""

    def __init__(
        self,
        api,
        client_id: str = None,
        verify_ssl: bool = True,
        proxy: str = None,
        proxy_port: int = None,
    ):
        self.event_loop = asyncio.get_running_loop()
        self.api = api
        self.pending_acks = {}
        self.topics = []
        self.connect_evt: asyncio.Event = asyncio.Event()
        self.connect_task = None
        self.disconnect_evt: Optional[asyncio.Event] = None
        self.reconnect_evt: asyncio.Event = asyncio.Event()
        self.host = None
        self.port = 443

        if client_id is None:
            client_id = "aiophyn-%s" % int(time.time())

        self.client = paho_mqtt.Client(client_id=client_id, transport="websockets")
        self.helper: AIOHelper = None
        self.reconnect_timer = Timer(self._process_reconnect)

        self.verify_ssl: bool = verify_ssl
        self.proxy: Optional[str] = proxy
        self.proxy_port: Optional[int] = proxy_port

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_subscribe = self._on_subscribe
        self.client.on_message = self._on_message

        self._handlers = {"connect": [], "disconnect": [], "update": []}

    async def add_event_handler(self, type, target):
        """Add an event handler for MQTT events"""
        if type not in self._handlers.keys():
            return False

        if target in self._handlers[type]:
            return True
        self._handlers[type].append(target)

    async def connect(self):
        """Create a conenction to the MQTT server"""
        self.host, path = await self.get_mqtt_info()
        self.client.ws_set_options(path, headers={"Host": self.host})

        if self.verify_ssl:
            context = ssl.SSLContext()
            self.client.tls_set_context(context)
        else:
            context = ssl.SSLContext()
            context.verify_mode = ssl.CERT_NONE
            context.check_hostname = False
            self.client.tls_set_context(context)
            self.client.tls_insecure_set(True)

        if self.proxy is not None and self.proxy_port is not None:
            self.client.proxy_set(
                proxy_type=socks.HTTP, proxy_addr=self.proxy, proxy_port=self.proxy_port
            )

        self.helper = AIOHelper(self.client)
        _LOGGER.info("Connecting to mqtt websocket: %s", self.host)
        self.reconnect_timer.start(5)
        await self.event_loop.run_in_executor(
            None,
            self.client.connect,
            self.host,
            self.port,
        )

    def disconnect(self):
        """Disconnect from server"""
        self.disconnect_evt = asyncio.Event()
        _LOGGER.info("MQTT client disconnecting...")
        self.client.disconnect()

    async def disconnect_and_wait(self):
        """Disconnect from server and wait"""
        self.disconnect()
        await self.disconnect_evt.wait()

    async def get_mqtt_info(self):
        """Gets WebSocket URL and parameters for a MQTT connection
        Returns a list of url and path
        """
        user_id = urllib.parse.quote_plus(self.api.username)
        try:
            wss_data = await self.api._request(
                "post", f"{API_BASE}/users/{user_id}/iot_policy", token_type="id"
            )
        except:
            Exception("Could not get WebSocket/MQTT url from API")

        match = re.match(r"wss:\/\/([a-zA-Z0-9\.\-]+)(\/mqtt?.*)", wss_data["wss_url"])
        if not match:
            raise Exception("Could not find WebSocket/MQTT url")

        return match.group(1), match.group(2)

    async def subscribe(self, topic):
        """Subscribe to a MQTT topic"""
        _LOGGER.info("Attempting to subscribe to: %s", topic)
        res, msg_id = self.client.subscribe(topic, 0)
        self.pending_acks[msg_id] = topic

    async def _subscribe_with_ack(self, topic: str, timeout: int = 5) -> tuple:
        """Subscribe and wait for acknowledgment

        Args:
            topic: The MQTT topic to subscribe to
            timeout: Maximum time to wait for subscription acknowledgment

        Returns:
            Tuple of (result_code, message_id)

        Raises:
            RequestError: If subscription acknowledgment times out
        """
        # Use a unique event per subscription attempt
        ack_evt = asyncio.Event()

        try:
            async with asyncio.timeout(timeout):
                res, msg_id = self.client.subscribe(topic, 0)
                if res != paho_mqtt.MQTT_ERR_SUCCESS:
                    _LOGGER.error(
                        "Failed to initiate subscription to %s: %s", topic, res
                    )
                    return res, msg_id

                # Store event for this specific message ID
                self.pending_acks[msg_id] = (topic, ack_evt)

                # Wait for SUBACK
                await ack_evt.wait()
                return res, msg_id
        except asyncio.TimeoutError:
            raise RequestError(f"Subscription ACK timeout for {topic}")

    def _on_connect(
        self,
        client: paho_mqtt.Client,
        user_data: Any,
        flags: Dict[str, Any],
        reason_code: Union[int, paho_mqtt.ReasonCodes],
        properties: Optional[paho_mqtt.Properties] = None,
    ) -> None:
        # pylint: disable=unused-argument
        _LOGGER.info("MQTT Client Connected")
        if reason_code == 0:
            _LOGGER.info("Trying to run timer...")
            self.reconnect_timer.cancel()
            self.reconnect_timer.start(3600)
            self.connect_evt.set()
        else:
            if isinstance(reason_code, int):
                err_str = paho_mqtt.connack_string(reason_code)
            else:
                err_str = reason_code.getName()
            _LOGGER.info("MQTT Connection Failed: %s", err_str)

    def _on_disconnect(
        self,
        client: paho_mqtt.Client,
        user_data: Any,
        reason_code: int,
        properties: Optional[paho_mqtt.Properties] = None,
    ) -> None:
        # pylint: disable=unused-argument
        # Validate reason_code to handle None or invalid values
        if reason_code is None:
            reason_str = "Unknown (None)"
        else:
            try:
                reason_str = paho_mqtt.error_string(reason_code)
            except Exception:
                reason_str = f"Unknown ({reason_code})"

        if self.disconnect_evt is not None:
            self.disconnect_evt.set()
            _LOGGER.info("Client disconnected, not attempting to reconnect")
        elif self.is_connected():
            # The server connection was dropped, attempt to reconnect
            _LOGGER.info("MQTT Server Disconnected, reason: %s", reason_str)
            self.reconnect_timer.cancel()
            if self.connect_task is None:
                self.connect_task = asyncio.create_task(self._do_reconnect(True))
        self.connect_evt.clear()

    def is_connected(self) -> bool:
        """Checks if the client is connected"""
        return self.client.is_connected()

    async def _process_reconnect(self):
        # self.connect_task = True
        _LOGGER.info("Processing reconnect request")

        self.disconnect_evt = asyncio.Event()
        if self.is_connected():
            self.client.disconnect()
            await self.disconnect_evt.wait()

        self.connect_task = asyncio.create_task(self._do_reconnect(True))

    async def _do_reconnect(self, first: bool = False) -> None:
        if self.reconnect_evt.is_set():
            _LOGGER.info("Already attempting to reconnect, second attemp cancelled.")
            return

        _LOGGER.info("Attempting MQTT Connect/Reconnect")
        self.reconnect_evt.set()
        max_attempts = 20
        last_err: Exception = Exception()
        connect_attempts = 0
        t: float = 2.0

        try:
            while connect_attempts < max_attempts:
                if not first:
                    try:
                        if connect_attempts > 6:
                            t = 60.0
                        elif connect_attempts > 3:
                            t = 10.0
                        _LOGGER.debug("MQTT throttle for %s seconds", t)
                        await asyncio.sleep(t)
                    except asyncio.CancelledError:
                        raise
                first = False
                connect_attempts += 1
                try:
                    # Use timeout to prevent hanging on get_mqtt_info
                    async with asyncio.timeout(5):
                        self.host, path = await self.get_mqtt_info()
                    self.client.ws_set_options(path, headers={"Host": self.host})
                    _LOGGER.info("Attempting to reconnnect...")
                    await self.event_loop.run_in_executor(
                        None,
                        self.client.connect,
                        self.host,
                        self.port,
                    )

                    await asyncio.wait_for(self.connect_evt.wait(), timeout=2.0)
                    if not self.connect_evt.is_set():
                        _LOGGER.info("Timeout while waiting for MQTT connection")
                        continue

                    # Clean up stale pending_acks from before disconnect
                    stale_acks = [mid for mid in self.pending_acks.keys()]
                    if stale_acks:
                        _LOGGER.debug(
                            "Cleaning up %d stale pending_acks", len(stale_acks)
                        )
                        for mid in stale_acks:
                            del self.pending_acks[mid]

                    # Re-subscribe to all topics with verification
                    topics_to_subscribe = list(set(self.topics))
                    successful_subs = set()

                    for topic in topics_to_subscribe:
                        try:
                            async with asyncio.timeout(5):
                                res, msg_id = await self._subscribe_with_ack(topic)
                                if res == paho_mqtt.MQTT_ERR_SUCCESS:
                                    successful_subs.add(topic)
                                else:
                                    _LOGGER.error(
                                        "Failed to re-subscribe to %s: %s", topic, res
                                    )
                        except RequestError as e:
                            _LOGGER.error("Error re-subscribing to %s: %s", topic, e)
                        except Exception as e:
                            _LOGGER.error(
                                "Unexpected error re-subscribing to %s: %s", topic, e
                            )

                    if successful_subs != set(topics_to_subscribe):
                        failed = set(topics_to_subscribe) - successful_subs
                        _LOGGER.warning(
                            "Failed to re-subscribe to %d topics: %s",
                            len(failed),
                            failed,
                        )
                        # Continue anyway - partial success is better than failing the whole reconnect
                    else:
                        _LOGGER.info(
                            "Successfully re-subscribed to all %d topics",
                            len(topics_to_subscribe),
                        )

                    break  # Success - exit the retry loop
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    if type(last_err) is not asyncio.TimeoutError:
                        _LOGGER.warning("MQTT Connection Timeout")
                        last_err = asyncio.TimeoutError()
                    continue
                except Exception as e:
                    if type(last_err) is not type(e) or last_err.args != e.args:
                        _LOGGER.warning("MQTT Connection Error: %s", str(e))
                        last_err = e
                    continue

            if connect_attempts >= max_attempts:
                _LOGGER.error(
                    "Max reconnection attempts (%d) reached, giving up", max_attempts
                )
        finally:
            self.reconnect_evt.clear()
            self.disconnect_evt = None
            self.connect_task = None

    def _on_message(
        self, client: paho_mqtt.Client, userdata: Any, message: paho_mqtt.MQTTMessage
    ) -> None:
        # pylint: disable=unused-argument
        msg = message.payload.decode()
        _LOGGER.debug("Message received on %s: %s", message.topic, msg)
        try:
            data = json.loads(msg)
        except json.decoder.JSONDecodeError:
            _LOGGER.info("Received invalid JSON message: %s", msg)

        if message.topic.startswith("prd/app_subscriptions/"):
            device_id = message.topic.split("/")[2]
        else:
            device_id = None

        for h in self._handlers["update"]:
            asyncio.ensure_future(h(device_id, data))

    def _on_subscribe(
        self,
        client: paho_mqtt.Client,
        userdata: Any,
        mid: int,
        granted_qos: tuple[int] | list[paho_mqtt.ReasonCodes],
        properties: paho_mqtt.Properties | None = None,
    ) -> None:
        # pylint: disable=unused-argument
        if mid in self.pending_acks:
            pending_info = self.pending_acks[mid]

            # Handle both old format (string) and new format (tuple with event)
            if isinstance(pending_info, tuple):
                topic, ack_evt = pending_info
                _LOGGER.info("Subscribed to: %s", topic)
                if topic not in self.topics:
                    self.topics.append(topic)
                ack_evt.set()  # Signal that subscription is complete
            else:
                # Old format - just a topic string
                topic = pending_info
                _LOGGER.info("Subscribed to: %s", topic)
                if topic not in self.topics:
                    self.topics.append(topic)

            del self.pending_acks[mid]
        else:
            _LOGGER.info("Subscribed: %s %s %s", userdata, str(mid), str(granted_qos))

"""Define Kohler Partner class"""

import logging
import re
import uuid
import json
import base64
import binascii

from datetime import datetime, timedelta
from typing import Optional
from aiohttp import ClientSession, ClientTimeout, CookieJar, ClientError


from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from ..errors import KohlerB2CError, KohlerTokenError

_LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT: int = 10


class KOHLER_API:
    """API for Kohler to access Phyn Devices"""

    def __init__(
        self,
        username: str,
        password: str,
        verify_ssl: bool = True,
        proxy: Optional[str] = None,
        proxy_port: Optional[int] = None,
    ):
        self._username: str = username
        self._password: str = password
        self._phyn_password: str = None
        self._user_id: str = None

        self._token: str = None
        self._token_expiration = None
        self._refresh_token = None
        self._refresh_token_expiration = None
        self._mobile_data = None

        self.verify_ssl = verify_ssl
        self.ssl = False if verify_ssl is False else None
        self.proxy = proxy
        self.proxy_port = proxy_port
        self.proxy_url: Optional[str] = None
        if self.proxy is not None and self.proxy_port is not None:
            self.proxy_url = f"https://{proxy}:{proxy_port}"

        self._session: ClientSession = None

    def get_cognito_info(self):
        """Get cognito information"""
        return self._mobile_data["cognito"]

    def get_mqtt_info(self):
        """Get MQTT url"""
        return self._mobile_data["wss"]

    def get_phyn_password(self):
        """Get phyn password"""
        return self._phyn_password

    async def authenticate(self):
        """Authenticate with Kohler and Phyn"""
        use_running_session = self._session and not self._session.closed
        if not use_running_session:
            self._session = ClientSession(
                timeout=ClientTimeout(total=DEFAULT_TIMEOUT),
                cookie_jar=CookieJar(quote_cookie=False),
            )

        await self.b2c_login()
        token = await self.get_phyn_token()
        await self._session.close()
        self._phyn_password = await self.token_to_password(token)

    async def b2c_login(self):
        """Login to Kohler"""
        _LOGGER.debug("Logging into Kohler")

        try:
            client_request_id = str(uuid.uuid4())

            # Get CSRF token and initialize values
            params = {
                "response_type": "code",
                "client_id": "8caf9530-1d13-48e6-867c-0f082878debc",
                "client-request-id": client_request_id,
                "scope": "https%3A%2F%2Fkonnectkohler.onmicrosoft.com%2Ff5d87f3d-bdeb-4933-ab70-ef56cc343744%2Fapiaccess%20"
                + "openid%20offline_access%20profile",
                "redirect_uri": "msauth%3A%2F%2Fcom.kohler.hermoth%2F2DuDM2vGmcL4bKPn2xKzKpsy68k%253D",
                "prompt": "login",
            }
            get_vars = "&".join([f"{x[0]}={x[1]}" for x in params.items()])
            resp = await self._session.get(
                "https://konnectkohler.b2clogin.com/tfp/konnectkohler.onmicrosoft.com/B2C_1A_signin/oAuth2/v2.0/authorize?"
                + get_vars,
                ssl=self.ssl,
                proxy=self.proxy_url,
            )

            if resp.status != 200:
                resp_text = await resp.text()
                _LOGGER.error(
                    "B2C initial request failed with status %s: %s",
                    resp.status,
                    resp_text[:500],
                )
                raise KohlerB2CError(f"B2C initial request failed: HTTP {resp.status}")

            resp_text = await resp.text()
            match = re.search(r'"(StateProperties=([a-zA-Z0-9]+))"', resp_text)
            if not match:
                _LOGGER.error("Could not extract StateProperties from B2C response")
                raise KohlerB2CError(
                    "Could not extract StateProperties from B2C response"
                )
            state_properties = match.group(1)

            cookies = self._session.cookie_jar.filter_cookies(
                "https://konnectkohler.b2clogin.com"
            )
            csrf = None
            for key, cookie in cookies.items():
                if key == "x-ms-cpim-csrf":
                    csrf = cookie.value

            if not csrf:
                _LOGGER.error("CSRF token not found in cookies")
                raise KohlerB2CError("CSRF token not found in cookies")

            # Login
            headers = {
                "X-CSRF-TOKEN": csrf,
            }
            login_vars = {
                "request_type": "RESPONSE",
                "signInName": self._username,
                "password": self._password,
            }
            resp = await self._session.post(
                "https://konnectkohler.b2clogin.com/konnectkohler.onmicrosoft.com/"
                + "B2C_1A_signin/SelfAsserted?p=B2C_1A_signin&"
                + state_properties,
                headers=headers,
                data=login_vars,
                ssl=self.ssl,
                proxy=self.proxy_url,
            )

            if resp.status != 200:
                resp_text = await resp.text()
                _LOGGER.error(
                    "B2C login post failed with status %s: %s",
                    resp.status,
                    resp_text[:500],
                )
                raise KohlerB2CError(f"B2C login post failed: HTTP {resp.status}")

            params = {
                "rememberMe": "false",
                "csrf_token": csrf,
                "tx": state_properties,
                "p": "B2C_1A_signin",
            }
            args = "&".join([f"{x[0]}={x[1]}" for x in params.items()])
            resp = await self._session.get(
                "https://konnectkohler.b2clogin.com/konnectkohler.onmicrosoft.com/"
                + "B2C_1A_signin/api/CombinedSigninAndSignup/confirmed?"
                + args,
                allow_redirects=False,
                ssl=self.ssl,
                proxy=self.proxy_url,
            )

            if resp.status not in [302, 303]:
                resp_text = await resp.text()
                _LOGGER.error(
                    "B2C confirmation failed with status %s: %s",
                    resp.status,
                    resp_text[:500],
                )
                raise KohlerB2CError(f"B2C confirmation failed: HTTP {resp.status}")

            if "Location" not in resp.headers:
                _LOGGER.error("B2C login succeeded but no redirect Location header")
                raise KohlerB2CError("Missing Location header in B2C response")

            matches = re.search(r"code=([^&]+)", resp.headers["Location"])
            if not matches:
                _LOGGER.error(
                    "Could not extract code from Location: %s",
                    resp.headers["Location"][:100],
                )
                raise KohlerB2CError("Could not extract authorization code")
            code = matches.group(1)

            # Get tokens
            headers = {
                "x-app-name": "com.kohler.hermoth",
                "x-app-ver": "2.7",
            }
            params = {
                "client-request-id": client_request_id,
                "client_id": "8caf9530-1d13-48e6-867c-0f082878debc",
                "client_info": "1",
                "x-app-name": "com.kohler.hermoth",
                "x-app-ver": "2.7",
                "redirect_uri": "msauth://com.kohler.hermoth/2DuDM2vGmcL4bKPn2xKzKpsy68k%3D",
                "scope": "https://konnectkohler.onmicrosoft.com/f5d87f3d-bdeb-4933-ab70-ef56cc343744/apiaccess"
                + " openid offline_access profile",
                "grant_type": "authorization_code",
                "code": code,
            }
            resp = await self._session.post(
                "https://konnectkohler.b2clogin.com/tfp/konnectkohler.onmicrosoft.com/"
                + "B2C_1A_signin/%2FoAuth2%2Fv2.0%2Ftoken",
                data=params,
                ssl=self.ssl,
                proxy=self.proxy_url,
            )

            if resp.status != 200:
                resp_text = await resp.text()
                _LOGGER.error(
                    "Failed to get OAuth tokens: HTTP %s: %s",
                    resp.status,
                    resp_text[:500],
                )
                raise KohlerB2CError(f"Failed to get OAuth tokens: HTTP {resp.status}")

            try:
                data = await resp.json()
            except json.JSONDecodeError as e:
                resp_text = await resp.text()
                _LOGGER.error(
                    "Failed to parse OAuth token response as JSON: %s. Response: %s",
                    e,
                    resp_text[:500],
                )
                raise KohlerB2CError(f"Invalid JSON in OAuth response: {e}") from e

            if "client_info" not in data:
                _LOGGER.error("client_info missing from OAuth token response")
                await self._session.close()
                raise KohlerB2CError("client_info missing from OAuth token response")

            try:
                client_info = json.loads(
                    base64.b64decode(data["client_info"] + "==").decode()
                )
            except (KeyError, ValueError, binascii.Error) as e:
                _LOGGER.error("Failed to decode client_info: %s", e)
                await self._session.close()
                raise KohlerB2CError(f"Failed to decode client_info: {e}") from e

            if "uid" not in client_info:
                _LOGGER.error("uid missing from client_info")
                await self._session.close()
                raise KohlerB2CError("uid missing from client_info")

            self._user_id = re.sub("-b2c_1a_signin$", "", client_info["uid"])

            if "access_token" not in data or "expires_in" not in data:
                _LOGGER.error("access_token or expires_in missing from OAuth response")
                await self._session.close()
                raise KohlerB2CError(
                    "access_token or expires_in missing from OAuth response"
                )

            self._token = data["access_token"]
            self._token_expiration = datetime.now() + timedelta(
                seconds=data["expires_in"]
            )
            self._refresh_token = data.get("refresh_token")
            self._refresh_token_expiration = datetime.now() + timedelta(
                seconds=data.get("refresh_token_expires_in", 0)
            )
            _LOGGER.debug("Received Kohler Token")

        except ClientError as e:
            _LOGGER.error("Network error during Kohler B2C login: %s", e)
            raise KohlerB2CError(f"Network error: {e}") from e
        except KohlerB2CError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            _LOGGER.error("Unexpected error during Kohler B2C login: %s", e)
            raise KohlerB2CError(f"Unexpected error: {e}") from e

    async def get_phyn_token(self):
        """Get a phyn access token"""
        try:
            params = {
                "partner": "kohler",
                "partner_user_id": self._user_id,
                "email": self._username,
            }
            args = "&".join([f"{x[0]}={x[1]}" for x in params.items()])
            headers = {
                "Accept": "application/json",
                "Accept-encoding": "gzip",
                "Authorization": f"Bearer partner-{self._token}",
                "Content-Type": "application/json",
                "User-Agent": "okhttp/4.10.0",
            }

            _LOGGER.info("Getting Kohler settings from Phyn")
            resp = await self._session.get(
                f"https://api.phyn.com/settings/app/com.kohler.mobile?{args}",
                headers=headers,
                ssl=self.ssl,
                proxy=self.proxy_url,
            )

            if resp.status != 200:
                resp_text = await resp.text()
                _LOGGER.error(
                    "Failed to get Phyn settings: HTTP %s: %s",
                    resp.status,
                    resp_text[:500],
                )
                raise KohlerTokenError(
                    f"Failed to get Phyn settings: HTTP {resp.status}"
                )

            try:
                mobile_data = await resp.json()
            except json.JSONDecodeError as e:
                resp_text = await resp.text()
                _LOGGER.error(
                    "Failed to parse Phyn settings response as JSON: %s. Response: %s",
                    e,
                    resp_text[:500],
                )
                raise KohlerTokenError(f"Invalid JSON in settings response: {e}") from e

            if "error_msg" in mobile_data:
                _LOGGER.error("Kohler returned error: %s", mobile_data["error_msg"])
                await self._session.close()
                raise KohlerTokenError(mobile_data["error_msg"])

            if "cognito" not in mobile_data:
                _LOGGER.error("Cognito info missing from Phyn settings response")
                await self._session.close()
                raise KohlerTokenError("Missing cognito information")

            if "pws_api" not in mobile_data or "app_api_key" not in mobile_data.get(
                "pws_api", {}
            ):
                _LOGGER.error(
                    "pws_api or app_api_key missing from Phyn settings response"
                )
                await self._session.close()
                raise KohlerTokenError("Missing pws_api or app_api_key")

            self._mobile_data = mobile_data

            _LOGGER.debug("Getting token from Phyn")
            params = {
                "email": self._username,
                "partner": "kohler",
                "partner_user_id": self._user_id,
            }
            args = "&".join([f"{x[0]}={x[1]}" for x in params.items()])
            headers = {
                "Accept": "application/json, text/plain, */*",
                "Accept-encoding": "gzip",
                "Authorization": f"Bearer partner-{self._token}",
                "Content-Type": "application/json",
                "x-api-key": mobile_data["pws_api"]["app_api_key"],
            }
            resp = await self._session.get(
                f"https://api.phyn.com/partner-user-setup/token?{args}",
                headers=headers,
                ssl=self.ssl,
                proxy=self.proxy_url,
            )

            if resp.status != 200:
                resp_text = await resp.text()
                _LOGGER.error(
                    "Failed to get Phyn token: HTTP %s: %s",
                    resp.status,
                    resp_text[:500],
                )
                raise KohlerTokenError(f"Failed to get Phyn token: HTTP {resp.status}")

            try:
                data = await resp.json()
            except json.JSONDecodeError as e:
                resp_text = await resp.text()
                _LOGGER.error(
                    "Failed to parse Phyn token response as JSON: %s. Response: %s",
                    e,
                    resp_text[:500],
                )
                raise KohlerTokenError(f"Invalid JSON in token response: {e}") from e

            if "token" not in data:
                _LOGGER.error("Token not found in response")
                await self._session.close()
                raise KohlerTokenError("Token not found in response")

            _LOGGER.debug("Token received")
            return data["token"]

        except ClientError as e:
            _LOGGER.error("Network error during Phyn token retrieval: %s", e)
            raise KohlerTokenError(f"Network error: {e}") from e
        except KohlerTokenError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            _LOGGER.error("Unexpected error getting Phyn token: %s", e)
            raise KohlerTokenError(f"Unexpected error: {e}") from e

    async def token_to_password(self, token):
        """Convert a token to a Phyn password"""
        try:
            b64hex = base64.b64decode(
                (token + "=" * (5 - (len(token) % 4)))
                .replace("_", "/")
                .replace("-", "+")
            ).hex()
        except (ValueError, binascii.Error) as e:
            _LOGGER.error("Failed to decode token: %s", e)
            raise KohlerTokenError(f"Failed to decode token: {e}") from e

        try:
            if (
                "partner" not in self._mobile_data
                or "comm_id" not in self._mobile_data.get("partner", {})
            ):
                raise KohlerTokenError("partner or comm_id missing from mobile data")
            keydata = binascii.hexlify(
                base64.b64decode(self._mobile_data["partner"]["comm_id"])
            ).decode()
        except (KeyError, ValueError, binascii.Error) as e:
            _LOGGER.error("Error getting password decryption key: %s", e)
            raise KohlerTokenError(f"Error getting password decryption key: {e}") from e

        try:
            key = keydata[32:]
            iv = b64hex[18 : (18 + 32)]
            ct = b64hex[50 : (len(b64hex) - 64)]
            cipher = AES.new(bytes.fromhex(key), AES.MODE_CBC, iv=bytes.fromhex(iv))
            return unpad(cipher.decrypt(bytearray.fromhex(ct)), AES.block_size).decode()
        except Exception as e:
            _LOGGER.error("Error decrypting password: %s", e)
            raise KohlerTokenError(f"Error decrypting password: {e}") from e

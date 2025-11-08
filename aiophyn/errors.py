"""Define package errors."""


class PhynError(Exception):
    """Define a base error."""

    ...


class RequestError(PhynError):
    """Define an error related to invalid requests."""

    ...


class BrandError(Exception):
    """Define an error related to invalid brands."""

    ...


class KohlerAuthError(PhynError):
    """Base Kohler authentication error."""

    ...


class KohlerB2CError(KohlerAuthError):
    """Kohler B2C authentication failed."""

    ...


class KohlerTokenError(KohlerAuthError):
    """Failed to get Kohler tokens."""

    ...

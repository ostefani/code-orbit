class ProviderError(Exception):
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(message)
        self.provider = provider


class UnsupportedChatProviderError(ProviderError):
    pass


class ProviderConfigurationError(ProviderError):
    pass


class ProviderValidationError(ProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class ProviderRequestError(ProviderError):
    pass


class CapabilityNotSupportedError(ProviderError):
    pass

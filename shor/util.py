import enum


class ExitStatus(enum.IntEnum):
    """Enum to track the Exit Status of an iteration of Shor's algorithm"""

    NO_ORDER_FOUND = enum.auto()
    ORDER_CONDITION = enum.auto()
    FAILED_FACTOR = enum.auto()
    SUCCESS = enum.auto()


class ShorError(Exception):
    """Exception that can be raised when Shor's algorithm fails during an iteration."""

    def __init__(self, message, fail_reason, failed_factor=None):
        self.message = message
        self.failed_factors = failed_factor
        self.fail_reason: enum.Enum = fail_reason

"""Custom exceptions for this package."""


class QupathException(Exception):
    """Custom Exception to raise when qupath fails to run."""


class MagnificationException(Exception):
    """Exception to raise when we can't find the slide's magnification."""

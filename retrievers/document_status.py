"""Interpret explicit source status without treating unknown values as facts."""


def is_explicitly_repealed(value: object) -> bool:
    return value is True or isinstance(value, str) and value.strip().casefold() in {"yes", "true", "oui"}

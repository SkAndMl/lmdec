PARAM_UNITS = ("", "K", "M", "B", "T")
BYTE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")


def format_params(total_params: int) -> str:
    size, unit = scale(total_params, base=1_000, units=PARAM_UNITS)
    return f"{format_number(size)}{unit}"


def format_bytes(num_bytes: int) -> str:
    size, unit = scale(num_bytes, base=1_024, units=BYTE_UNITS)
    return f"{format_number(size)} {unit}"


def format_number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def scale(value: int, base: int, units: tuple[str, ...]) -> tuple[float, str]:
    if value < 0:
        raise ValueError(f"value ({value}) cannot be less than 0")

    size = float(value)
    unit_index = 0

    while size >= base and unit_index < len(units) - 1:
        size /= base
        unit_index += 1

    # 1023.996 would render as "1024 B" once rounded to two decimals.
    if round(size, 2) >= base and unit_index < len(units) - 1:
        size /= base
        unit_index += 1

    return size, units[unit_index]

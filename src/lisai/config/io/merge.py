def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries.
    override takes precedence over base.
    """
    result = dict(base)

    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v

    return result

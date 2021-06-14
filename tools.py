def to_clear_ml_params(d, to_ignore=None):
    if to_ignore is None:
        to_ignore = []
    return {k: v for k, v in d.items() if k not in to_ignore}

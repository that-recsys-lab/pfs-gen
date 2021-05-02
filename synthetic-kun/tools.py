def to_clear_ml_params(d, to_ignore=None):
    if to_ignore is None:
        to_ignore = []
    return {k: v for k, v in d.items() if k not in to_ignore}

    r = []
    for k, v in d.items():
        if to_ignore is not None:
            if k in to_ignore:
                continue
        r.append({"name": k, "value": v})
    return r
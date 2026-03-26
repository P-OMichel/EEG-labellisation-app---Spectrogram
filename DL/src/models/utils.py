def _as_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _get_first(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
        if hasattr(d, k):
            return getattr(d, k)
    return default
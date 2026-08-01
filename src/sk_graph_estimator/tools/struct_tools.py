def get_any(d,keys,fallback=None,err=None):
    for k in keys:
        if k in d:
            return d[k]
        
    if err is None:
        return fallback

    raise err
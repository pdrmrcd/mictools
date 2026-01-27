_default_path = None

def set_path(path):
    global _default_path
    _default_path = path

def get_path(path=None):
    if path is not None:
        return path
    if _default_path is None:
        raise ValueError("No default path set. Use set_path() to set a default path.")
    return _default_path
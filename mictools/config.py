import os

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

def get_path_or_none():
    '''Return the default data root if one has been set, otherwise ``None`` (never raises).

    Use this when the absence of a path is a valid, non-exceptional state —
    e.g. when auto-registering an ROI at construction time.
    '''
    return _default_path

def get_analysis_path(path=None, create=False):
    '''
    Return the experiment-level ``analysis/`` directory for the given data root.

    This is where writable, human-curated artifacts (e.g. the ROI registry)
    live, kept separate from the machine-written ``Raw/`` and ``Processed/``
    trees. It resolves to ``{data_root}/analysis`` where the data root comes
    from :func:`get_path` (i.e. respects ``set_path``/explicit ``path=``).

    Resolution is pure by default (like :func:`get_path`); pass ``create=True``
    to create the directory as a side effect. Callers that only need to *read*
    (e.g. "does a registry exist yet?") should leave ``create=False``.
    '''
    analysis_path = os.path.join(os.path.dirname(get_path(path)), "analysis")
    if create:
        os.makedirs(analysis_path, exist_ok=True)
    return analysis_path

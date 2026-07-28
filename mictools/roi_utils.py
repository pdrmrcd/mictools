import os

from .config import get_analysis_path, get_path_or_none

REGISTRY_FILENAME = "roi_registry.yaml"


def _validate_name(name):
    '''
    Validate an ROI name for use as a registry key and a path component.

    ROI names are interpolated raw into HDF5 group paths (e.g.
    ``entry/data/{name}``, ``.../Images/{name}``), so a ``None``, empty, or
    path-hostile name silently corrupts outputs. Raise ``ValueError`` if the
    name is unusable; return the name otherwise.
    '''
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"ROI name must be a non-empty string, got {name!r}.")
    if any(c in name for c in ('/', '\\')) or name != name.strip():
        raise ValueError(
            f"ROI name {name!r} is not path-safe: it must not contain '/' or "
            "'\\' or leading/trailing whitespace."
        )
    return name


class Roi(object):
    def __init__(self, y_start, y_end, x_start, x_end, name=None):
        # Coerce bounds to int at this single choke point so downstream slicing
        # (dset[:, y_start:y_end, ...]) and YAML round-trips are always clean.
        self.y_start = int(y_start)
        self.y_end = int(y_end)
        self.x_start = int(x_start)
        self.x_end = int(x_end)
        if not (0 <= self.y_start < self.y_end):
            raise ValueError(
                f"Invalid y bounds: require 0 <= y_start < y_end, "
                f"got y_start={self.y_start}, y_end={self.y_end}."
            )
        if not (0 <= self.x_start < self.x_end):
            raise ValueError(
                f"Invalid x bounds: require 0 <= x_start < x_end, "
                f"got x_start={self.x_start}, x_end={self.x_end}."
            )
        # name is optional at construction for backward-compat, but validated if
        # provided; presence is enforced at the registry / pipeline boundary.
        self.name = _validate_name(name) if name is not None else None

        # Auto-register in the experiment registry as soon as this ROI is
        # created, if a data root has already been set via set_path().
        if self.name is not None:
            _path = get_path_or_none()
            if _path is not None:
                try:
                    RoiRegistry.load(_path).add(self)
                except ValueError:
                    print(
                        f"[roi_utils] Warning: ROI {self.name!r} is already registered "
                        f"with a different geometry. The registry was NOT updated. "
                        f"To replace the existing definition call "
                        f"RoiRegistry.load().add(roi, override=True)."
                    )

    def as_tuple(self):
        return (self.y_start, self.y_end, self.x_start, self.x_end)

    def to_dict(self):
        '''Serialize the ROI to a plain dict (used by the ROI registry / YAML).'''
        return {
            "name": self.name,
            "y_start": int(self.y_start),
            "y_end": int(self.y_end),
            "x_start": int(self.x_start),
            "x_end": int(self.x_end),
        }

    @classmethod
    def from_dict(cls, d):
        '''Rebuild a Roi from a dict produced by :meth:`to_dict`.

        Raises ``ValueError`` (not a bare ``KeyError``) if a bound key is
        missing, since the registry YAML is hand-editable.
        '''
        missing = [k for k in ("y_start", "y_end", "x_start", "x_end") if k not in d]
        if missing:
            raise ValueError(
                f"ROI entry {d!r} is missing required key(s): {', '.join(missing)}."
            )
        return cls(
            y_start=d["y_start"],
            y_end=d["y_end"],
            x_start=d["x_start"],
            x_end=d["x_end"],
            name=d.get("name"),
        )

    @classmethod
    def _from_registry_entry(cls, d, name):
        '''
        Reconstruct a Roi from a stored registry entry **without** triggering
        auto-registration. Used internally by :class:`RoiRegistry` to break the
        ``get() → from_dict() → __init__() → add()`` recursion cycle.
        '''
        obj = object.__new__(cls)
        obj.y_start = int(d["y_start"])
        obj.y_end   = int(d["y_end"])
        obj.x_start = int(d["x_start"])
        obj.x_end   = int(d["x_end"])
        obj.name    = name
        return obj

    def same_geometry(self, other):
        '''True if the pixel bounds match (name is ignored).'''
        if not isinstance(other, Roi):
            return NotImplemented
        return self.as_tuple() == other.as_tuple()

    def __eq__(self, other):
        if not isinstance(other, Roi):
            return NotImplemented
        return self.as_tuple() == other.as_tuple() and self.name == other.name

    def __hash__(self):
        # Defining __eq__ sets __hash__ to None in Python 3; restore hashability
        # (consistent with __eq__) so Roi can live in sets/dicts.
        return hash((self.as_tuple(), self.name))

    def __repr__(self):
        return (
            f"Roi(y_start={self.y_start}, y_end={self.y_end}, "
            f"x_start={self.x_start}, x_end={self.x_end}, name={self.name!r})"
        )


class RoiRegistry(object):
    '''
    A per-experiment, writable registry of named ROIs.

    Backed by a YAML sidecar at ``{data_root}/analysis/roi_registry.yaml`` so
    users define ROIs once and reuse them across scans. ROI **names must be
    unique**; redefining a name with a different geometry is an *override* that
    archives the previous definition — along with the list of scans already
    processed with it — into a ``history`` block, so prior maps stay traceable.

    The on-disk schema (version 1)::

        version: 1
        rois:
          roi1: {y_start: 100, y_end: 200, x_start: 150, x_end: 250,
                 used_by_scans: [40, 41]}
        history:
          roi1:
            - {y_start: 90, y_end: 190, x_start: 150, x_end: 250,
               used_by_scans: [12, 13]}

    Load with :meth:`load`, mutate with :meth:`add` / :meth:`record_usage`, and
    persist with :meth:`save` (mutating methods save automatically).
    '''

    VERSION = 1

    def __init__(self, path=None, rois=None, history=None):
        # `path` is the data root (respects set_path / explicit path=), resolved
        # lazily on save/load rather than stored eagerly.
        self._path = path
        self._rois = rois if rois is not None else {}
        self._history = history if history is not None else {}

    # -- persistence ---------------------------------------------------------

    @classmethod
    def registry_file(cls, path=None, create=False):
        '''Absolute path to the registry YAML for the given data root.'''
        return os.path.join(get_analysis_path(path, create=create), REGISTRY_FILENAME)

    @classmethod
    def load(cls, path=None):
        '''
        Read the registry sidecar for the given data root.

        Returns an empty registry if the file does not exist yet. Resolving the
        path never creates the ``analysis/`` directory (that happens on save).
        '''
        import yaml

        reg_file = cls.registry_file(path, create=False)
        if not os.path.exists(reg_file):
            return cls(path=path)
        with open(reg_file, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            path=path,
            rois=data.get("rois", {}) or {},
            history=data.get("history", {}) or {},
        )

    def save(self, path=None):
        '''Persist the registry to its YAML sidecar (creates ``analysis/``).'''
        import yaml

        if path is not None:
            self._path = path
        reg_file = self.registry_file(self._path, create=True)
        payload = {
            "version": self.VERSION,
            "rois": self._rois,
            "history": self._history,
        }
        with open(reg_file, "w") as f:
            yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=True)
        return reg_file

    # -- queries -------------------------------------------------------------

    def names(self):
        '''Sorted list of registered ROI names.'''
        return sorted(self._rois.keys())

    def list(self):
        '''All registered ROIs as :class:`Roi` objects.'''
        return [self.get(name) for name in self.names()]

    def get(self, name):
        '''Return the registered :class:`Roi` for ``name`` (``ValueError`` if absent).'''
        if name not in self._rois:
            raise ValueError(
                f"No ROI named {name!r} in the registry. "
                f"Known ROIs: {self.names()}."
            )
        # Use _from_registry_entry (bypasses __init__) to avoid the recursion:
        # get() → from_dict() → __init__() → RoiRegistry.add() → get() → ...
        return Roi._from_registry_entry(self._rois[name], name)

    def used_by_scans(self, name):
        '''Scans already processed with ``name`` (``ValueError`` if absent).'''
        if name not in self._rois:
            raise ValueError(f"No ROI named {name!r} in the registry.")
        return list(self._rois[name].get("used_by_scans", []))

    # -- mutations -----------------------------------------------------------

    def add(self, roi, override=False):
        '''
        Register ``roi`` under its (required, unique) name.

        - New name: added with an empty ``used_by_scans`` list.
        - Existing name, identical geometry: idempotent no-op.
        - Existing name, different geometry, ``override=False``: raise
          ``ValueError`` reporting the prior geometry and the scans that used it.
        - Existing name, different geometry, ``override=True``: archive the prior
          definition (with its ``used_by_scans``) into ``history`` and install
          the new one with a fresh ``used_by_scans`` list.

        Saves on any change and returns ``roi``.
        '''
        if not isinstance(roi, Roi):
            raise ValueError("add() expects a Roi instance.")
        name = _validate_name(roi.name)

        if name in self._rois:
            existing = self.get(name)
            if existing.same_geometry(roi):
                return roi  # idempotent re-declaration
            if not override:
                raise ValueError(
                    f"ROI name {name!r} already defined with a different geometry "
                    f"{existing.as_tuple()} (used by scans "
                    f"{self.used_by_scans(name)}). Pass override=True to replace it; "
                    f"the previous definition and its scan list will be archived."
                )
            # Override: archive the prior definition + its usage.
            self._history.setdefault(name, []).append(dict(self._rois[name]))

        entry = roi.to_dict()
        entry.pop("name", None)  # name is the dict key, not stored inside
        entry["used_by_scans"] = []
        self._rois[name] = entry
        self.save()
        return roi

    def record_usage(self, name, scanno):
        '''
        Record that ``scanno`` was processed with ROI ``name`` (dedup, then save).

        Raises ``ValueError`` if ``name`` is not registered — usage tracking
        requires the ROI to have been defined first.
        '''
        if name not in self._rois:
            raise ValueError(
                f"Cannot record usage: no ROI named {name!r} in the registry."
            )
        scans = self._rois[name].setdefault("used_by_scans", [])
        if int(scanno) not in scans:
            scans.append(int(scanno))
            scans.sort()
            self.save()
        return scans

    def remove(self, name):
        '''Remove ``name`` from the active registry (kept in history). Saves.'''
        if name not in self._rois:
            raise ValueError(f"No ROI named {name!r} in the registry.")
        self._history.setdefault(name, []).append(dict(self._rois[name]))
        del self._rois[name]
        self.save()

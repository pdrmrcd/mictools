import os

from .config import get_analysis_path, get_path_or_none

REGISTRY_FILENAME = "roi_registry.yaml"

def _resolve_roi(roi, path, register=False, override=False):
    '''
    Normalize the ``roi`` argument to a :class:`Roi` (or ``None``) via the registry.

    Accepts either a :class:`Roi` instance or a registered ROI **name** (str):

    - ``None`` -> ``None`` (channel/current mode).
    - ``str``  -> looked up in the experiment's ROI registry (``ValueError`` if unknown).
    - ``Roi``  -> validated; a falsy name is rejected (guards ``Scan_XXXX_None.h5``).
                  If its name is already registered with a *different* geometry,
                  raise ``ValueError`` unless ``override=True`` (this is what closes
                  the silent stale-cache collision, since caching keys on name).
                  When ``register=True`` the ROI is added to the registry so its
                  usage can be tracked.
    '''
    if roi is None:
        return None
    if isinstance(roi, str):
        return RoiRegistry.load(path).get(roi)
    if not isinstance(roi, Roi):
        raise ValueError(
            "roi must be a Roi instance (roi_utils.Roi) or a registered ROI name (str)."
        )
    if not roi.name:
        raise ValueError(
            "roi.name must be set (it is used to build HDF5 group paths)."
        )
    reg = RoiRegistry.load(path)
    if roi.name in reg.names() and not reg.get(roi.name).same_geometry(roi) and not override:
        raise ValueError(
            f"ROI name {roi.name!r} is already registered with a different geometry "
            f"{reg.get(roi.name).as_tuple()}. Pass roi_override=True to replace it "
            f"(the previous definition and its scan list are archived), or use a new name."
        )
    if register:
        reg.add(roi, override=override)
    return roi

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
    '''
    A 1D, 2D, or 3D region of interest within an area-detector frame.

    Dimensionality is inferred from which bound pairs are supplied: ``x_start``/
    ``x_end`` are always required (x is the mandatory, innermost/fastest axis,
    so a 1D ROI is x-only). Adding ``y_start``/``y_end`` makes it 2D. Adding
    ``z_start``/``z_end`` on top of that makes it 3D (z cannot be given without
    y - dimensions can't be skipped). Axes are ordered outer-to-inner as
    z, y, x, matching the assumed on-disk dataset layout
    ``(n_frames, [z,] [y,] x)`` - this is a naming convention, not something
    verified against a real 3D fixture in this repo.
    '''

    def __init__(self, y_start=None, y_end=None, x_start=None, x_end=None, name=None,
                 z_start=None, z_end=None):
        # Coerce bounds to int at this single choke point so downstream slicing
        # (dset[:, y_start:y_end, ...]) and YAML round-trips are always clean.
        if x_start is None or x_end is None:
            raise ValueError(
                f"x_start and x_end are required: an Roi spans at least the x "
                f"axis (got x_start={x_start!r}, x_end={x_end!r})."
            )
        if (y_start is None) != (y_end is None):
            raise ValueError(
                f"y_start and y_end must both be provided or both omitted, "
                f"got y_start={y_start!r}, y_end={y_end!r}."
            )
        if (z_start is None) != (z_end is None):
            raise ValueError(
                f"z_start and z_end must both be provided or both omitted, "
                f"got z_start={z_start!r}, z_end={z_end!r}."
            )
        if z_start is not None and y_start is None:
            raise ValueError(
                "z_start/z_end require y_start/y_end to also be provided "
                "(an Roi cannot skip the y dimension when z is present)."
            )

        self.x_start = int(x_start)
        self.x_end = int(x_end)
        if not (0 <= self.x_start < self.x_end):
            raise ValueError(
                f"Invalid x bounds: require 0 <= x_start < x_end, "
                f"got x_start={self.x_start}, x_end={self.x_end}."
            )

        if y_start is not None:
            self.y_start = int(y_start)
            self.y_end = int(y_end)
            if not (0 <= self.y_start < self.y_end):
                raise ValueError(
                    f"Invalid y bounds: require 0 <= y_start < y_end, "
                    f"got y_start={self.y_start}, y_end={self.y_end}."
                )
        else:
            self.y_start = None
            self.y_end = None

        if z_start is not None:
            self.z_start = int(z_start)
            self.z_end = int(z_end)
            if not (0 <= self.z_start < self.z_end):
                raise ValueError(
                    f"Invalid z bounds: require 0 <= z_start < z_end, "
                    f"got z_start={self.z_start}, z_end={self.z_end}."
                )
        else:
            self.z_start = None
            self.z_end = None

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

    @property
    def ndim(self):
        '''Number of spatial axes this ROI spans (1, 2, or 3).'''
        if self.z_start is not None:
            return 3
        if self.y_start is not None:
            return 2
        return 1

    def axis_names(self):
        '''Axis names present, ordered outer-to-inner (e.g. ``['y', 'x']`` for a 2D ROI).'''
        return {1: ['x'], 2: ['y', 'x'], 3: ['z', 'y', 'x']}[self.ndim]

    def axis_ranges(self):
        '''``(start, end)`` bounds for each present axis, ordered outer-to-inner.'''
        return tuple(
            (getattr(self, f'{n}_start'), getattr(self, f'{n}_end'))
            for n in self.axis_names()
        )

    def as_tuple(self):
        return tuple(v for pair in self.axis_ranges() for v in pair)

    def to_dict(self):
        '''Serialize the ROI to a plain dict (used by the ROI registry / YAML).'''
        d = {"name": self.name}
        for n, (start, end) in zip(self.axis_names(), self.axis_ranges()):
            d[f'{n}_start'] = int(start)
            d[f'{n}_end'] = int(end)
        return d

    @classmethod
    def from_dict(cls, d):
        '''Rebuild a Roi from a dict produced by :meth:`to_dict`.

        Dimensionality is inferred from which bound keys are present (``x``
        always required; ``y`` and/or ``z`` present makes it 2D/3D). Raises
        ``ValueError`` (not a bare ``KeyError``) if a required bound key is
        missing, since the registry YAML is hand-editable.
        '''
        missing_x = [k for k in ("x_start", "x_end") if k not in d]
        if missing_x:
            raise ValueError(
                f"ROI entry {d!r} is missing required key(s): {', '.join(missing_x)}."
            )
        kwargs = dict(x_start=d["x_start"], x_end=d["x_end"], name=d.get("name"))

        if "y_start" in d or "y_end" in d:
            missing_y = [k for k in ("y_start", "y_end") if k not in d]
            if missing_y:
                raise ValueError(
                    f"ROI entry {d!r} is missing required key(s): {', '.join(missing_y)}."
                )
            kwargs["y_start"] = d["y_start"]
            kwargs["y_end"] = d["y_end"]

        if "z_start" in d or "z_end" in d:
            missing_z = [k for k in ("z_start", "z_end") if k not in d]
            if missing_z:
                raise ValueError(
                    f"ROI entry {d!r} is missing required key(s): {', '.join(missing_z)}."
                )
            kwargs["z_start"] = d["z_start"]
            kwargs["z_end"] = d["z_end"]

        return cls(**kwargs)

    @classmethod
    def _from_registry_entry(cls, d, name):
        '''
        Reconstruct a Roi from a stored registry entry **without** triggering
        auto-registration. Used internally by :class:`RoiRegistry` to break the
        ``get() → from_dict() → __init__() → add()`` recursion cycle.
        '''
        obj = object.__new__(cls)
        for n in ('x', 'y', 'z'):
            if f'{n}_start' in d:
                setattr(obj, f'{n}_start', int(d[f'{n}_start']))
                setattr(obj, f'{n}_end', int(d[f'{n}_end']))
            else:
                setattr(obj, f'{n}_start', None)
                setattr(obj, f'{n}_end', None)
        obj.name = name
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
        bounds = ', '.join(
            f'{n}_start={s}, {n}_end={e}'
            for n, (s, e) in zip(self.axis_names(), self.axis_ranges())
        )
        return f"Roi({bounds}, name={self.name!r})"


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

    ``x_start``/``x_end`` are always present. ``y_start``/``y_end`` appear for
    2D and 3D ROIs; ``z_start``/``z_end`` appear only for 3D ROIs (see
    :class:`Roi`).

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

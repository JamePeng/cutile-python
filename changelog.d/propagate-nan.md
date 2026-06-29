- Added a ``propagate_nan`` option to ``ct.min()``, ``ct.max()``,
  ``ct.argmin()``, ``ct.argmax()``, ``ct.minimum()`` and ``ct.maximum()``. By
  default ``NaN`` values are ignored; with ``propagate_nan=True`` a ``NaN``
  propagates -- ``min``/``max``/``minimum``/``maximum`` return ``NaN`` and
  ``argmin``/``argmax`` return the index of the first ``NaN``.
- Fixed ``ct.argmin()`` and ``ct.argmax()`` to ignore ``NaN`` values under the
  default ``propagate_nan=False``, consistent with ``ct.min()`` and ``ct.max()``.

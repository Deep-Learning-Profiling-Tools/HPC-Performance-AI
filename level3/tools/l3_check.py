"""l3_check -- shared helpers for the Level 3 validators.

Every numeric quantity that enters a pass/fail decision goes through require_finite,
so a NaN/Inf (a diverged solve, an uninitialised field, a truncated reference) is an
explicit FAIL with a named quantity -- never a silently-swallowed comparison. The
validators import this via  sys.path.insert(0, os.environ["L3_TOOLS"]).
"""
import math
import sys


class ValidationError(Exception):
    pass


def require_finite(name, x):
    """Return float(x) if finite; raise ValidationError naming the quantity otherwise."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} is not a number ({x!r})")
    if not math.isfinite(v):
        raise ValidationError(f"{name} is not finite ({v})")
    return v


def require_finite_seq(name, xs):
    xs = list(xs)
    if not xs:
        raise ValidationError(f"{name}: empty sequence (no data)")
    return [require_finite(f"{name}[{i}]", x) for i, x in enumerate(xs)]


def rel_error(name, got, ref):
    g = require_finite(f"{name}(got)", got)
    r = require_finite(f"{name}(ref)", ref)
    return abs(g - r) / max(abs(r), 1e-300)


def fail(msg):
    print(f"  VALIDATION ERROR: {msg}")
    sys.exit(1)

"""The "none" collector: no profiler at all.

Works on any hardware from day one: the ROI markers and a monotonic clock give the
clean ROI wall time and the application's FOM; every device-side column is null
(not observable), never 0. Use it for a platform whose vendor adapter does not exist
yet or has not passed the conformance probe.
"""

NAME = "none"
RUNTIME = None
VERIFIED = True
CAPABILITIES = frozenset()


class Trace:
    def info(self):
        return {"collector": NAME}

    def markers(self):
        return []

    def intervals(self):
        return iter(())

    def op_names(self, keys):
        return {}

    def runtime_calls(self, windows=None):
        return None

    def close(self):
        pass


def open(raw_dir):  # noqa: A001
    return Trace()

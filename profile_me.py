import cProfile
import pstats
import io
from functools import wraps


class GlobalProfiler:
    # Class-level storage for all profiling data
    _stats_dict = {}

    @classmethod
    def profile(cls, func):
        """Static decorator to profile functions across the entire app."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            if func.__qualname__ not in cls._stats_dict:
                cls._stats_dict[func.__qualname__] = cProfile.Profile()

            # Use __qualname__ to distinguish between methods in different classes
            prof = cls._stats_dict[func.__qualname__]

            prof.enable()
            try:
                return func(*args, **kwargs)
            finally:
                prof.disable()

        return wrapper

    @classmethod
    def report(cls, sort_by='cumulative', limit=15):
        """Prints the aggregated report for all decorated functions."""
        if not cls._stats_dict:
            print("No profiling data captured.")
            return

        output = io.StringIO()
        output.write("\n" + "=" * 60 + "\n")
        output.write(" STATIC GLOBAL PROFILING REPORT ".center(60, "=") + "\n")
        output.write("=" * 60 + "\n")

        for name, pr in cls._stats_dict.items():
            output.write(f"\n[FUNCTION]: {name}\n")
            ps = pstats.Stats(pr, stream=output).sort_stats(sort_by)
            ps.print_stats(limit)

        output.write("=" * 60 + "\n")
        print(output.getvalue())

    @classmethod
    def clear(cls):
        """Resets all captured stats."""
        cls._stats_dict = {}
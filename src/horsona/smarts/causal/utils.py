import math


def merge_diff(value_type, orig_value, change):
    if value_type == "boolean":
        original = min(max(original, 1e-10), 1)
        log = math.log(original)
        return math.exp(log + change)
    else:  # continuous
        squared = original * original * (1 if original >= 0 else -1)
        new_squared = squared + change
        return math.sqrt(abs(new_squared)) * (1 if new_squared >= 0 else -1)

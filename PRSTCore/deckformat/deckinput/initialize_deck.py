"""Initialize ECLIPSE deck structures."""


def initialize_deck():
    """Create an empty ECLIPSE deck structure."""
    return {
        "RUNSPEC": {},
        "GRID": {},
        "PROPS": {},
        "REGIONS": {},
        "SOLUTION": {},
        "SUMMARY": {},
        "SCHEDULE": {},
        "UnhandledKeywords": {},
    }

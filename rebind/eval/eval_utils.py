@staticmethod
def parse_yes_no(response: str) -> str:
    """Parse yes/no from model response.

    Args:
        response: Model's response string

    Returns:
        'yes' or 'no' based on response
    """
    response = response.lower()
    if "yes" in response:
        return "yes"
    elif "no" in response:
        return "no"
    else:
        return "unknown"

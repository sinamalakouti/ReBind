def parse_intermediate_prompts(file_path: str) -> dict:
    """
    Parse a file containing intermediate prompts and store them in a dictionary.

    Args:
        file_path: Path to the JSON file containing intermediate prompts

    Returns:
        dict: Dictionary with structure:
        {
            "one": [
                {
                    "triplet": str,
                    "prompt": str,
                    "changes": str,
                    "reasoning": str
                },
                ...
            ],
            "two": [...],
            "three": [...]
        }
    """
    import json

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Validate expected structure
        required_keys = ["one", "two", "three"]
        required_fields = ["triplet", "prompt", "changes", "reasoning"]

        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

            # Validate each entry in the lists
            for entry in data[key]:
                missing_fields = [
                    field for field in required_fields if field not in entry
                ]
                if missing_fields:
                    raise ValueError(f"Entry in {key} missing fields: {missing_fields}")

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")

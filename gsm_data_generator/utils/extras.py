import json


def read_json(file_path: str):
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return dict(data)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None  # You can choose to return None or raise a custom exception here
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{file_path}': {e}")
        return None  # You can choose to return None or raise a custom exception here


def copy_function(x):
    return str(x)

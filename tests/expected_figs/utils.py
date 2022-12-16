# pylint: disable=missing-function-docstring, missing-class-docstring


def expected_json_string(filename: str) -> str:

    with open(f"tests/expected_figs/json_strings/{filename}", encoding="utf-8") as file:
        json_sting = file.read().rstrip().replace("'", "")
    return json_sting

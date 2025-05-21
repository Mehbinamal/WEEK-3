import io
import contextlib
from pylint import lint
from typing import Dict, List, Union

def debug_code_with_pylint(file_path: str) -> Dict[str, Union[str, List[str]]]:
    """Analyze Python code using Pylint and return the score and messages.
    
    Args:
        file_path (str): Path to the Python file to analyze
        
    Returns:
        Dict[str, Union[str, List[str]]]: A dictionary containing:
            - score (str): The Pylint score message
            - messages (List[str]): List of Pylint messages/warnings
    """
    pylint_output = io.StringIO()

    # Redirect stdout to capture Pylint's output
    with contextlib.redirect_stdout(pylint_output):
        lint.Run([file_path], exit=False)

    output = pylint_output.getvalue()
    score_line = ""
    messages = []

    for line in output.splitlines():
        if "Your code has been rated at" in line:
            score_line = line.strip()
        elif ": " in line:
            messages.append(line.strip())

    return {
        "score": score_line,
        "messages": messages
    }


import io
import contextlib
from pylint import lint

def debug_code_with_pylint(file_path):
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


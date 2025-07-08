import subprocess
import json
import xml.etree.ElementTree as ET
import os # Added for example usage file cleanup

def run_pylint(filepath, config_path=None, output_format="json"):
    """
    Runs Pylint on a given Python file and returns the parsed output.
    """
    if not filepath.endswith(".py"):
        return {"error": "Pylint can only analyze Python (.py) files."}

    command = ["pylint", filepath, f"--output-format={output_format}"]
    if config_path:
        command.append(f"--rcfile={config_path}")

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode != 0 and output_format == "json": # Pylint exits with non-zero for issues found
            try:
                # Pylint's JSON output is a list of dictionaries, one per message
                return json.loads(process.stdout) if process.stdout.strip() else []
            except json.JSONDecodeError:
                return {"error": "Failed to decode Pylint JSON output.", "details": process.stdout, "stderr": process.stderr}
        elif process.returncode == 0 and output_format == "json": # No issues found
             return []
        elif output_format != "json": # For text or other formats, return raw output
            return {"stdout": process.stdout, "stderr": process.stderr, "returncode": process.returncode}
        # Handle other pylint exit codes if necessary (e.g. fatal errors)
        if process.returncode > 0 and (process.returncode & 1 or process.returncode & 2 or process.returncode & 4 or process.returncode & 8 or process.returncode & 16): # Bitmask for different error types
             # If it's an issue code, it's fine, JSON should have been produced.
             # If it's a fatal error (32), then we have a problem.
             pass # Already handled if JSON was produced.
        if process.returncode == 32: # Fatal error
            return {"error": "Pylint reported a fatal error.", "details": process.stderr or process.stdout}


    except FileNotFoundError:
        return {"error": "Pylint command not found. Please ensure it's installed and in PATH."}
    except Exception as e:
        return {"error": f"An unexpected error occurred while running Pylint: {str(e)}"}

def run_cppcheck(filepath, config_path=None, enable_all_checks=True):
    """
    Runs Cppcheck on a given C/C++ file and returns parsed XML output.
    """
    if not any(filepath.endswith(ext) for ext in ['.c', '.cpp', '.h', '.hpp', '.cc', '.cxx', '.hh', '.hxx']):
        return {"error": "Cppcheck is intended for C/C++ source files."}

    command = ["cppcheck", "--enable=all" if enable_all_checks else "--enable=warning,style,performance,portability", "--xml", filepath]
    if config_path: # Cppcheck doesn't have a direct rcfile like pylint, but uses --std or project files.
                    # This might need more sophisticated handling for specific project configs.
                    # For now, config_path is a placeholder for future enhancement.
        print(f"[Cppcheck Wrapper] Config path for Cppcheck is not directly used in this basic wrapper: {config_path}")

    try:
        # Cppcheck writes XML to stderr
        process = subprocess.run(command, capture_output=True, text=True, check=False)

        # Cppcheck returns 0 if no errors, non-zero if errors are found or other issues.
        # The XML output is on stderr.
        if not process.stderr.strip():
            if process.stdout.strip(): # Check stdout for any messages if stderr is empty
                 print(f"[Cppcheck Wrapper] Info: Cppcheck stdout: {process.stdout}")
            return {"results": [], "summary": "No issues found or no XML output to stderr."}

        try:
            root = ET.fromstring(process.stderr)
            errors = []
            for error_element in root.findall(".//error"):
                error_data = {
                    "id": error_element.get("id"),
                    "severity": error_element.get("severity"),
                    "msg": error_element.get("msg"),
                    "verbose": error_element.get("verbose"),
                    "cwe": error_element.get("cwe"),
                    "locations": []
                }
                for loc_element in error_element.findall(".//location"):
                    error_data["locations"].append({
                        "file": loc_element.get("file"),
                        "line": loc_element.get("line"),
                        "column": loc_element.get("column"),
                        "info": loc_element.get("info")
                    })
                errors.append(error_data)
            return {"results": errors, "summary": f"Found {len(errors)} issues."}
        except ET.ParseError:
            return {"error": "Failed to parse Cppcheck XML output.", "details": process.stderr, "stdout": process.stdout}

    except FileNotFoundError:
        return {"error": "Cppcheck command not found. Please ensure it's installed and in PATH."}
    except Exception as e:
        return {"error": f"An unexpected error occurred while running Cppcheck: {str(e)}"}

if __name__ == '__main__':
    # Example Usage (for testing the module directly)
    print("Testing Pylint (requires a test.py file)...")
    with open("test.py", "w") as f:
        f.write("import os\n\ndef myfunc():\n  unused_variable = 1\n  print(os.listdir())\n\n# Missing docstring\nclass MyClass:\n  pass\n")
    pylint_results = run_pylint("test.py")
    print(json.dumps(pylint_results, indent=2))
    os.remove("test.py")

    print("\nTesting Cppcheck (requires a test.cpp file)...")
    with open("test.cpp", "w") as f:
        f.write("struct S { int i; }; // Unused struct member 'i'\nint main() { char arr[10]; arr[10] = 0; return 0; } // Buffer overflow and unused struct member\n")
    cppcheck_results = run_cppcheck("test.cpp")
    print(json.dumps(cppcheck_results, indent=2))
    os.remove("test.cpp")

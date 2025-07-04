# Provides a way of using operating system dependent functionality.
import os
# For advanced control over the import system.
import importlib.util
# Imports our custom data validation function.
from utils import validate_data


# A dictionary to act as a cache for our loaded rule modules.
# This prevents us from having to rediscover and reload modules every time.
_rule_modules = {}


def _discover_rules():
    """Dynamically discover and load all coaching rule modules."""

    # 1. Define the base path to start searching for rules.
    rules_base_path = os.path.join(os.path.dirname(__file__), "rules")

    # 2. Use os.walk() to recursively go through all directories and files.
    # It explores the entire directory tree automatically.
    for dirpath, _, filenames in os.walk(rules_base_path):
        # 3. Iterate through all files found in the current directory.
        for file_name in filenames:
            # 4. Check if the file is a Python file and not a special system file.
            if file_name.endswith(".py") and not file_name.startswith("__"):
                # 5. Get the module name from the file name (e.g., 'franco.py' -> 'franco').
                module_name = file_name[:-3]

                # 6. Get the full, absolute path to the Python file.
                file_path = os.path.join(dirpath, file_name)

                # 7. Create a unique name for the module based on its path.
                # Example: 'rules.roles.tank.franco'
                relative_path = os.path.relpath(
                    file_path, os.path.dirname(__file__)
                )
                module_import_name = os.path.splitext(relative_path)[
                    0].replace(os.sep, '.')

                # 8. Programmatically load the module from its file path.
                # Create a "module specification" which tells Python how to load it.
                spec = importlib.util.spec_from_file_location(
                    module_import_name, file_path
                )
                # Create the module object from the specification.
                module = importlib.util.module_from_spec(spec)
                # Execute the module's code to make its contents available.
                spec.loader.exec_module(module)

                # 9. Store the loaded module in our cache dictionary.
                # The key is the hero's name, and the value is the module object.
                _rule_modules[module_name] = module


def generate_feedback(match_data, include_severity=False):
    """
    Generates coaching feedback by validating data and finding the correct rule module.
    
    Args:
        match_data: Dictionary containing match statistics
        include_severity: If True, returns tuples of (severity, message).
                         If False, returns just messages for backward compatibility.
    
    Returns:
        List of feedback messages or (severity, message) tuples
    """

    # Validate the incoming match data using the function from utils.py.
    if not validate_data(match_data):
        if include_severity:
            return [("error", "Invalid match data format. Please check your input.")]
        return ["Invalid match data format. Please check your input."]

    # Get the hero's name from the match data.
    hero = match_data.get("hero")

    # Check if a rule module for this hero was found and loaded.
    if hero in _rule_modules:
        # Get match duration if available
        minutes = match_data.get('match_duration')
        
        # Check if the module has an evaluate function
        if hasattr(_rule_modules[hero], 'evaluate'):
            # Call the evaluate function with match duration support
            try:
                # Try calling with minutes parameter (new format)
                feedback = _rule_modules[hero].evaluate(match_data, minutes)
            except TypeError:
                # Fall back to old format without minutes
                feedback = _rule_modules[hero].evaluate(match_data)
            
            # Handle different return formats
            if feedback and isinstance(feedback[0], tuple):
                # New format with severity levels
                return feedback if include_severity else [msg for _, msg in feedback]
            else:
                # Old format - just messages
                return feedback
        else:
            if include_severity:
                return [("error", f"No evaluate function found for hero: {hero}")]
            return [f"No evaluate function found for hero: {hero}"]
    else:
        # If no matching rule is found, return a helpful message.
        if include_severity:
            return [("warning", f"No coaching logic found for hero: {hero}")]
        return [f"No coaching logic found for hero: {hero}"]


# Run the discovery process once when the application first starts.
# This populates the _rule_modules cache so it's ready for requests.
_discover_rules()
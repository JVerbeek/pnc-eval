import importlib
import yaml

def import_object_from_string(dotted_path):
    """Import an object (function, class, etc.) from a dotted module path like 'module.submodule.func'."""
    module_path, obj_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

def handle_open_file(filename):
    try:
        if filename is None:
            return {}
        with open(filename, "r") as f:
                kwargs = yaml.safe_load(f)
        return kwargs
    except FileNotFoundError:
        print(f"File not found for {filename}. Did you supply the correct path?")
        sys.exit()

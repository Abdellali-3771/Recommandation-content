import os

def print_tree(dir_path, indent=""):
    try:
        items = sorted(os.listdir(dir_path))
    except PermissionError:
        return
    
    for index, item in enumerate(items):
        if item.startswith("."):
            continue
        
        path = os.path.join(dir_path, item)
        
        # ASCII only
        connector = "+-- " if index < len(items) - 1 else "\\-- "
        print(indent + connector + item)
        
        if os.path.isdir(path):
            extension = "|   " if index < len(items) - 1 else "    "
            print_tree(path, indent + extension)

if __name__ == "__main__":
    print("Project Tree:\n")
    print_tree(".")

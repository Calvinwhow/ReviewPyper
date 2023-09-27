import ast
import os

def find_imports_in_file(file_path):
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read(), filename=file_path)

    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imported_modules.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for n in node.names:
                imported_modules.append(f"{module}.{n.name}" if module else n.name)
    return imported_modules

def find_imports_in_dir(dir_path):
    imports = set()
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                imports.update(find_imports_in_file(file_path))
    return imports

if __name__ == '__main__':
    dir_path = '.'  # This means the script will look in the current directory
    imports = find_imports_in_dir(dir_path)
    with open('requirements.txt', 'w') as f:
        for package in imports:
            f.write(f"{package}\n")

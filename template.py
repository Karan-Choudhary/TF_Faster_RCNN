import os

dirs = [
    "data",
    "notebooks",
    "saved_models",
    "src",
    "Reports",
    "Results"
]

for dir_ in dirs:
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_,'.gitkeep'), 'w') as f:
        pass

files = [
    "dvc.yaml",
    "params.yaml",
    ".gitignore",
    "README.md",
    os.path.join("src","__init__.py")
]

for file_ in files:
    with open(file_, 'w') as f:
        pass
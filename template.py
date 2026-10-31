import os
from pathlib import Path

list_of_files = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "app.py",
    "src/__init__.py",
    "src/config.py",
    "src/prompts.py",
    "src/chains.py",
    "src/cache_manager.py",
    "src/utils.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w", encoding="utf-8") as f:
            pass
        print(f"فائل بن گئی: {filepath}")
    else:
        print(f"فائل پہلے سے موجود ہے: {filepath}")
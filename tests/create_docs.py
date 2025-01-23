#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import sys
import pydoc

os.system("uv add toml")

import toml

SRC_DIR = "../src/agefreighter/"
sys.path.append(os.path.join(os.path.dirname(__file__), SRC_DIR))


def create_pydoc():
    DOCS_DIR = os.path.abspath("../docs/")
    files = glob.glob(f"{SRC_DIR}*.py")
    print(files)
    for module_path in files:
        if "__init__.py" not in module_path:
            base_name = os.path.basename(module_path)
            module_name = os.path.splitext(base_name)[0]
            if module_name != "agefreighter":
                module = __import__(module_name)
            else:
                module = __import__(
                    "agefreighter.agefreighter", fromlist=["agefreighter"]
                )
            doc = pydoc.plain(pydoc.render_doc(module))

            file_path = os.path.abspath(module.__file__)
            doc = doc.replace("agefreighter.agefreighter", "agefreighter")
            doc = doc.replace("agefreighter.AgeFreighter", "AgeFreighter")
            doc = doc.replace(file_path, base_name)

            with open(f"{DOCS_DIR}/{module_name}.txt", "w") as f:
                f.write(doc)


def extract_requirements():
    toml_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../pyproject.toml")
    )
    with open(toml_file, "r") as f:
        toml_dict = toml.load(f)
    req_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../requirements.txt")
    )
    with open(req_file, "w") as f:
        for dep in toml_dict["project"]["dependencies"]:
            f.write(dep + "\n")


def main():
    create_pydoc()
    extract_requirements()
    os.system("uv remove toml")


if __name__ == "__main__":
    main()

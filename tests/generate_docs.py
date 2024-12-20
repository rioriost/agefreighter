#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pydoc
import os
import sys

SRC_DIR = "../src/agefreighter/"
sys.path.append(os.path.join(os.path.dirname(__file__), SRC_DIR))


def custom_pydoc(module_path: str = "", output_dir: str = ""):
    base_name = os.path.basename(module_path)
    module_name = os.path.splitext(base_name)[0]
    if module_name != "agefreighter":
        module = __import__(module_name)
    else:
        module = __import__("agefreighter.agefreighter", fromlist=["agefreighter"])
    doc = pydoc.plain(pydoc.render_doc(module))

    file_path = os.path.abspath(module.__file__)
    doc = doc.replace("agefreighter.agefreighter", "agefreighter")
    doc = doc.replace("agefreighter.AgeFreighter", "AgeFreighter")
    doc = doc.replace(file_path, base_name)

    with open(f"{output_dir}{module_name}.txt", "w") as f:
        f.write(doc)


def main():
    import glob

    DOCS_DIR = "../docs/"
    files = glob.glob(f"{SRC_DIR}*.py")
    for file in files:
        if "__init__.py" not in file:
            custom_pydoc(module_path=file, output_dir=DOCS_DIR)


if __name__ == "__main__":
    main()

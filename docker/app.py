#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import threading
import time

from agefreighter.view import app

def main():
    flask_port = os.environ.get("FLASK_PORT", 5050)
    app.logger.setLevel(logging.INFO)
    app.run(port=flask_port)

if __name__ == "__main__":
    main()

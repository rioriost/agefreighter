#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.metadata
import subprocess

version=importlib.metadata.version('agefreighter')
print(version)

result=subprocess.run(['docker', 'buildx', 'build', '--no-cache', '--platform', 'linux/amd64,linux/arm64', '-t', 'rioriost/agefreighter-viewer:' + version, '.'], cwd='docker', capture_output=True, text=True)
print(result.stderr)

"""`python -m continualcode` entry point."""

import chz

from .cli import main

chz.nested_entrypoint(main)

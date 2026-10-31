"""Run the bundled progressive distillation trainer."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
runpy.run_module('distillation.train', run_name='__main__')

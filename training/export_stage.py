"""Export a progressive stage to a FastGen checkpoint."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
runpy.run_module('distillation.export', run_name='__main__')

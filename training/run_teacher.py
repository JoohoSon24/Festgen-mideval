"""Run the bundled teacher trainer with this submission's training modules."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
runpy.run_module('teacher.train', run_name='__main__')

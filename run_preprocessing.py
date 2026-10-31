"""Portable entry point; use the approved project Python environment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from nyishi_asr.restore import main

if __name__ == '__main__':
    main()

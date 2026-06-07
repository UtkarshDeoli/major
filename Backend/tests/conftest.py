import sys
import os

# Add the parent directory (Backend/) to the Python path
# so that `import src.xxx` works in tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

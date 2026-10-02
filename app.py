import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "api"))
from serve_api import main  # noqa: E402

if __name__ == "__main__":
    main()

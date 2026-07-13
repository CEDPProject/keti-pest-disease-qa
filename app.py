"""감귤 병해충 AI 진단 — 웹 데모 진입점.

  python app.py --port 5100            # 실제 모델 (GPU ~16GB) → http://<host>:5100
  python app.py --port 5100 --mock     # 모델 없이 UI 테스트

실제 구현은 web/backend/serve_api.py (FastAPI) 참고.
프론트엔드는 web/frontend/index.html (단일 HTML, 백엔드 없이 열어도 Mock 동작).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "backend"))
from serve_api import main  # noqa: E402

if __name__ == "__main__":
    main()

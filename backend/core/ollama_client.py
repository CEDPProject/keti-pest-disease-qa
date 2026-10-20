import json
import urllib.request

from backend.core.settings import OLLAMA_URL


# Ollama generate API 를 호출해 응답 문장을 돌려준다
def ollama_generate(prompt: str, model: str, options: dict, timeout: int) -> str:
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                         "options": options}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"].strip()

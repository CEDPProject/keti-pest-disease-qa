from PIL import Image, ImageEnhance

AUG_MAX_SIDE = 1024


# 확정된 학명·영명만 괄호 표기로 만든다
def _ident(k: dict) -> str:
    parts = []
    if k["scientific"] and "미확정" not in k["scientific"]:
        parts.append(f"학명 {k['scientific']}")
    if k["english"] and "가칭" not in k["english"]:
        parts.append(f"영명 {k['english']}")
    return f"({' / '.join(parts)})" if parts else ""


# 병명·증상·생태·방제·약제를 담은 종합 진단 답변을 만든다
def build_full(label: str, k: dict) -> str:
    lines = [f"사진 속 감귤에서 관찰되는 증상으로 볼 때 **{label}**(으)로 진단됩니다.", ""]
    if _ident(k):
        lines += [_ident(k), ""]
    lines.append("## 관찰 증상")
    lines += [f"- {s}" for s in k["symptoms"][:3]]
    if k["ecology"]:
        lines += ["", "## 발생 원인 및 생태"] + [f"- {s}" for s in k["ecology"][:2]]
    lines += ["", "## 방제 및 관리"] + [f"- {s}" for s in k["control"][:3]]
    if k["pesticides"]:
        lines += ["", "## 추천 약제(예시)",
                  f"- {', '.join(k['pesticides'][:3])} 등 (정확한 등록 약제·사용 시기는 농약안전정보시스템 확인)"]
    lines += ["", "## 추가 확인",
              "- 이미지만으로 확정 진단은 어렵습니다. 병반의 위치·크기, 발생 시기, 주변 나무 상태를 함께 확인하세요."]
    return "\n".join(lines)


# 병명과 판단 근거 한 줄을 담은 짧은 답변을 만든다
def build_name(label: str, k: dict) -> str:
    reason = k["symptoms"][0] if k["symptoms"] else ""
    out = [f"이 감귤은 **{label}**(으)로 진단됩니다."]
    if _ident(k):
        out.append(_ident(k))
    if reason:
        out += ["", f"판단 근거: {reason}"]
    out += ["", "정확한 확진과 방제 방법은 증상 부위·발생 시기를 함께 확인해 주세요."]
    return "\n".join(out)


# 방제 방법과 약제 중심의 답변을 만든다
def build_control(label: str, k: dict) -> str:
    lines = [f"**{label}**의 방제 및 관리 방법입니다.", "", "## 방제 및 관리"]
    lines += [f"- {s}" for s in k["control"][:3]]
    if k["pesticides"]:
        lines += ["", "## 추천 약제(예시)",
                  f"- {', '.join(k['pesticides'][:3])} 등 (정확한 등록 약제·사용 시기는 농약안전정보시스템 확인)"]
    lines += ["", "약제는 등록 정보를 확인하고, 발생 초기에 예방적으로 관리하는 것이 효과적입니다."]
    return "\n".join(lines)


# 관찰 증상 중심의 답변을 만든다
def build_symptom(label: str, k: dict) -> str:
    lines = [f"이 감귤에서는 **{label}**의 증상이 관찰됩니다.", "", "## 관찰 증상"]
    lines += [f"- {s}" for s in k["symptoms"][:4]]
    lines += ["", "동일 증상이라도 병 확진에는 발생 부위·시기 확인이 필요합니다."]
    return "\n".join(lines)


# 무작위 변형으로 증강 이미지를 만든다
def augment(img: Image.Image, rng) -> Image.Image:
    im = img
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.rotate(rng.uniform(-12, 12), resample=Image.BILINEAR, fillcolor=(255, 255, 255))
    w, h = im.size
    scale = rng.uniform(0.85, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    im = im.crop((x0, y0, x0 + cw, y0 + ch))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.85, 1.15))
    w, h = im.size
    if max(w, h) > AUG_MAX_SIDE:
        r = AUG_MAX_SIDE / max(w, h)
        im = im.resize((int(w * r), int(h * r)), Image.LANCZOS)
    return im


# 이미지·질문·답변을 LLaMA-Factory sharegpt 학습 샘플로 묶는다
def make_sample(image_path: str, question: str, answer: str) -> dict:
    return {"messages": [{"role": "user", "content": "<image>" + question},
                         {"role": "assistant", "content": answer}], "images": [image_path]}

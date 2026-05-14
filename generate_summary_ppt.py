"""감귤 병해충 Q&A 시스템 — 요약 미팅 PPT (예시 슬라이드 포함)
ppt_promp.md 피드백 반영 + 실행 예시 3장 추가
"""

from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os, json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(PROJECT_DIR, "result_md", "감귤_병해충_QA_요약보고.pptx")
EXAMPLES_PATH = os.path.join(PROJECT_DIR, "result_md", "ppt_examples.json")
CM_CLASSIFIER = os.path.join(PROJECT_DIR, "result_md", "classifier_confusion_matrix.png")
CM_VLM = os.path.join(PROJECT_DIR, "result_md", "vlm_confusion_matrix.png")
TEST_IMAGES_DIR = os.path.join(PROJECT_DIR, "image_qa", "test_images")

# ── 색상 ──
C_TITLE = RGBColor(0x1B, 0x2A, 0x3D)
C_ACCENT = RGBColor(0x0D, 0x73, 0x77)
C_POINT = RGBColor(0xFF, 0x8C, 0x42)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_TEXT = RGBColor(0x2C, 0x3E, 0x50)
C_GRAY = RGBColor(0x66, 0x66, 0x66)
C_LIGHT = RGBColor(0xF5, 0xF5, 0xF5)
C_GREEN = RGBColor(0xE8, 0xF5, 0xE9)
C_BLUE = RGBColor(0xE8, 0xF0, 0xFE)
C_PURPLE = RGBColor(0xE8, 0xE0, 0xFE)
C_WARN = RGBColor(0xFD, 0xF0, 0xE0)
C_RED_BG = RGBColor(0xFD, 0xE8, 0xE8)

SW = 12191695; SH = 6858000

def new_prs():
    p = Presentation(); p.slide_width = SW; p.slide_height = SH; return p

def S(p): return p.slides.add_slide(p.slide_layouts[6])

def bg(s, c): s.background.fill.solid(); s.background.fill.fore_color.rgb = c

def R(s, l, t, w, h, c, border=False):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = c
    if not border: sh.line.fill.background()
    else: sh.line.color.rgb = RGBColor(0xDD,0xDD,0xDD); sh.line.width = Pt(1)
    return sh

def RR(s, l, t, w, h, c, border=False):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = c
    if not border: sh.line.fill.background()
    else: sh.line.color.rgb = RGBColor(0xDD,0xDD,0xDD); sh.line.width = Pt(1)
    return sh

def T(s, l, t, w, h, text, sz=14, bold=False, c=C_TEXT, al=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(l, t, w, h); tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text; p.font.size = Pt(sz)
    p.font.bold = bold; p.font.color.rgb = c; p.alignment = al
    return tf

def M(s, l, t, w, h, lines, sz=14, c=C_TEXT, bold=False, sp=None):
    tb = s.shapes.add_textbox(l, t, w, h); tf = tb.text_frame; tf.word_wrap = True
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln; p.font.size = Pt(sz); p.font.color.rgb = c
        p.font.bold = bold; p.space_after = Pt(sp if sp else sz * 0.4)
    return tf

def hdr(s):
    R(s, 0, 0, SW, Emu(54864), C_TITLE)
    R(s, 0, Emu(54864), SW, Emu(27432), C_ACCENT)

def stitle(s, title):
    hdr(s); bg(s, C_WHITE)
    T(s, Emu(457200), Emu(180000), Emu(10500000), Emu(500000), title, sz=28, bold=True, c=C_TITLE)
    R(s, Emu(457200), Emu(640000), Emu(10500000), Emu(18000), C_ACCENT)

def tbl(s, l, t, w, headers, rows, hc=C_TITLE, rh=Emu(360000), green_cols=None):
    nr = len(rows) + 1; nc = len(headers)
    ts = s.shapes.add_table(nr, nc, l, t, w, rh * nr); tb = ts.table
    for j, h in enumerate(headers):
        cl = tb.cell(0, j); cl.text = h; cl.fill.solid(); cl.fill.fore_color.rgb = hc
        for p in cl.text_frame.paragraphs:
            p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = C_WHITE; p.alignment = PP_ALIGN.CENTER
        cl.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cl = tb.cell(i+1, j); cl.text = str(val)
            if green_cols and j in green_cols:
                cl.fill.solid(); cl.fill.fore_color.rgb = C_GREEN
            elif i % 2 == 0:
                cl.fill.solid(); cl.fill.fore_color.rgb = RGBColor(0xF8,0xF8,0xF8)
            for p in cl.text_frame.paragraphs:
                p.font.size = Pt(11); p.font.color.rgb = C_TEXT; p.alignment = PP_ALIGN.CENTER
                if green_cols and j in green_cols: p.font.bold = True
            cl.vertical_anchor = MSO_ANCHOR.MIDDLE
    return tb


def build():
    prs = new_prs()
    gap = Emu(200000)

    # ══════ 1. 표지 ══════
    s = S(prs); bg(s, C_TITLE)
    R(s, 0, 0, SW, Emu(54864), C_ACCENT)
    T(s, Emu(914400), Emu(1600000), Emu(10058400), Emu(1200000),
      "감귤 병해충 질의응답 시스템", sz=36, bold=True, c=C_WHITE)
    T(s, Emu(914400), Emu(2800000), Emu(10058400), Emu(600000),
      "텍스트 RAG / 이미지 분류 RAG / VLM 기반 접근법 비교",
      sz=18, bold=True, c=RGBColor(0xB0,0xDB,0xD5))
    R(s, Emu(914400), Emu(3500000), Emu(10058400), Emu(27432), C_ACCENT)
    T(s, Emu(914400), Emu(5200000), Emu(10058400), Emu(400000),
      "2026. 05.", sz=16, c=RGBColor(0x80,0xCB,0xC4))

    # ══════ 2. 3가지 방법 개요 ══════
    s = S(prs); stitle(s, "지금 하려는 것 — 3가지 방법 개요")
    cards = [
        ("💬", "방법 1: 텍스트 RAG",
         ["증상 텍스트 입력", "→ RAG 후보 3개 제시", "→ 반복 질문으로 범위 좁히기", "→ 병명 확정 → 상세 정보"],
         ["이미지 없이 텍스트만으로 진단", "대화형 반복으로 정확도 향상", "RAG 기반 공식 정보 제공"],
         ["사용자 설명 품질에 좌우", "모호한 증상 시 후보 많아짐"],
         C_BLUE, RGBColor(0x1A,0x73,0xE8)),
        ("🖼️", "방법 2: 분류 + RAG",
         ["감귤 이미지 입력", "→ ViT 분류기 → 병명 라벨", "→ 라벨로 RAG 검색", "→ LLM이 상세 답변 생성"],
         ["빠른 추론 (0.09초/장)", "분류 라벨 → RAG 검색 정확"],
         ["분류기 오류 시 전체 답변 오류", "3-class 모델 → 클래스 확장 불가", "\"왜 이 병인가\" 시각 근거 없음"],
         C_GREEN, RGBColor(0x2E,0x7D,0x32)),
        ("🧠", "방법 3: VLM + RAG",
         ["이미지 + 질문 동시 입력", "→ VLM이 이미지 관찰 + 병명 추정", "→ 병명 키워드로 RAG 검색", "→ VLM 해석 + RAG 통합 답변"],
         ["시각적 관찰 근거 설명 가능", "RAG으로 공식 지식 보강", "새 클래스 추가 시 재학습 불필요"],
         ["GPU 필요 (6~12초/장)", "zero-shot 시 과잉 진단 경향"],
         C_PURPLE, RGBColor(0x6A,0x1B,0x9A)),
    ]
    cw = Emu(3500000); ch = Emu(5000000); y0 = Emu(850000)
    for i, (icon, title, flow, pros, cons, bgc, ac) in enumerate(cards):
        x = Emu(457200) + i * (cw + gap)
        RR(s, x, y0, cw, ch, bgc, border=True)
        R(s, x, y0, cw, Emu(60000), ac)
        T(s, x+Emu(120000), y0+Emu(100000), cw-Emu(240000), Emu(350000),
          f"{icon}  {title}", sz=15, bold=True, c=C_TITLE)
        M(s, x+Emu(120000), y0+Emu(500000), cw-Emu(240000), Emu(1100000), flow, sz=11, c=C_TEXT, sp=5)
        R(s, x+Emu(100000), y0+Emu(1700000), cw-Emu(200000), Emu(12000), C_LIGHT)
        T(s, x+Emu(120000), y0+Emu(1750000), cw-Emu(240000), Emu(250000), "✅ 장점", sz=11, bold=True, c=RGBColor(0x2E,0x7D,0x32))
        M(s, x+Emu(120000), y0+Emu(2020000), cw-Emu(240000), Emu(900000), [f"• {p}" for p in pros], sz=10, c=C_TEXT, sp=3)
        cy = y0+Emu(2900000) if len(pros)<=2 else y0+Emu(3050000)
        T(s, x+Emu(120000), cy, cw-Emu(240000), Emu(250000), "⚠ 단점", sz=11, bold=True, c=RGBColor(0x8B,0x4C,0x00))
        M(s, x+Emu(120000), cy+Emu(270000), cw-Emu(240000), Emu(900000), [f"• {c}" for c in cons], sz=10, c=RGBColor(0x8B,0x4C,0x00), sp=3)
    R(s, Emu(457200), Emu(6050000), Emu(10800000), Emu(400000), C_LIGHT)
    T(s, Emu(557200), Emu(6100000), Emu(10600000), Emu(300000),
      "공통: RAG 지식베이스 (ChromaDB 35문서, citrus_PD_ver3.md) + LLM gemma3:12B (Ollama) + 임베딩 E5-large", sz=12, c=C_TEXT)

    # ══════ 3. 기존 기술 대비 ══════
    s = S(prs); stitle(s, "기존 기술 대비 — 왜 이 접근이 의미 있는가")
    gens = [
        ("1세대", "분류 단독", "라벨만 출력\n방제 정보 없음", "Mohanty 2016\nPlantVillage\n38클래스 CNN", C_ACCENT),
        ("2세대", "분류 + LLM", "LLM이 근거 없이\n답변 생성 (환각)", "Qing 2023\nYOLO + GPT-4", C_ACCENT),
        ("3세대", "분류 + RAG", "이미지를 직접 보지\n못함 → 시각 근거 X", "FarmTalk 2024\nYOLOv8+GPT-3.5\n+RAG", C_ACCENT),
        ("4세대", "VLM 단독", "학습 데이터에 없는\n정보는 환각 발생", "Agri-LLaVA 2024\n농업 VLM\n221종", C_POINT),
    ]
    gw = Emu(2500000); gh = Emu(1500000); gy = Emu(820000)
    for i, (gen, name, limit, ref, gc) in enumerate(gens):
        gx = Emu(350000+i*2850000)
        RR(s, gx, gy, gw, gh, C_LIGHT, border=True)
        R(s, gx, gy, gw, Emu(45000), gc)
        T(s, gx+Emu(80000), gy+Emu(80000), gw-Emu(160000), Emu(280000), f"{gen}: {name}", sz=13, bold=True, c=C_TITLE)
        M(s, gx+Emu(80000), gy+Emu(400000), gw-Emu(160000), Emu(450000),
          [f"한계: {l}" if j==0 else l for j, l in enumerate(limit.split("\n"))], sz=10, c=RGBColor(0xC6,0x28,0x28), sp=2)
        M(s, gx+Emu(80000), gy+Emu(900000), gw-Emu(160000), Emu(500000), ref.split("\n"), sz=9, c=C_GRAY, sp=1)
        if i < 3:
            T(s, gx+gw+Emu(30000), gy+Emu(450000), Emu(300000), Emu(300000), "→", sz=22, bold=True, c=C_ACCENT, al=PP_ALIGN.CENTER)
    tbl(s, Emu(350000), Emu(2550000), Emu(11200000),
        ["구분", "기존 분류+RAG\n(3세대)", "기존 VLM 단독\n(4세대)", "본 방법 3\n(VLM+RAG)"],
        [["시각적 근거 설명","X","O","O ✅"], ["공식 지식 기반","O","X","O ✅"],
         ["Hallucination 억제","O","X","O ✅"], ["감귤 특화","X","X","O ✅"]],
        rh=Emu(340000), green_cols=[3])
    R(s, Emu(350000), Emu(4400000), Emu(11200000), Emu(800000), C_GREEN)
    M(s, Emu(500000), Emu(4440000), Emu(11000000), Emu(700000), [
        "핵심: VLM + RAG 결합은 농업 병해충 분야에서 아직 시도되지 않은 접근",
        "• VLM이 이미지에서 증상을 관찰하고, RAG이 공식 약제·방제 정보를 보강",
        "• 기존 3세대(분류+RAG)는 \"왜 이 병인가\"를 설명 못함 → VLM이 시각 근거 제공",
        "• 기존 4세대(VLM 단독)는 학습 데이터 밖의 약제명/방제시기를 지어냄 → RAG이 해결",
    ], sz=12, c=RGBColor(0x2E,0x7D,0x32), sp=4)

    # ══════ 4. 실험 결과 요약 ══════
    s = S(prs); stitle(s, "실험 결과 요약")
    results_data = [
        ("방법 1\n텍스트 RAG", "파이프라인 성공", "5/5",
         ["RAG 검색 → LLM 답변 정상 동작", "학명·약제·방제시기 정보 제공", "", "※ 정량 지표(Top-3 Hit Rate)", "   평가 체계 구축 후 측정 예정"],
         "분석: 텍스트 기반이라 이미지 없이도\n동작하나, 사용자 설명이 모호하면\n후보가 넓어지는 한계 존재",
         C_BLUE, RGBColor(0x1A,0x73,0xE8)),
        ("방법 2\n이미지 분류 + RAG", "Accuracy", "71.0%",
         ["Macro F1: 0.49", "정상/궤양 2클래스 · 321장", "", "Canker P=0.96 / R=0.66", "Healthy P=0.58 / R=0.85"],
         "원인: 3rd-party 오픈 모델이 감귤\n특화 아님 (오렌지 학습 모델).\n궤양병 1/3을 melanose로 오분류",
         C_GREEN, RGBColor(0x2E,0x7D,0x32)),
        ("방법 3\nVLM + RAG", "Accuracy", "58.7%",
         ["JSON 파싱: 93.5%", "Macro F1: 0.25", "4클래스 zero-shot · 46장", "", "⚠ 응애/진딧물 각 1장 → 0.00"],
         "원인: ① zero-shot(학습 없이 추론)\n② 정상→궤양 과잉 진단 경향\n③ 응애·진딧물 데이터 극소 (각 1장)",
         C_PURPLE, RGBColor(0x6A,0x1B,0x9A)),
    ]
    rw = Emu(3500000); rh2 = Emu(5300000); ry = Emu(850000)
    for i, (title, mlabel, mval, subs, analysis, bgc, ac) in enumerate(results_data):
        x = Emu(457200) + i * (rw + gap)
        RR(s, x, ry, rw, rh2, bgc, border=True)
        R(s, x, ry, rw, Emu(60000), ac)
        M(s, x+Emu(120000), ry+Emu(100000), rw-Emu(240000), Emu(450000), title.split("\n"), sz=14, c=C_TITLE, bold=True, sp=4)
        T(s, x+Emu(120000), ry+Emu(600000), rw-Emu(240000), Emu(250000), mlabel, sz=11, c=C_GRAY)
        T(s, x+Emu(120000), ry+Emu(850000), rw-Emu(240000), Emu(600000), mval, sz=42, bold=True, c=ac, al=PP_ALIGN.CENTER)
        R(s, x+Emu(200000), ry+Emu(1550000), rw-Emu(400000), Emu(10000), C_LIGHT)
        M(s, x+Emu(120000), ry+Emu(1620000), rw-Emu(240000), Emu(1200000), subs, sz=10, c=C_TEXT, sp=4)
        R(s, x+Emu(200000), ry+Emu(3000000), rw-Emu(400000), Emu(10000), C_LIGHT)
        R(s, x+Emu(60000), ry+Emu(3100000), rw-Emu(120000), Emu(1000000), C_WARN)
        M(s, x+Emu(160000), ry+Emu(3150000), rw-Emu(320000), Emu(900000), analysis.split("\n"), sz=10, c=RGBColor(0x8B,0x4C,0x00), sp=3)

    # ══════ 5~7. 실행 예시 슬라이드 (방법 1, 2, 3) ══════
    examples = {}
    if os.path.exists(EXAMPLES_PATH):
        with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
            examples = json.load(f)

    # ── 5. 방법 1 실행 예시 ──
    s = S(prs); stitle(s, "실행 예시 — 방법 1: 텍스트 RAG")
    m1 = examples.get("method1", [])
    ey = Emu(850000)
    for idx, ex in enumerate(m1):
        # 입력 박스
        R(s, Emu(350000), ey, Emu(3800000), Emu(1500000), C_BLUE)
        T(s, Emu(450000), ey+Emu(30000), Emu(3600000), Emu(250000),
          f"입력 질문 {idx+1}", sz=11, bold=True, c=C_TITLE)
        T(s, Emu(450000), ey+Emu(300000), Emu(3600000), Emu(400000),
          f'"{ex["input"]}"', sz=14, bold=True, c=RGBColor(0x1A,0x73,0xE8))
        T(s, Emu(450000), ey+Emu(750000), Emu(3600000), Emu(250000),
          f'⏱ {ex["time_sec"]}초  |  📄 {ex["full_length"]}자  |  📚 RAG {ex["sources_count"]}문서',
          sz=10, c=C_GRAY)
        # 응답 박스
        R(s, Emu(4300000), ey, Emu(7400000), Emu(1500000), C_LIGHT)
        T(s, Emu(4400000), ey+Emu(30000), Emu(7200000), Emu(250000),
          "시스템 응답 (일부)", sz=11, bold=True, c=C_TITLE)
        # 답변 미리보기 (150자로 제한)
        preview = ex["output_preview"][:180]
        if len(ex["output_preview"]) > 180:
            preview += "..."
        M(s, Emu(4400000), ey+Emu(300000), Emu(7200000), Emu(1100000),
          preview.split("  ")[:5], sz=9, c=C_TEXT, sp=2)
        ey += Emu(1650000)

    # 하단 요약
    R(s, Emu(350000), Emu(5850000), Emu(11200000), Emu(400000), C_GREEN)
    T(s, Emu(500000), Emu(5900000), Emu(11000000), Emu(300000),
      f"✅ 3/3 성공  |  평균 응답 시간: {sum(e['time_sec'] for e in m1)/len(m1):.1f}초  |  RAG 기반 학명·약제·방제시기 포함 답변",
      sz=13, bold=True, c=RGBColor(0x2E,0x7D,0x32))

    # ── 6. 방법 2 실행 예시 ──
    s = S(prs); stitle(s, "실행 예시 — 방법 2: 이미지 분류 + RAG")
    m2 = examples.get("method2", [])
    # 테이블로 표시
    tbl(s, Emu(350000), Emu(900000), Emu(11200000),
        ["입력 이미지", "정답", "예측", "신뢰도", "정확", "RAG 호출", "시간"],
        [[e["input"], e["gt_label"], e["predicted_label"],
          f'{e["confidence"]*100:.1f}%', "✅" if e.get("correct") else "❌",
          "O" if e.get("rag_used") else "X", f'{e["time_sec"]}초'] for e in m2],
        rh=Emu(400000))

    # 테스트 이미지 썸네일 삽입
    img_y = Emu(2550000)
    img_w = Emu(1600000); img_h = Emu(1200000)
    for idx, ex in enumerate(m2):
        img_path = os.path.join(TEST_IMAGES_DIR, ex["input"])
        ix = Emu(350000) + idx * Emu(3800000)
        if os.path.exists(img_path):
            # 이미지 배경 + 테두리
            RR(s, ix, img_y, Emu(3600000), Emu(1600000), C_LIGHT, border=True)
            s.shapes.add_picture(img_path, ix + Emu(100000), img_y + Emu(50000), img_w, img_h)
            # 이미지 오른쪽에 라벨 정보
            lbl_x = ix + img_w + Emu(200000)
            T(s, lbl_x, img_y + Emu(80000), Emu(1600000), Emu(250000),
              ex["input"].replace(".jpg", ""), sz=11, bold=True, c=C_TITLE)
            T(s, lbl_x, img_y + Emu(380000), Emu(1600000), Emu(250000),
              f'정답: {ex["gt_label"]}', sz=10, c=C_GRAY)
            pred_c = RGBColor(0x2E,0x7D,0x32) if ex.get("correct") else RGBColor(0xC6,0x28,0x28)
            T(s, lbl_x, img_y + Emu(630000), Emu(1600000), Emu(250000),
              f'예측: {ex["predicted_label"]} ({ex["confidence"]*100:.0f}%)', sz=10, bold=True, c=pred_c)
            result_mark = "✅ 정답" if ex.get("correct") else "❌ 오답"
            T(s, lbl_x, img_y + Emu(880000), Emu(1600000), Emu(250000),
              result_mark, sz=12, bold=True, c=pred_c)

    # 핵심 발견
    correct_cnt = sum(1 for e in m2 if e.get("correct"))
    ky = Emu(4350000)
    R(s, Emu(350000), ky, Emu(11200000), Emu(1600000), C_LIGHT)
    T(s, Emu(450000), ky + Emu(50000), Emu(11000000), Emu(300000),
      "핵심 발견", sz=14, bold=True, c=C_TITLE)
    M(s, Emu(450000), ky + Emu(350000), Emu(5500000), Emu(1200000), [
        f"• 테스트 정확도: {correct_cnt}/{len(m2)} ({correct_cnt/len(m2)*100:.0f}%)",
        f"• 평균 추론 시간: {sum(e['time_sec'] for e in m2)/len(m2):.2f}초/장",
        "• 궤양병 이미지를 '건강'으로 오분류 (2/3)",
        "  → 오렌지 학습 모델의 감귤 특화 부재",
        "• 분류 오류 시 RAG 미호출 → 전체 파이프라인 무의미",
    ], sz=10, c=C_TEXT, sp=3)

    # Confusion Matrix 삽입
    if os.path.exists(CM_CLASSIFIER):
        s.shapes.add_picture(CM_CLASSIFIER, Emu(6500000), ky + Emu(200000), Emu(4500000), Emu(1300000))

    R(s, Emu(350000), Emu(6100000), Emu(11200000), Emu(400000), C_WARN)
    T(s, Emu(500000), Emu(6140000), Emu(11000000), Emu(300000),
      "⚠ 3rd-party 오픈 모델(오렌지 학습)을 감귤에 적용한 한계 — 감귤 특화 파인튜닝 필요",
      sz=12, bold=True, c=RGBColor(0x8B,0x4C,0x00))

    # ── 7. 방법 3 실행 예시 ──
    s = S(prs); stitle(s, "실행 예시 — 방법 3: VLM + RAG")
    m3 = examples.get("method3", [])
    ey = Emu(850000)
    img_sz = Emu(1200000)  # 이미지 크기
    for idx, ex in enumerate(m3):
        card_h = Emu(1500000)
        img_path = os.path.join(TEST_IMAGES_DIR, ex["input_image"])
        # 이미지 썸네일 (왼쪽)
        if os.path.exists(img_path):
            RR(s, Emu(350000), ey, img_sz + Emu(100000), card_h, C_LIGHT, border=True)
            s.shapes.add_picture(img_path, Emu(400000), ey + Emu(50000), img_sz, img_sz)
            T(s, Emu(380000), ey + img_sz + Emu(80000), img_sz + Emu(60000), Emu(200000),
              ex["input_image"].replace(".jpg", ""), sz=9, bold=True, c=C_GRAY, al=PP_ALIGN.CENTER)
        # 입력 정보 (중앙)
        info_x = Emu(1650000)
        info_w = Emu(2700000)
        R(s, info_x, ey, info_w, card_h, C_PURPLE)
        T(s, info_x + Emu(100000), ey + Emu(30000), info_w - Emu(200000), Emu(250000),
          f"예시 {idx+1}", sz=11, bold=True, c=C_TITLE)
        T(s, info_x + Emu(100000), ey + Emu(300000), info_w - Emu(200000), Emu(400000),
          f'질문: "{ex["input_question"]}"', sz=12, bold=True, c=RGBColor(0x6A,0x1B,0x9A))
        T(s, info_x + Emu(100000), ey + Emu(750000), info_w - Emu(200000), Emu(250000),
          f'정답: {ex["gt_label"]}', sz=10, c=C_GRAY)
        T(s, info_x + Emu(100000), ey + Emu(1000000), info_w - Emu(200000), Emu(250000),
          f'⏱ {ex["time_sec"]}초', sz=10, c=C_GRAY)
        # 응답 박스 (오른쪽)
        resp_x = Emu(4500000)
        resp_w = Emu(7200000)
        R(s, resp_x, ey, resp_w, card_h, C_LIGHT)
        T(s, resp_x + Emu(100000), ey + Emu(30000), resp_w - Emu(200000), Emu(250000),
          "VLM 응답 (일부)", sz=11, bold=True, c=C_TITLE)
        preview = ex["answer_preview"][:220]
        if len(ex["answer_preview"]) > 220: preview += "..."
        M(s, resp_x + Emu(100000), ey + Emu(300000), resp_w - Emu(200000), Emu(1100000),
          [ln.strip() for ln in preview.split("  ") if ln.strip()][:6], sz=9, c=C_TEXT, sp=2)
        ey += Emu(1650000)

    # 하단 요약
    R(s, Emu(350000), Emu(5850000), Emu(11200000), Emu(500000), C_GREEN)
    M(s, Emu(500000), Emu(5880000), Emu(11000000), Emu(420000), [
        f"평균 응답 시간: {sum(e['time_sec'] for e in m3)/len(m3):.1f}초  |  이미지 관찰 근거 + 증상 설명 + 관리 방법을 자연어로 제공",
        "VLM이 이미지를 직접 보고 관찰 근거를 설명하는 것이 방법 2(분류+RAG) 대비 핵심 차별점",
    ], sz=12, c=RGBColor(0x2E,0x7D,0x32), bold=True, sp=3)

    # ══════ 8. 한계점 + 필요 데이터 + 시계열 ══════
    s = S(prs); stitle(s, "현재 한계점 및 데이터 확보 방향")
    R(s, Emu(350000), Emu(850000), Emu(5300000), Emu(2400000), C_RED_BG)
    T(s, Emu(450000), Emu(900000), Emu(5100000), Emu(300000),
      "⚠ 현재 오픈 모델 · 오픈 데이터 사용의 한계", sz=14, bold=True, c=RGBColor(0xC6,0x28,0x28))
    M(s, Emu(450000), Emu(1250000), Emu(5100000), Emu(1800000), [
        "• 분류 모델: 오렌지 학습 모델을 감귤에 적용",
        "  → 감귤 특유의 병징 패턴 학습 부재 (궤양 1/3 오분류)",
        "• VLM: 범용 모델의 zero-shot 추론",
        "  → 감귤 병해충 특화 학습 없음 (58.7%)",
        "• 데이터셋: Mendeley 오픈 데이터셋",
        "  → 클래스 불균형 심각 (1~241장), 한국 감귤 아님",
        "• 지식베이스: 공개 자료 기반 수작업 구축",
        "  → 전문가 검증 부재",
    ], sz=11, c=C_TEXT, sp=3)
    R(s, Emu(5850000), Emu(850000), Emu(5700000), Emu(1600000), C_GREEN)
    T(s, Emu(5950000), Emu(900000), Emu(5500000), Emu(300000),
      "📋 향후 확보 필요 데이터", sz=14, bold=True, c=RGBColor(0x2E,0x7D,0x32))
    M(s, Emu(5950000), Emu(1230000), Emu(5500000), Emu(1100000), [
        "• 감귤 병해충 이미지: 클래스당 최소 200장 이상",
        "• 한국 감귤 촬영 데이터 (제주 과수원 등)",
        "• 전문가 레이블링 + 검증된 병해충 지식",
        "• 분류기 파인튜닝용 학습/검증/테스트 분할",
    ], sz=11, c=C_TEXT, sp=3)

    # 시계열
    R(s, Emu(350000), Emu(3500000), Emu(11200000), Emu(2800000), C_BLUE)
    R(s, Emu(350000), Emu(3500000), Emu(11200000), Emu(50000), C_ACCENT)
    T(s, Emu(450000), Emu(3600000), Emu(11000000), Emu(300000),
      "📊 시계열 데이터 활용 방안 — 환경 정보 기반 병해충 예측", sz=14, bold=True, c=C_TITLE)
    cols = [
        ("수집 대상 데이터", ["• 발병 시점의 기상 데이터", "  (온도, 습도, 강수량, 일조)", "• 병해충 발생 날짜·시간", "• 병해충 종류 레이블", "• 과수원 위치·환경 메타정보"], C_ACCENT),
        ("활용 방법", ["• 환경 조건 → 병해충 발생", "  상관관계 분석", "• 시계열 패턴 학습으로", "  발생 시기 예측 모델 구축", "• RAG 지식에 환경 조건 추가"], RGBColor(0x6A,0x1B,0x9A)),
        ("기대 효과", ["• \"다음 주 궤양병 발생 위험\"", "  사전 경고 시스템", "• 이미지 진단 + 환경 정보", "  결합으로 진단 정확도 향상", "• 예방적 방제 시기 추천"], RGBColor(0x2E,0x7D,0x32)),
    ]
    coly = Emu(4000000)
    for i, (title, lines, ac) in enumerate(cols):
        cx = Emu(450000+i*3700000)
        R(s, cx, coly, Emu(3500000), Emu(350000), ac)
        T(s, cx+Emu(80000), coly+Emu(60000), Emu(3340000), Emu(250000), title, sz=12, bold=True, c=C_WHITE)
        M(s, cx+Emu(80000), coly+Emu(420000), Emu(3340000), Emu(1500000), lines, sz=10, c=C_TEXT, sp=2)

    # ══════ 9. 향후 계획 ══════
    s = S(prs); stitle(s, "향후 계획")
    steps = [
        ("1", "감귤 특화 데이터셋 확보", "4클래스 균형 데이터 수집 + 전문가 검증 레이블링"),
        ("2", "분류 모델 파인튜닝", "감귤 이미지로 ViT 모델 재학습 → 분류 정확도 개선"),
        ("3", "VLM + RAG 파이프라인 최적화", "VLM 응답 → RAG 자동 보강, 프롬프트 개선 (few-shot)"),
        ("4", "시계열 환경 데이터 연계", "기상·환경 조건 기반 병해충 발생 예측 모델"),
        ("5", "실제 농가 현장 테스트", "현장 촬영 이미지 + 농민 질문으로 실용성 검증"),
    ]
    y = Emu(1050000)
    for num, title, desc in steps:
        RR(s, Emu(457200), y, Emu(600000), Emu(700000), C_ACCENT)
        T(s, Emu(457200), y+Emu(150000), Emu(600000), Emu(400000), num, sz=28, bold=True, c=C_WHITE, al=PP_ALIGN.CENTER)
        T(s, Emu(1200000), y+Emu(80000), Emu(9500000), Emu(350000), title, sz=18, bold=True, c=C_TITLE)
        T(s, Emu(1200000), y+Emu(420000), Emu(9500000), Emu(280000), desc, sz=13, c=C_GRAY)
        y += Emu(850000)

    # ── Save ──
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    prs.save(OUTPUT_PATH)
    print(f"✅ PPT 저장: {OUTPUT_PATH}")
    print(f"   슬라이드: {len(prs.slides)}장")


if __name__ == "__main__":
    build()

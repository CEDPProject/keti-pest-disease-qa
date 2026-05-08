"""감귤 병해충 Q&A 시스템 — Classifier vs VLM 평가 결과 미팅 PPT
SSL_종합_미팅_PPT.pptx 스타일 기준
"""

from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "result_md", "감귤_병해충_Classifier_VLM_평가결과.pptx")

# ── 색상 ──
C_DARK = RGBColor(0x1B, 0x5E, 0x7B)
C_ACCENT = RGBColor(0x00, 0x89, 0x7B)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_BLACK = RGBColor(0x33, 0x33, 0x33)
C_GRAY = RGBColor(0x66, 0x66, 0x66)
C_BG_BOX = RGBColor(0xF5, 0xF5, 0xF5)
C_BG_WARN = RGBColor(0xFD, 0xF0, 0xE0)
C_BG_GREEN = RGBColor(0xE8, 0xF5, 0xE9)
C_BG_BLUE = RGBColor(0xE8, 0xF0, 0xFE)
C_BG_RED = RGBColor(0xFD, 0xE8, 0xE8)

SLIDE_W = 12191695
SLIDE_H = 6858000

# ── 헬퍼 ──

def new_prs():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs

def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color

def rect(slide, l, t, w, h, c):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = c; s.line.fill.background()
    return s

def txt(slide, l, t, w, h, text, sz=14, bold=False, color=C_BLACK, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text; p.font.size = Pt(sz)
    p.font.bold = bold; p.font.color.rgb = color; p.alignment = align
    return tf

def multi(slide, l, t, w, h, lines, sz=14, color=C_BLACK, bold=False):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.font.size = Pt(sz); p.font.color.rgb = color
        p.font.bold = bold; p.space_after = Pt(sz * 0.3)
    return tf

def header_bars(slide):
    rect(slide, 0, 0, SLIDE_W, Emu(54864), C_DARK)
    rect(slide, 0, Emu(54864), SLIDE_W, Emu(27432), C_ACCENT)

def slide_title(slide, part, title, sub=None):
    header_bars(slide); bg(slide, C_WHITE)
    txt(slide, Emu(457200), Emu(200000), Emu(10500000), Emu(500000),
        f"{part}  {title}", sz=24, bold=True, color=C_DARK)
    if sub:
        txt(slide, Emu(457200), Emu(680000), Emu(10500000), Emu(350000),
            sub, sz=14, color=C_GRAY)

def table(slide, l, t, w, headers, rows, hc=C_DARK, rh=Emu(320000)):
    nr = len(rows) + 1; nc = len(headers)
    ts = slide.shapes.add_table(nr, nc, l, t, w, rh * nr)
    tb = ts.table
    for j, h in enumerate(headers):
        c = tb.cell(0, j); c.text = h; c.fill.solid(); c.fill.fore_color.rgb = hc
        for p in c.text_frame.paragraphs:
            p.font.size = Pt(11); p.font.bold = True
            p.font.color.rgb = C_WHITE; p.alignment = PP_ALIGN.CENTER
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tb.cell(i + 1, j); c.text = str(val)
            if i % 2 == 0:
                c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xF8, 0xF8, 0xF8)
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(11); p.font.color.rgb = C_BLACK
                p.alignment = PP_ALIGN.CENTER
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
    return tb

def info_box(slide, l, t, w, h, text, bgc=C_BG_BOX, sz=12, tc=C_BLACK):
    rect(slide, l, t, w, h, bgc)
    txt(slide, l + Emu(100000), t + Emu(60000), w - Emu(200000), h - Emu(120000),
        text, sz=sz, color=tc)

def info_multi(slide, l, t, w, h, lines, bgc=C_BG_BOX, sz=12, tc=C_BLACK):
    rect(slide, l, t, w, h, bgc)
    multi(slide, l + Emu(100000), t + Emu(60000), w - Emu(200000), h - Emu(120000),
          lines, sz=sz, color=tc)


# ========== 슬라이드 ========== #

def build():
    prs = new_prs()

    # ── 1: 표지 ──
    s = blank(prs); bg(s, C_DARK)
    rect(s, 0, 0, SLIDE_W, Emu(54864), C_ACCENT)
    txt(s, Emu(914400), Emu(1000000), Emu(10058400), Emu(1200000),
        "감귤 병해충 이미지 Q&A 시스템", sz=36, bold=True, color=C_WHITE)
    txt(s, Emu(914400), Emu(2100000), Emu(10058400), Emu(600000),
        "Classifier vs VLM 정량 평가 결과", sz=22, bold=True, color=RGBColor(0xB0, 0xDB, 0xD5))
    rect(s, Emu(914400), Emu(2900000), Emu(10058400), Emu(27432), C_ACCENT)
    multi(s, Emu(914400), Emu(3100000), Emu(10058400), Emu(800000), [
        "경로 A: dima806/orange_fruit_disease_detection (ViT) — Mendeley 321장",
        "경로 B: Qwen2.5-VL-7B (Ollama, qwen2.5vl:7b) — 46장 zero-shot",
    ], sz=14, color=RGBColor(0x90, 0xCA, 0xF9))
    multi(s, Emu(914400), Emu(4100000), Emu(10058400), Emu(600000), [
        "데이터: Mendeley Citrus Dataset (759장) + 로컬 테스트 이미지 (6장)",
        "환경: NVIDIA RTX A6000 · PyTorch 2.10 · Ollama · conda env: cedpllm",
    ], sz=12, color=RGBColor(0x90, 0xCA, 0xF9))
    txt(s, Emu(914400), Emu(5600000), Emu(10058400), Emu(400000),
        "2026. 05.", sz=14, color=RGBColor(0x80, 0xCB, 0xC4))

    # ── 2: 목차 ──
    s = blank(prs); bg(s, C_WHITE); header_bars(s)
    txt(s, Emu(457200), Emu(200000), Emu(4000000), Emu(500000),
        "목차", sz=28, bold=True, color=C_DARK)
    toc = [
        ("Part 1", "테스트 개요", "데이터셋 · 모델 · 평가 방법"),
        ("Part 2", "경로 A: Classifier 결과", "ViT 3-class → Mendeley 321장"),
        ("Part 3", "경로 B: VLM 결과", "Qwen2.5-VL zero-shot 4-class 46장"),
        ("Part 4", "두 방식 비교 분석", "정량 · 정성 · 속도 비교"),
        ("Part 5", "핵심 발견 & 향후 계획", "인사이트 · Next Steps"),
    ]
    y = Emu(1000000)
    for part, title, desc in toc:
        rect(s, Emu(457200), y, Emu(10800000), Emu(600000), C_BG_BOX)
        txt(s, Emu(557200), y + Emu(80000), Emu(1200000), Emu(440000),
            part, sz=13, bold=True, color=C_ACCENT)
        txt(s, Emu(1900000), y + Emu(80000), Emu(3500000), Emu(440000),
            title, sz=14, bold=True, color=C_DARK)
        txt(s, Emu(5500000), y + Emu(80000), Emu(5500000), Emu(440000),
            desc, sz=12, color=C_GRAY)
        y += Emu(680000)

    # ── 3: 테스트 개요 ──
    s = blank(prs)
    slide_title(s, "Part 1", "테스트 개요", "update_qa_prompt.md 기준 · 2가지 경로 평가")

    table(s, Emu(457200), Emu(1200000), Emu(10800000),
        ["항목", "경로 A (Classifier)", "경로 B (VLM)"],
        [
            ["모델", "dima806/orange_fruit_\ndisease_detection (ViT)", "Qwen2.5-VL-7B\n(Ollama: qwen2.5vl:7b)"],
            ["방식", "이미지 분류 파이프라인", "멀티모달 VLM zero-shot"],
            ["데이터", "Mendeley Citrus\n(Healthy+Canker)", "로컬 6장 + Mendeley 40장"],
            ["이미지 수", "321장", "46장"],
            ["분류 클래스", "2-class\n(healthy, canker)", "4-class\n(normal, canker,\nred_mite, aphid)"],
            ["실행 장치", "CPU", "GPU (Ollama)"],
            ["소요 시간", "29초 (0.09초/장)", "279초 (6.1초/장)"],
        ], rh=Emu(350000))

    # ── 4: 데이터셋 ──
    s = blank(prs)
    slide_title(s, "Part 1", "Mendeley Citrus Dataset", "759장 · 6 클래스 · CC BY 4.0")

    table(s, Emu(457200), Emu(1200000), Emu(5400000),
        ["부위", "클래스", "이미지 수"],
        [
            ["Fruits", "Black spot", "19"],
            ["Fruits", "Canker", "78"],
            ["Fruits", "Greening", "16"],
            ["Fruits", "healthy", "22"],
            ["Fruits", "Scab", "15"],
        ], rh=Emu(280000))

    table(s, Emu(6100000), Emu(1200000), Emu(5160000),
        ["부위", "클래스", "이미지 수"],
        [
            ["Leaves", "Black spot", "171"],
            ["Leaves", "canker", "163"],
            ["Leaves", "greening", "204"],
            ["Leaves", "healthy", "58"],
            ["Leaves", "Melanose", "13"],
        ], rh=Emu(280000))

    info_multi(s, Emu(457200), Emu(3500000), Emu(10800000), Emu(700000), [
        "경로 A: Healthy(80장) + Canker(241장) = 321장만 필터 사용",
        "경로 B: 로컬 6장(4클래스) + Mendeley Healthy 20장 + Canker 20장 = 46장",
        "데이터 출처: https://data.mendeley.com/datasets/3f83gxmv57/2",
    ], bgc=C_BG_BOX, sz=12)

    # ── 5: 경로 A 결과 ──
    s = blank(prs)
    slide_title(s, "Part 2", "경로 A: Classifier 전체 결과",
                "dima806/orange_fruit_disease_detection · Mendeley 321장")

    # 핵심 지표 3개 박스
    for i, (label, value, bgc) in enumerate([
        ("Accuracy", "71.0%", C_BG_BLUE),
        ("Macro F1", "0.49", C_BG_BOX),
        ("melanose 오분류", "37장 (11.5%)", C_BG_WARN),
    ]):
        x = Emu(457200 + i * 3700000)
        rect(s, x, Emu(1100000), Emu(3400000), Emu(700000), bgc)
        txt(s, x + Emu(100000), Emu(1130000), Emu(3200000), Emu(300000),
            label, sz=13, bold=True, color=C_GRAY)
        txt(s, x + Emu(100000), Emu(1430000), Emu(3200000), Emu(350000),
            value, sz=22, bold=True, color=C_DARK)

    table(s, Emu(457200), Emu(2050000), Emu(10800000),
        ["클래스", "Precision", "Recall", "F1-Score", "이미지 수"],
        [
            ["healthy", "0.58", "0.85", "0.69", "80장"],
            ["canker", "0.96", "0.66", "0.78", "241장"],
            ["unknown (melanose)", "0.00", "0.00", "0.00", "0장"],
        ])

    info_multi(s, Emu(457200), Emu(3250000), Emu(5200000), Emu(1200000), [
        "강점",
        "• Canker Precision 96%",
        "  → 궤양병이라 판정하면 거의 정확",
        "• Healthy Recall 85%",
        "  → 정상 이미지 대부분 정상 분류",
    ], bgc=C_BG_GREEN, sz=12)

    info_multi(s, Emu(5900000), Emu(3250000), Emu(5350000), Emu(1200000), [
        "한계",
        "• Canker Recall 66%",
        "  → 궤양병 241장 중 81장(34%) 놓침",
        "• 37장을 melanose로 오분류",
        "• 귤응애·진딧물 클래스 없음",
    ], bgc=C_BG_RED, sz=12)

    # ── 6: 경로 A 예측 분포 ──
    s = blank(prs)
    slide_title(s, "Part 2", "경로 A: 예측 분포 분석",
                "321장의 예측 라벨 분포 · 오분류 패턴")

    table(s, Emu(457200), Emu(1200000), Emu(5000000),
        ["예측 라벨", "건수", "비율"],
        [
            ["healthy", "117", "36.4%"],
            ["canker", "167", "52.0%"],
            ["melanose", "37", "11.5%"],
        ])

    info_multi(s, Emu(5900000), Emu(1200000), Emu(5350000), Emu(1400000), [
        "오분류 패턴 분석",
        "",
        "• Canker → healthy (44장): 궤양병을 정상으로 오판",
        "• Canker → melanose (37장): 3번째 클래스로 분산",
        "• Healthy → canker (12장): 정상을 궤양병으로 오판",
        "",
        "→ melanose 클래스가 canker 이미지를 흡수",
        "→ 2-class만 쓸 때도 3-class 모델의 한계 노출",
    ], bgc=C_BG_BOX, sz=12)

    info_box(s, Emu(457200), Emu(3000000), Emu(10800000), Emu(500000),
             "⚠ Classifier는 3-class(healthy/canker/melanose) 고정 → 귤응애, 진딧물 등 해충 분류 구조적 불가",
             bgc=C_BG_WARN, sz=13, tc=RGBColor(0x8B, 0x4C, 0x00))

    # ── 7: 경로 B 결과 ──
    s = blank(prs)
    slide_title(s, "Part 3", "경로 B: VLM 전체 결과",
                "Qwen2.5-VL-7B · 46장 · 4-class zero-shot")

    for i, (label, value, bgc) in enumerate([
        ("Accuracy", "58.7%", C_BG_BLUE),
        ("Macro F1", "0.25", C_BG_BOX),
        ("JSON 파싱 성공", "93.5%", C_BG_GREEN),
    ]):
        x = Emu(457200 + i * 3700000)
        rect(s, x, Emu(1100000), Emu(3400000), Emu(700000), bgc)
        txt(s, x + Emu(100000), Emu(1130000), Emu(3200000), Emu(300000),
            label, sz=13, bold=True, color=C_GRAY)
        txt(s, x + Emu(100000), Emu(1430000), Emu(3200000), Emu(350000),
            value, sz=22, bold=True, color=C_DARK)

    table(s, Emu(457200), Emu(2050000), Emu(10800000),
        ["클래스", "Precision", "Recall", "F1-Score", "이미지 수"],
        [
            ["normal", "0.67", "0.55", "0.60", "22장"],
            ["canker", "0.60", "0.68", "0.64", "22장"],
            ["red_mite", "0.00", "0.00", "0.00", "1장"],
            ["aphid", "0.00", "0.00", "0.00", "1장"],
        ])

    info_multi(s, Emu(457200), Emu(3500000), Emu(5200000), Emu(1100000), [
        "강점",
        "• JSON 구조화 응답 93.5% 성공",
        "• 증상(symptoms) + 방제법(treatment) 함께 제공",
        "• 프롬프트 수정만으로 클래스 확장 가능",
    ], bgc=C_BG_GREEN, sz=12)

    info_multi(s, Emu(5900000), Emu(3500000), Emu(5350000), Emu(1100000), [
        "한계",
        "• Accuracy 58.7% (Classifier 71%보다 낮음)",
        "• normal→canker 과잉 진단 경향 (10장)",
        "• 귤응애/진딧물 zero-shot 인식 실패",
        "• 추론 속도 6.1초/장 (68배 느림)",
    ], bgc=C_BG_RED, sz=12)

    # ── 8: VLM 로컬 이미지 상세 ──
    s = blank(prs)
    slide_title(s, "Part 3", "경로 B: 로컬 이미지 상세 결과",
                "test_images/ 6장 (4클래스) 결과")

    table(s, Emu(457200), Emu(1200000), Emu(10800000),
        ["이미지", "GT", "예측", "정확", "JSON", "Confidence"],
        [
            ["열매_정상.jpg", "normal", "canker", "❌", "✅", "high"],
            ["잎_정상.jpg", "normal", "normal", "✅", "✅", "high"],
            ["열매_궤양병.jpg", "canker", "canker", "✅", "✅", "high"],
            ["잎_궤양병.jpg", "canker", "canker", "✅", "✅", "high"],
            ["잎_귤응애.jpg", "red_mite", "normal", "❌", "✅", "high"],
            ["잎_진딧물.jpg", "aphid", "normal", "❌", "✅", "high"],
        ])

    info_multi(s, Emu(457200), Emu(3500000), Emu(10800000), Emu(900000), [
        "분석:",
        "• 궤양병 이미지 2장 모두 정확 분류 ✅",
        "• 정상(열매) → canker 과잉 진단 ❌ — 표면 질감을 병변으로 오인 가능성",
        "• 귤응애·진딧물 → normal로 분류 — 세부 해충 증상 인식 못함",
    ], bgc=C_BG_BOX, sz=13)

    # ── 9: VLM 예측 분포 ──
    s = blank(prs)
    slide_title(s, "Part 3", "경로 B: 예측 분포 & 프롬프트 설계",
                "4-class JSON 분류 프롬프트 · 예측 분포")

    table(s, Emu(457200), Emu(1200000), Emu(5000000),
        ["예측 라벨", "건수", "비율"],
        [
            ["canker", "25", "54.3%"],
            ["normal", "18", "39.1%"],
            ["unknown", "3", "6.5%"],
            ["red_mite", "0", "0%"],
            ["aphid", "0", "0%"],
        ], rh=Emu(280000))

    info_multi(s, Emu(5900000), Emu(1200000), Emu(5350000), Emu(2000000), [
        "프롬프트 핵심:",
        "",
        '• 시스템: "감귤 병해충 전문가"',
        "• 4-class 분류 + JSON 응답 요청",
        '• {classification, confidence,',
        '   symptoms, treatment}',
        "",
        "파싱 전략:",
        "• 1순위: JSON 블록 추출",
        "• 2순위: 키워드 매칭 fallback",
        "• 실패 시: unknown 처리",
    ], bgc=C_BG_BLUE, sz=12)

    info_box(s, Emu(457200), Emu(3300000), Emu(10800000), Emu(500000),
             "⚠ VLM이 red_mite / aphid를 한 번도 예측하지 않음 — zero-shot으로는 세부 해충 구분 어려움",
             bgc=C_BG_WARN, sz=13, tc=RGBColor(0x8B, 0x4C, 0x00))

    # ── 10: 정량 비교 ──
    s = blank(prs)
    slide_title(s, "Part 4", "두 방식 정량 비교", "Classifier vs VLM · 동일 기준 비교")

    table(s, Emu(457200), Emu(1200000), Emu(10800000),
        ["비교 항목", "경로 A (Classifier)", "경로 B (VLM)", "우세"],
        [
            ["Accuracy", "71.0%", "58.7%", "경로 A ✅"],
            ["Macro F1", "0.49", "0.25", "경로 A ✅"],
            ["Canker Precision", "0.96", "0.60", "경로 A ✅"],
            ["Canker Recall", "0.66", "0.68", "경로 B ✅"],
            ["Healthy Precision", "0.58", "0.67", "경로 B ✅"],
            ["Healthy Recall", "0.85", "0.55", "경로 A ✅"],
            ["추론 속도", "0.09초/장", "6.1초/장", "경로 A ✅"],
            ["분류 클래스 수", "3 (고정)", "4+ (확장)", "경로 B ✅"],
        ], rh=Emu(290000))

    info_box(s, Emu(457200), Emu(3800000), Emu(10800000), Emu(500000),
             "정량 지표: Classifier 우세  |  확장성·유연성: VLM 우세  |  실용: 하이브리드 필요",
             bgc=C_BG_BLUE, sz=14)

    # ── 11: 정성 비교 ──
    s = blank(prs)
    slide_title(s, "Part 4", "두 방식 정성 비교", "증상 설명 · 방제법 · 확장성 · 과잉 진단")

    table(s, Emu(457200), Emu(1200000), Emu(10800000),
        ["항목", "경로 A (Classifier)", "경로 B (VLM)"],
        [
            ["증상 설명", "❌ 없음 (라벨만 출력)", "✅ 관찰 증상 서술"],
            ["방제법 제공", "❌ 없음 (RAG 연동 시 가능)", "✅ 일반적 방제법 제공"],
            ["확장성", "모델 재학습 필요", "프롬프트 수정만으로 확장 ✅"],
            ["신뢰도 표현", "수치 (softmax)", "high / medium / low"],
            ["과잉 진단", "melanose로 분산", "canker로 과잉 판정"],
            ["RAG 연동", "분류 결과 → RAG 가능", "미사용 (향후 결합 가능)"],
            ["대규모 스크리닝", "✅ (0.09초/장)", "❌ (6.1초/장)"],
        ], rh=Emu(300000))

    # ── 12: 핵심 발견 ──
    s = blank(prs)
    slide_title(s, "Part 5", "핵심 발견", "")

    info_multi(s, Emu(457200), Emu(1100000), Emu(10800000), Emu(900000), [
        "① Classifier: Canker Precision 96%로 높지만, Recall 66% → 1/3을 놓침",
        "   3-class 한계로 귤응애·진딧물 분류 불가 · melanose가 canker 흡수",
    ], bgc=C_BG_BLUE, sz=13)

    info_multi(s, Emu(457200), Emu(2200000), Emu(10800000), Emu(900000), [
        "② VLM: JSON 구조화 응답 93.5% 성공 · 증상+방제법 동시 제공",
        "   하지만 Accuracy 58.7% · 정상→궤양병 과잉 진단 · 세부 해충 zero-shot 인식 실패",
    ], bgc=C_BG_BLUE, sz=13)

    info_multi(s, Emu(457200), Emu(3300000), Emu(10800000), Emu(900000), [
        "③ 공통: 궤양병(canker) 인식에는 양쪽 모두 일정 수준 성공",
        "   세부 해충(귤응애, 진딧물) 분류는 두 방식 모두 실패 → 데이터 부족 or 모델 한계",
    ], bgc=C_BG_BOX, sz=13)

    info_multi(s, Emu(457200), Emu(4400000), Emu(10800000), Emu(1000000), [
        "④ 최적 방향: Classifier + VLM + RAG 하이브리드",
        "",
        "   Classifier(빠른 스크리닝) → VLM(증상 서술) → RAG(약제/학명 보강)",
        "   또는 VLM 단독 + RAG 결합 파이프라인",
    ], bgc=C_BG_GREEN, sz=13)

    # ── 13: 향후 계획 ──
    s = blank(prs)
    slide_title(s, "Part 5", "향후 계획", "Next Steps")

    steps = [
        ("1", "VLM + RAG 결합 파이프라인", "VLM 응답에서 병명 키워드 추출 → text_qa RAG로 약제/학명 보강"),
        ("2", "프롬프트 최적화", "과잉 진단 문제 해결 (few-shot 예시 추가, confidence 기준 강화)"),
        ("3", "분류기 교체/파인튜닝", "감귤 전용 다중 클래스 모델 (궤양병+귤응애+진딧물+정상)"),
        ("4", "테스트 데이터 확장", "귤응애/진딧물 이미지 확보 → 4-class 균형 평가"),
        ("5", "VLM 모델 비교", "gemma3:12b vs qwen2.5vl:7b vs InternVL2 동일 조건 비교"),
        ("6", "평가 체계 고도화", "정성 평가 루브릭 설계 (증상 설명 품질, 방제법 적절성)"),
    ]
    y = Emu(1100000)
    for num, title, desc in steps:
        rect(s, Emu(457200), y, Emu(600000), Emu(550000), C_ACCENT)
        txt(s, Emu(457200), y + Emu(100000), Emu(600000), Emu(350000),
            num, sz=20, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        txt(s, Emu(1200000), y + Emu(50000), Emu(9500000), Emu(250000),
            title, sz=15, bold=True, color=C_DARK)
        txt(s, Emu(1200000), y + Emu(300000), Emu(9500000), Emu(250000),
            desc, sz=12, color=C_GRAY)
        y += Emu(650000)

    # ── 14: 산출물 ──
    s = blank(prs)
    slide_title(s, "참고", "테스트 산출물 & 파일 구조", "")

    info_multi(s, Emu(457200), Emu(1100000), Emu(5200000), Emu(3500000), [
        "테스트 스크립트:",
        "• test_classifier.py — 경로 A 실행",
        "• test_vlm_classifier.py — 경로 B 실행",
        "• eval_utils.py — 공통 평가 모듈",
        "",
        "결과 파일:",
        "• classifier_results.csv (321행)",
        "• classifier_confusion_matrix.png",
        "• vlm_classification_results.csv (46행)",
        "• vlm_detail_results.csv (증상/방제)",
        "• vlm_confusion_matrix.png",
        "• update_qa_test_results.md",
    ], bgc=C_BG_BOX, sz=12)

    info_multi(s, Emu(5900000), Emu(1100000), Emu(5350000), Emu(3500000), [
        "재현 정보:",
        "• 환경: conda env cedpllm",
        "• GPU: NVIDIA RTX A6000 (48GB)",
        "• Driver: 535.183.01 (CUDA 12.2)",
        "• PyTorch: 2.10.0+cu128",
        "• transformers: 5.3.0",
        "• Ollama: qwen2.5vl:7b, gemma3:12b",
        "",
        "데이터셋:",
        "• Mendeley 3f83gxmv57/v2",
        "• 759장 · 256×256 · CC BY 4.0",
        "• Healthy + Canker 필터 사용",
    ], bgc=C_BG_BOX, sz=12)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    prs.save(OUTPUT_PATH)
    print(f"✅ PPT 저장 완료: {OUTPUT_PATH}")
    print(f"   슬라이드 수: {len(prs.slides)}")


if __name__ == "__main__":
    build()

"""감귤 병해충 Q&A 시스템 - 중간 발표 PPT 자동 생성
SSL_종합_미팅_PPT.pptx 스타일 기준
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_md", "감귤_병해충_QA_시스템_중간발표.pptx")

# 색상 팔레트 (SSL PPT 기준)
C_DARK = RGBColor(0x1B, 0x5E, 0x7B)       # 진한 청록 (상단 바, 제목)
C_ACCENT = RGBColor(0x00, 0x89, 0x7B)      # 민트 (악센트 바)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_BLACK = RGBColor(0x33, 0x33, 0x33)
C_GRAY = RGBColor(0x66, 0x66, 0x66)
C_LIGHT_GRAY = RGBColor(0xE0, 0xE0, 0xE0)
C_BG_BOX = RGBColor(0xF5, 0xF5, 0xF5)     # 회색 박스 배경
C_BG_WARN = RGBColor(0xFD, 0xF0, 0xE0)    # 주의 박스 배경
C_BG_GREEN = RGBColor(0xE8, 0xF5, 0xE9)   # 성공 박스 배경
C_BG_BLUE = RGBColor(0xE8, 0xF0, 0xFE)    # 파란 박스 배경

SLIDE_W = 12191695  # 와이드스크린
SLIDE_H = 6858000


def new_prs():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def add_blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])  # Blank


def set_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill_color):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape


def add_text(slide, left, top, width, height, text, size=14, bold=False, color=C_BLACK, align=PP_ALIGN.LEFT, font_name=None):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align
    if font_name:
        p.font.name = font_name
    return tf


def add_multiline(slide, left, top, width, height, lines, size=14, color=C_BLACK, line_spacing=1.3, bold=False):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.bold = bold
        p.space_after = Pt(size * 0.3)
    return tf


def add_header_bars(slide):
    """SSL 스타일 상단 바 2줄"""
    add_rect(slide, 0, 0, SLIDE_W, Emu(54864), C_DARK)
    add_rect(slide, 0, Emu(54864), SLIDE_W, Emu(27432), C_ACCENT)


def add_slide_title(slide, part_text, title_text, subtitle_text=None):
    """SSL 스타일 슬라이드 제목"""
    add_header_bars(slide)
    set_bg(slide, C_WHITE)
    
    # Part + Title
    add_text(slide, Emu(457200), Emu(200000), Emu(10500000), Emu(500000),
             f"{part_text}  {title_text}", size=24, bold=True, color=C_DARK)
    
    if subtitle_text:
        add_text(slide, Emu(457200), Emu(680000), Emu(10500000), Emu(350000),
                 subtitle_text, size=14, color=C_GRAY)


def add_table(slide, left, top, width, headers, rows, header_color=C_DARK, row_height=Emu(320000)):
    """SSL 스타일 테이블"""
    num_rows = len(rows) + 1
    num_cols = len(headers)
    height = row_height * num_rows
    
    table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, height)
    table = table_shape.table
    
    # Header
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_color
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = C_WHITE
            p.alignment = PP_ALIGN.CENTER
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # Data
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = str(val)
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xF8, 0xF8, 0xF8)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.color.rgb = C_BLACK
                p.alignment = PP_ALIGN.CENTER
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    return table


def add_info_box(slide, left, top, width, height, text, bg_color=C_BG_BOX, text_size=12, text_color=C_BLACK):
    """SSL 스타일 정보 박스"""
    add_rect(slide, left, top, width, height, bg_color)
    add_text(slide, left + Emu(100000), top + Emu(60000),
             width - Emu(200000), height - Emu(120000),
             text, size=text_size, color=text_color)


def add_info_box_multiline(slide, left, top, width, height, lines, bg_color=C_BG_BOX, text_size=12, text_color=C_BLACK):
    """SSL 스타일 정보 박스 (여러 줄)"""
    add_rect(slide, left, top, width, height, bg_color)
    add_multiline(slide, left + Emu(100000), top + Emu(60000),
                  width - Emu(200000), height - Emu(120000),
                  lines, size=text_size, color=text_color)


# ==========================================================
# 슬라이드 빌드
# ==========================================================

def build_ppt():
    prs = new_prs()
    
    # ========== 슬라이드 1: 표지 ==========
    slide = add_blank_slide(prs)
    set_bg(slide, C_DARK)
    add_rect(slide, 0, 0, SLIDE_W, Emu(54864), C_ACCENT)
    
    add_text(slide, Emu(914400), Emu(1100000), Emu(10058400), Emu(1200000),
             "감귤 병해충 이미지 Q&A 시스템", size=36, bold=True, color=C_WHITE)
    
    add_multiline(slide, Emu(914400), Emu(2400000), Emu(10058400), Emu(800000), [
        "이미지 분류 모델 + RAG  vs  VLM 멀티모달 방식 비교",
        "중간 진행 현황 보고",
    ], size=18, color=RGBColor(0xB0, 0xDB, 0xD5))
    
    add_rect(slide, Emu(914400), Emu(3500000), Emu(10058400), Emu(27432), C_ACCENT)
    
    add_multiline(slide, Emu(914400), Emu(3700000), Emu(10058400), Emu(600000), [
        "이미지 분류: dima806/orange_fruit_disease_detection (ViT)",
        "VLM/LLM: gemma3:12B (Ollama)  |  임베딩: intfloat/multilingual-e5-large",
    ], size=12, color=RGBColor(0x90, 0xCA, 0xF9))
    
    add_multiline(slide, Emu(914400), Emu(4500000), Emu(10058400), Emu(400000), [
        "데이터: 감귤 병해충 35종 정보  |  테스트 이미지 6장",
        "환경: NVIDIA RTX A6000, PyTorch 2.10, conda env: cedpllm",
    ], size=12, color=RGBColor(0x90, 0xCA, 0xF9))
    
    add_text(slide, Emu(914400), Emu(5600000), Emu(10058400), Emu(400000),
             "2026. 05.", size=14, color=RGBColor(0x80, 0xCB, 0xC4))
    
    # ========== 슬라이드 2: 목차 ==========
    slide = add_blank_slide(prs)
    set_bg(slide, C_WHITE)
    add_header_bars(slide)
    
    add_text(slide, Emu(457200), Emu(200000), Emu(4000000), Emu(500000),
             "목차", size=28, bold=True, color=C_DARK)
    add_text(slide, Emu(457200), Emu(650000), Emu(4000000), Emu(300000),
             "Table of Contents", size=12, color=C_GRAY)
    
    toc_items = [
        ("Part 1", "프로젝트 개요", "목적 · 구현 방향 · 사용 데이터"),
        ("Part 2", "시스템 아키텍처", "방식 1 (분류기+RAG) / 방식 2 (VLM)"),
        ("Part 3", "테스트 이미지 & 환경", "6장 테스트 이미지 · 실행 환경"),
        ("Part 4", "방식 1: 분류기 + RAG 결과", "이미지별 분류 결과 · 답변 예시"),
        ("Part 5", "방식 2: VLM 멀티모달 결과", "이미지별 VLM 응답 · 답변 예시"),
        ("Part 6", "두 방식 비교 분석", "정량 · 정성 비교"),
        ("Part 7", "핵심 발견 & 향후 계획", "인사이트 · Next Steps"),
    ]
    
    y = Emu(1100000)
    for part, title, desc in toc_items:
        add_rect(slide, Emu(457200), y, Emu(10800000), Emu(520000), C_BG_BOX)
        add_text(slide, Emu(557200), y + Emu(40000), Emu(1200000), Emu(440000),
                 part, size=13, bold=True, color=C_ACCENT)
        add_text(slide, Emu(1900000), y + Emu(40000), Emu(3500000), Emu(440000),
                 title, size=14, bold=True, color=C_DARK)
        add_text(slide, Emu(5500000), y + Emu(40000), Emu(5500000), Emu(440000),
                 desc, size=12, color=C_GRAY)
        y += Emu(580000)
    
    # ========== 슬라이드 3: Part 1 프로젝트 개요 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 1", "프로젝트 개요", "목적 · 구현 방향 · 핵심 질문")
    
    add_text(slide, Emu(457200), Emu(1100000), Emu(10800000), Emu(350000),
             "프로젝트 목적", size=16, bold=True, color=C_DARK)
    
    add_info_box_multiline(slide, Emu(457200), Emu(1500000), Emu(10800000), Emu(1000000), [
        "• 감귤 이미지를 입력하면 병해충을 자동 진단하는 Q&A 시스템 개발",
        "• 텍스트 RAG + 이미지 분류 + VLM 멀티모달 방식 비교 연구",
        "• 농업 현장에서 비전문가도 즉시 활용 가능한 진단 도구 목표",
    ], bg_color=C_BG_GREEN, text_size=13)
    
    add_text(slide, Emu(457200), Emu(2700000), Emu(10800000), Emu(350000),
             "구현 방향 (2가지 방식 비교)", size=16, bold=True, color=C_DARK)
    
    # 방식 1 박스
    add_info_box_multiline(slide, Emu(457200), Emu(3100000), Emu(5200000), Emu(1600000), [
        "방식 1: 이미지 분류 모델 + RAG",
        "",
        "이미지 → ViT 분류기 → 병명 라벨",
        "→ 건강이면 고정 답변",
        "→ 병해충이면 RAG 검색 → LLM 답변",
    ], bg_color=C_BG_BLUE, text_size=12)
    
    # 방식 2 박스
    add_info_box_multiline(slide, Emu(5900000), Emu(3100000), Emu(5350000), Emu(1600000), [
        "방식 2: VLM 멀티모달",
        "",
        "이미지 + 질문 → VLM(gemma3:12b)",
        "→ 이미지 직접 관찰하여 답변 생성",
        "→ 분류기 없이 자유 서술",
    ], bg_color=C_BG_BLUE, text_size=12)
    
    add_info_box(slide, Emu(457200), Emu(4950000), Emu(10800000), Emu(500000),
                 "핵심 질문:  '이미지 분류기 + RAG' vs 'VLM 직접 분석' 중 어떤 방식이 더 효과적인가?",
                 bg_color=C_BG_WARN, text_size=13, text_color=RGBColor(0x8B, 0x4C, 0x00))
    
    # ========== 슬라이드 4: 사용 데이터 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 1", "사용 데이터 & 모델", "텍스트 지식 · 모델 · 벡터DB")
    
    add_table(slide, Emu(457200), Emu(1200000), Emu(10800000),
        ["분류", "데이터 / 모델", "설명"],
        [
            ["텍스트 지식", "citrus_PD_ver3.md", "감귤 병해충 35종 정보 (증상, 원인, 방제, 약제)"],
            ["벡터DB", "Chroma DB (35개 문서)", "E5 임베딩 기반 벡터 저장소 (RAG 검색용)"],
            ["이미지 분류", "dima806/orange_fruit_disease_detection", "ViT 기반 3-class (궤양병/흑점병/건강)"],
            ["LLM/VLM", "gemma3:12B (Ollama)", "텍스트 LLM + 멀티모달 VLM 모두 지원"],
            ["임베딩", "intfloat/multilingual-e5-large", "다국어 임베딩 (RAG 벡터 검색용)"],
        ])
    
    add_info_box(slide, Emu(457200), Emu(3600000), Emu(10800000), Emu(600000),
                 "지식 데이터 구성: 궤양병, 더뎅이병, 검은점무늬병, 흑점병, 귤응애, 귤녹응애, 진딧물류, 볼록총채벌레 등 35종\n"
                 "각 병해충별 학명, 영명, 형태/병징, 발생생태, 방제법, 약제 정보 포함",
                 bg_color=C_BG_BOX, text_size=12)
    
    # ========== 슬라이드 5: 테스트 이미지 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 3", "테스트 이미지 & 환경", "6장 테스트 이미지 · 실행 환경")
    
    add_table(slide, Emu(457200), Emu(1200000), Emu(10800000),
        ["파일명", "실제 라벨", "부위", "비고"],
        [
            ["열매_궤양병.jpg", "궤양병", "열매", "분류기 모델이 다루는 병해"],
            ["열매_정상.jpg", "정상", "열매", "건강한 열매"],
            ["잎_궤양병.jpg", "궤양병", "잎", "분류기 모델이 다루는 병해"],
            ["잎_귤응애.jpg", "귤응애", "잎", "분류기에 없는 해충 클래스"],
            ["잎_정상.jpg", "정상", "잎", "건강한 잎"],
            ["잎_진딧물.jpg", "진딧물", "잎", "분류기에 없는 해충 클래스"],
        ])
    
    add_info_box_multiline(slide, Emu(457200), Emu(3800000), Emu(10800000), Emu(700000), [
        "실행 환경",
        "GPU: NVIDIA RTX A6000  |  OS: Linux (Ubuntu)  |  Python 3.10 (conda env: cedpllm)",
        "PyTorch 2.10.0+cu128  |  transformers 5.3.0  |  Ollama (gemma3:12B)",
    ], bg_color=C_BG_BOX, text_size=12)
    
    add_info_box(slide, Emu(457200), Emu(4700000), Emu(10800000), Emu(400000),
                 "⚠ 분류기가 3-class만 지원하므로, 귤응애·진딧물 이미지는 구조적으로 오분류가 예상됨",
                 bg_color=C_BG_WARN, text_size=12, text_color=RGBColor(0x8B, 0x4C, 0x00))
    
    # ========== 슬라이드 6: 방식1 결과 요약 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 4", "방식 1: 이미지 분류 + RAG 결과", "6개 이미지 분류 결과 · 건강/병해충 분기 동작")
    
    add_table(slide, Emu(457200), Emu(1200000), Emu(10800000),
        ["이미지", "실제 라벨", "예측 라벨", "신뢰도", "RAG 사용", "분류 정확"],
        [
            ["열매_궤양병.jpg", "궤양병", "건강", "73.0%", "미사용", "❌"],
            ["열매_정상.jpg", "정상", "건강", "78.6%", "미사용", "✅"],
            ["잎_궤양병.jpg", "궤양병", "건강", "39.4%", "미사용", "❌"],
            ["잎_귤응애.jpg", "귤응애", "흑점병", "40.8%", "사용", "❌"],
            ["잎_정상.jpg", "정상", "건강", "45.0%", "미사용", "✅"],
            ["잎_진딧물.jpg", "진딧물", "궤양병", "41.7%", "사용", "❌"],
        ])
    
    # 결과 요약 박스
    add_info_box_multiline(slide, Emu(457200), Emu(3700000), Emu(5200000), Emu(900000), [
        "분류 정확도: 2/6 (33.3%)",
        "• 정상 이미지 2개만 정확 분류",
        "• 궤양병 열매를 73%로 건강 오분류",
        "• 귤응애·진딧물은 클래스 자체가 없음",
    ], bg_color=C_BG_WARN, text_size=12)
    
    add_info_box_multiline(slide, Emu(5900000), Emu(3700000), Emu(5350000), Emu(900000), [
        "파이프라인 동작: 6/6 성공",
        "• 건강 판정 → RAG 미호출, 고정 답변",
        "• 병해충 판정 → RAG 검색 → 상세 답변",
        "• 건강/병해충 분기 로직 정상 동작",
    ], bg_color=C_BG_GREEN, text_size=12)
    
    # ========== 슬라이드 7: 방식1 답변 예시 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 4", "방식 1: 답변 예시", "건강 판정 vs 병해충 판정 답변 비교")
    
    # 건강 답변
    add_text(slide, Emu(457200), Emu(1100000), Emu(5200000), Emu(300000),
             "건강 판정 (열매_정상.jpg)", size=14, bold=True, color=C_DARK)
    add_info_box_multiline(slide, Emu(457200), Emu(1400000), Emu(5200000), Emu(1400000), [
        "이미지 분석: 건강 78.6%",
        "분류: 건강(Healthy)",
        "",
        "이미지 분류 결과, 현재 감귤은 정상으로",
        "판단됩니다. 뚜렷한 병해충 증상은",
        "확인되지 않았습니다.",
        "",
        "→ RAG 호출 없음, 고정 답변 출력",
    ], bg_color=C_BG_GREEN, text_size=11)
    
    # 병해충 답변
    add_text(slide, Emu(5900000), Emu(1100000), Emu(5350000), Emu(300000),
             "병해충 판정 (잎_진딧물.jpg → 궤양병 41.7%)", size=14, bold=True, color=C_DARK)
    add_info_box_multiline(slide, Emu(5900000), Emu(1400000), Emu(5350000), Emu(1400000), [
        "병해충명: 궤양병",
        "학명: X. axonopodis pv.citri",
        "증상: 잎, 가지, 열매에 반점 형태",
        "방제: 5월 중·하순 첫 방제",
        "약제: 가스란(수), 네오보르도(수) 등 11종",
        "",
        "⚠ 이미지만으로 확정 진단 어려움",
        "→ RAG 기반 상세 정보 제공",
    ], bg_color=C_BG_BLUE, text_size=11)
    
    add_info_box(slide, Emu(457200), Emu(3050000), Emu(10800000), Emu(500000),
                 "장점: 병해충 판정 시 RAG의 상세 정보(약제명, 학명, 방제시기)는 정확하고 구체적\n"
                 "한계: 분류기 정확도(33%)가 병목 — 잘못된 분류 → 잘못된 RAG 검색 → 잘못된 답변",
                 bg_color=C_BG_WARN, text_size=12, text_color=RGBColor(0x8B, 0x4C, 0x00))
    
    # ========== 슬라이드 8: 방식2 결과 요약 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 5", "방식 2: VLM 멀티모달 결과", "6 이미지 × 8 질문 = 48개 테스트")
    
    add_text(slide, Emu(457200), Emu(1100000), Emu(10800000), Emu(350000),
             "테스트 질문 (이미지마다 8개 실행)", size=14, bold=True, color=C_DARK)
    
    add_table(slide, Emu(457200), Emu(1450000), Emu(10800000),
        ["번호", "질문"],
        [
            ["Q1", "이 감귤 상태가 이상한가요?"],
            ["Q2", "이 감귤에 병해충 증상이 보이나요?"],
            ["Q3", "이 감귤에 어떤 색의 점이나 반점이 보이나요?"],
            ["Q4", "이 감귤은 정상인가요, 병든 상태인가요?"],
            ["Q5", "의심되는 병명이 있다면 무엇인가요?"],
            ["Q6", "왜 그렇게 판단했나요?"],
            ["Q7", "어떻게 관리하거나 방제해야 하나요?"],
            ["Q8", "추가로 확인해야 할 점은 무엇인가요?"],
        ],
        row_height=Emu(280000))
    
    add_info_box(slide, Emu(457200), Emu(4200000), Emu(10800000), Emu(500000),
                 "전체 결과: 48/48 성공 ✅  |  VLM 백엔드: gemma3:12b (Ollama, 멀티모달)  |  분류기 없이 이미지 직접 분석",
                 bg_color=C_BG_GREEN, text_size=13)
    
    # ========== 슬라이드 9: VLM 정상 답변 예시 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 5", "VLM 결과 — 정상 이미지", "잎_정상.jpg: '이 감귤 상태가 이상한가요?'")
    
    add_info_box_multiline(slide, Emu(457200), Emu(1200000), Emu(10800000), Emu(3500000), [
        "관찰 증상:",
        "• 색상: 주황색, 익은 감귤의 일반적인 색상과 유사",
        "• 모양: 둥글고 균일, 변형 없음",
        "• 표면: 매끄럽고 흠집이나 반점 없음",
        "",
        "판정: 정상 ✅",
        "• 현재까지 이미지만으로는 특별한 병해충 감염이나 질병이 의심되지 않음",
        "• 감귤의 색상, 모양, 표면 상태가 정상적인 특징과 일치",
        "",
        "관리: 현재 특별한 관리 필요 없음",
        "• 정기적인 관찰, 통풍/햇빛 관리, 예방적 방제 권장",
        "",
        "→ 불필요한 병해충 정보를 출력하지 않음 ✅",
    ], bg_color=C_BG_GREEN, text_size=13)
    
    # ========== 슬라이드 10: VLM 병해충 답변 예시 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 5", "VLM 결과 — 병해충 이미지", "잎_궤양병.jpg: '이 감귤 상태가 이상한가요?'")
    
    add_info_box_multiline(slide, Emu(457200), Emu(1200000), Emu(5200000), Emu(3200000), [
        "관찰 증상:",
        "• 잎에 붉은색 반점 관찰",
        "• 반점 크기 다양, 특정 부위 집중",
        "• 다른 부분은 비교적 정상 녹색",
        "",
        "의심 병명:",
        "• 잎마름병 가능성",
        "• 탄저병 가능성",
        "• 응애류 피해 가능성",
        "",
        "근거: 붉은색 반점 → 곰팡이성",
        "질병 또는 응애류 증상과 유사",
    ], bg_color=C_BG_BLUE, text_size=12)
    
    add_info_box_multiline(slide, Emu(5900000), Emu(1200000), Emu(5350000), Emu(3200000), [
        "관리법:",
        "• 병든 잎 즉시 제거, 소각",
        "• 살균제 살포 (구리 계열 등)",
        "• 통풍 개선 가지치기",
        "• 응애 의심 시 살충제 (아바멕틴 등)",
        "",
        "추가 확인:",
        "• 잎 뒷면 응애류 확인 (돋보기)",
        "• 줄기/과실 유사 증상 여부",
        "• 전문가 상담 권장",
        "",
        "⚠ '가능성이 있습니다' 표현 사용",
        "→ 확정적 단정 없이 추정으로 표현 ✅",
    ], bg_color=C_BG_BOX, text_size=12)
    
    # ========== 슬라이드 11: VLM 해충 답변 예시 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 5", "VLM 결과 — 해충 이미지", "잎_진딧물.jpg (분류기는 궤양병으로 오분류)")
    
    add_info_box_multiline(slide, Emu(457200), Emu(1200000), Emu(5200000), Emu(2500000), [
        "VLM 응답 (이미지 직접 관찰):",
        "• 잎 표면에 검은색 반점 관찰",
        "• 반점 크기 작고, 잎 전체에 분포",
        "• 잎 색깔은 전체적으로 정상 녹색",
        "",
        "의심: 잎점무늬병 또는 깍지벌레 피해",
        "판정: '심각하지 않지만 원인 파악 필요'",
    ], bg_color=C_BG_BLUE, text_size=13)
    
    add_info_box_multiline(slide, Emu(5900000), Emu(1200000), Emu(5350000), Emu(2500000), [
        "분류기(방식1)의 판단:",
        "• 궤양병 41.7% → 궤양병 정보 출력",
        "• 실제는 진딧물 이미지",
        "• 완전히 잘못된 진단 ❌",
        "",
        "VLM(방식2)의 판단:",
        "• 이미지에서 실제 관찰 가능한 증상 기반",
        "• 확정 없이 가능성으로 표현 ✅",
    ], bg_color=C_BG_WARN, text_size=13)
    
    add_info_box(slide, Emu(457200), Emu(3950000), Emu(10800000), Emu(500000),
                 "핵심 차이: 분류기는 3-class로 제한되어 오분류 → VLM은 이미지를 직접 관찰하여 실제 증상 기반 답변 가능",
                 bg_color=C_BG_GREEN, text_size=13)
    
    # ========== 슬라이드 12: 두 방식 비교 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 6", "두 방식 비교 분석", "이미지 분류기 + RAG vs VLM 멀티모달")
    
    add_table(slide, Emu(457200), Emu(1200000), Emu(10800000),
        ["비교 항목", "방식 1 (분류기 + RAG)", "방식 2 (VLM 멀티모달)"],
        [
            ["이미지 분석", "ViT 3-class 분류 (라벨만 출력)", "VLM이 이미지 직접 관찰 (자유 서술)"],
            ["분류 범위", "궤양병/흑점병/건강 3개만", "제한 없음 (관찰 기반 자유 판단)"],
            ["분류 정확도", "33.3% (2/6)", "이미지 관찰 기반 (정량 비교 어려움)"],
            ["RAG 활용", "분류 결과로 RAG 검색", "미사용 (VLM 단독)"],
            ["약제/학명 제공", "RAG 기반 정확한 정보 ✅", "일반적 관리법 수준 ⚠"],
            ["병명 표현", "단정적 (분류기 결과)", "가능성/의심 표현 ✅"],
            ["확장성", "분류기 클래스 추가 필요", "프롬프트 수정만으로 확장 ✅"],
            ["실행 속도", "빠름 (분류+RAG)", "느림 (VLM 추론)"],
        ],
        row_height=Emu(300000))
    
    # ========== 슬라이드 13: 핵심 발견 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 7", "핵심 발견 & 시사점", "")
    
    # 발견 1
    add_info_box_multiline(slide, Emu(457200), Emu(1100000), Emu(10800000), Emu(1000000), [
        "① 분류기 + RAG: 분류기 정확도가 병목",
        "   분류가 맞으면 RAG의 상세 정보(약제, 학명, 방제시기)는 강력한 강점",
        "   하지만 3-class 한계 → 분류 실패 시 답변 전체가 틀림",
    ], bg_color=C_BG_BLUE, text_size=13)
    
    add_info_box_multiline(slide, Emu(457200), Emu(2300000), Emu(10800000), Emu(1000000), [
        "② VLM 멀티모달: 이미지 직접 관찰로 분류기 한계 극복",
        "   분류 클래스 제한 없이 다양한 증상 서술 가능",
        "   하지만 도메인 전문 지식(약제명, 학명 등) 부족 → RAG 보강 필요",
    ], bg_color=C_BG_BLUE, text_size=13)
    
    add_info_box_multiline(slide, Emu(457200), Emu(3500000), Emu(10800000), Emu(1200000), [
        "③ 최적 방향: VLM + RAG 결합",
        "",
        "   VLM으로 이미지 분석 → 병명 후보 키워드 추출",
        "   → 추출된 키워드로 text_qa RAG 검색 → 상세 약제/방제 정보 보강",
        "   → 두 방식의 장점만 결합한 하이브리드 파이프라인",
    ], bg_color=C_BG_GREEN, text_size=13)
    
    # ========== 슬라이드 14: 향후 계획 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "Part 7", "향후 계획", "Next Steps")
    
    steps = [
        ("1", "VLM + RAG 결합 파이프라인", "VLM 응답에서 병명 키워드 추출 → text_qa RAG로 상세 정보 보강"),
        ("2", "분류기 모델 개선 검토", "감귤 전용 다중 클래스 분류 모델 파인튜닝 또는 교체"),
        ("3", "VLM 모델 비교", "Agri-LLaVA, Qwen2.5-VL 등 농업 특화 모델 성능 비교"),
        ("4", "평가 체계 구축", "정량 평가 지표 설계 (정확도, 관련성, 답변 품질 등)"),
        ("5", "테스트 데이터 확장", "더 많은 병해충 종류 · 다양한 촬영 조건 이미지 확보"),
    ]
    
    y = Emu(1100000)
    for num, title, desc in steps:
        add_rect(slide, Emu(457200), y, Emu(600000), Emu(600000), C_ACCENT)
        add_text(slide, Emu(457200), y + Emu(100000), Emu(600000), Emu(400000),
                 num, size=20, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        add_text(slide, Emu(1200000), y + Emu(50000), Emu(9500000), Emu(280000),
                 title, size=15, bold=True, color=C_DARK)
        add_text(slide, Emu(1200000), y + Emu(330000), Emu(9500000), Emu(280000),
                 desc, size=12, color=C_GRAY)
        y += Emu(700000)
    
    # ========== 슬라이드 15: 프로젝트 구조 ==========
    slide = add_blank_slide(prs)
    add_slide_title(slide, "참고", "프로젝트 파일 구조", "")
    
    add_info_box_multiline(slide, Emu(457200), Emu(1100000), Emu(10800000), Emu(4500000), [
        "keti-pest-disease-qa/",
        "├── image_qa/",
        "│   ├── image_classifier.py        # ViT 이미지 분류기",
        "│   ├── image_qa.py                 # 분류기+RAG 파이프라인 (방식 1)",
        "│   ├── label_utils.py              # 정상/건강 라벨 판별",
        "│   ├── vlm_prompt_builder.py       # VLM 프롬프트 빌더",
        "│   ├── vlm_inference.py             # VLM 추론 모듈 (모델 교체 용이)",
        "│   ├── vlm_qa.py                    # VLM QA 파이프라인 (방식 2)",
        "│   ├── test_image_qa.py            # 방식 1 테스트 스크립트",
        "│   ├── test_vlm_qa.py              # 방식 2 테스트 스크립트",
        "│   ├── prompts/                     # 프롬프트 템플릿",
        "│   └── test_images/                 # 테스트 이미지 6장",
        "├── text_qa/",
        "│   ├── pest_disease_qa.py          # 텍스트 RAG Q&A 시스템",
        "│   ├── chroma_db/                   # 벡터 저장소",
        "│   └── text_info/                   # 병해충 지식 데이터",
        "└── result_md/                       # 테스트 결과 리포트 (MD + PPT)",
    ], bg_color=C_BG_BOX, text_size=12)
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    prs.save(OUTPUT_PATH)
    print(f"✅ PPT 저장 완료: {OUTPUT_PATH}")
    print(f"   슬라이드 수: {len(prs.slides)}")


if __name__ == "__main__":
    build_ppt()

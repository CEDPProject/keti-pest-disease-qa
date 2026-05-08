"""라벨 유틸리티 - 정상/건강 판정 키워드 관리"""

HEALTHY_KEYWORDS = [
    "healthy",
    "normal",
    "정상",
    "건강",
]


def is_healthy_label(label: str) -> bool:
    """Return True if the predicted label means healthy or normal."""
    return label.strip().lower() in [kw.lower() for kw in HEALTHY_KEYWORDS]

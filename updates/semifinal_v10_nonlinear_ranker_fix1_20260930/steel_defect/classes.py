"""Competition class definitions and conservative aliases."""

CLASS_NAMES = [
    "jieba",
    "zonglie",
    "qilie",
    "jiaza",
    "yiwuyaru",
    "huashang",
    "mamianmakeng",
    "yanghuatiepi",
    "gunyin",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Only spelling/format aliases are included. Semantic class merging must follow
# the official annotation package, not guesses made from display names.
CLASS_ALIASES = {
    "结疤": "jieba",
    "纵裂": "zonglie",
    "纵向裂纹": "zonglie",
    "气裂": "qilie",
    "气泡状裂纹": "qilie",
    "夹杂": "jiaza",
    "异物压入": "yiwuyaru",
    "划伤": "huashang",
    "麻面麻坑": "mamianmakeng",
    "氧化铁皮": "yanghuatiepi",
    "辊印": "gunyin",
}


def canonical_class(name: str) -> str:
    """Normalize an annotation class without silently merging unknown classes."""
    normalized = name.strip().lower().replace(" ", "")
    return CLASS_ALIASES.get(normalized, normalized)


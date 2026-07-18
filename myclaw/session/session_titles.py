"""Session title normalization rules."""

_TITLE_LIMIT = 60
_PAIRED_QUOTES = (
    ('"', '"'),
    ("'", "'"),
    ("\u201c", "\u201d"),
    ("\u2018", "\u2019"),
    ("\u300c", "\u300d"),
    ("\u300e", "\u300f"),
    ("\u00ab", "\u00bb"),
)


def normalize_session_title(value: str) -> str:
    """Normalize the first nonblank line into a bounded Session title."""
    for line in value.splitlines():
        title = " ".join(line.split())
        if not title:
            continue
        for opening, closing in _PAIRED_QUOTES:
            if len(title) >= 2 and title.startswith(opening) and title.endswith(closing):
                title = " ".join(title[1:-1].split())
                break
        return title[:_TITLE_LIMIT]
    return ""

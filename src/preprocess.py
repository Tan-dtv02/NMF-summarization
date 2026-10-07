"""Tiền xử lý văn bản tiếng Việt: làm sạch, tách câu, tách từ, bỏ từ dừng.

Đầu vào là một chuỗi văn bản thô. Đầu ra là danh sách câu gốc (để hiển thị và
đưa vào bản tóm tắt) cùng danh sách từ khóa của từng câu (để dựng ma trận V).

Ví dụ:
    >>> kq = preprocess("Pin của điện thoại. Phở và bún.")
    >>> kq.sentences
    ['Pin của điện thoại.', 'Phở và bún.']
    >>> kq.tokens
    [['pin', 'điện_thoại'], ['phở', 'bún']]
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# underthesea biết "điện thoại" là MỘT từ. Nếu không cài được thư viện này,
# chương trình vẫn chạy nhưng tách theo khoảng trắng ("điện" và "thoại" thành
# hai từ), nên chất lượng chủ đề sẽ kém hơn.
try:
    from underthesea import sent_tokenize as _uts_sent_tokenize
    from underthesea import word_tokenize as _uts_word_tokenize

    HAS_UNDERTHESEA = True
except Exception as loi:  # ImportError, hoặc lỗi khi thư viện tự nạp mô hình
    HAS_UNDERTHESEA = False
    warnings.warn(
        "Không nạp được underthesea nên sẽ tách từ theo khoảng trắng "
        f"(kém chính xác hơn). Lý do: {loi}",
        RuntimeWarning,
        stacklevel=2,
    )

# Kho từ dừng nằm trong MỘT tệp: kho công khai vietnamese-stopwords (khoảng
# 1.900 từ) đã gộp với phần bổ sung của nhóm. Nguồn và giấy phép ghi ở đầu tệp.
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STOPWORDS_PATH = _DATA_DIR / "stopwords_vi.txt"

# Dùng khi không có underthesea: cắt sau dấu kết thúc câu, trước khoảng trắng.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_PUNCT_EDGES = re.compile(r"^\W+|\W+$")

# Đường link, địa chỉ email và tên miền viết trần (vd "tin-tuc.vn/bai-1"). Chúng
# không phải nội dung, và dấu chấm bên trong làm bộ tách câu cắt nhầm một câu
# thành nhiều mảnh. Khoảng trắng đứng trước cũng được lấy theo để khi bỏ đường
# link đi, dấu câu phía sau dính lại vào chữ đứng trước.
_LINK = re.compile(
    r"[ \t]*(?:"
    r"(?:https?://|www\.)[^\s<>\"']+"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|vn|edu|gov|info|io)(?:\.vn)?\b(?:/[^\s<>\"']*)?"
    r")",
    flags=re.IGNORECASE,
)
_LINK_TAIL = ".,;:!?…)]}"  # dấu câu dính ở cuối đường link, không thuộc đường link

# Tiếng Việt có hai cách đặt dấu thanh cho vần oa, oe, uy đứng cuối tiếng:
# kiểu cũ "hòa, khỏe, thủy" và kiểu mới "hoà, khoẻ, thuỷ". Máy coi đó là hai từ
# khác nhau, nên phải đưa về một kiểu. Chọn kiểu cũ vì bộ tách từ underthesea
# nhận ra từ ghép viết kiểu cũ tốt hơn hẳn (thử 14 từ ghép: 9 so với 2).
_TONES = {"a": "àáảãạ", "e": "èéẻẽẹ", "y": "ỳýỷỹỵ", "o": "òóỏõọ", "u": "ùúủũụ"}
_TONE_OF = {char: (base, i) for base, chars in _TONES.items() for i, char in enumerate(chars)}
_NOT_LETTER_AHEAD = r"(?![^\W\d_])"  # ngay sau không còn chữ cái: vần đứng cuối tiếng
# "oà" -> "òa", "oẻ" -> "ỏe", "uỷ" -> "ủy" (không đụng "quý": u ở đó thuộc phụ âm qu)
_TONE_ON_SECOND = re.compile(
    rf"(?<![qQ])(?:([oO])([{_TONES['a']}{_TONES['e']}])|([uU])([{_TONES['y']}]))"
    rf"{_NOT_LETTER_AHEAD}",
    flags=re.IGNORECASE,
)
# "qủa" -> "quả", "qúy" -> "quý": sau q, dấu thanh không bao giờ nằm trên u
_TONE_ON_QU = re.compile(rf"([qQ])([{_TONES['u']}])([aeyAEY])", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ProcessedText:
    """Kết quả tiền xử lý. Hai danh sách có cùng độ dài: mỗi câu một phần tử."""

    sentences: list[str]  # câu gốc, giữ nguyên chữ hoa và dấu câu
    tokens: list[list[str]]  # từ khóa của từng câu sau khi tách từ, bỏ từ dừng


def tokenizer_name() -> str:
    """Tên bộ tách từ đang dùng, để giao diện báo cho người xem."""
    return "underthesea" if HAS_UNDERTHESEA else "khoảng trắng (dự phòng)"


@lru_cache(maxsize=4)
def load_stopwords(path: str | Path = STOPWORDS_PATH) -> frozenset[str]:
    """Đọc kho từ dừng từ một tệp, mỗi dòng một từ.

    Dòng bắt đầu bằng "#" là ghi chú. Từ nhiều âm tiết được nối bằng dấu gạch
    dưới cho khớp kết quả tách từ. Từ dừng cũng được đưa về cùng kiểu đặt dấu
    thanh với văn bản (xem normalize_tone), để "toà" trong kho khớp được với
    "tòa" trong bài. Tệp không tồn tại thì trả về tập rỗng.
    """
    path = Path(path)
    if not path.exists():
        return frozenset()
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = normalize_tone(unicodedata.normalize("NFC", line)).strip().lower()
        if line and not line.startswith("#"):
            words.add(line.replace(" ", "_"))
    return frozenset(words)


def _with_case(char: str, like: str) -> str:
    """Viết `char` hoa hay thường theo đúng chữ `like` mà nó thay thế."""
    return char.upper() if like.isupper() else char


def normalize_tone(text: str) -> str:
    """Đưa cách đặt dấu thanh về một kiểu: "hoà" -> "hòa", "thuỷ" -> "thủy".

    Chỉ đổi vần oa, oe, uy đứng CUỐI tiếng, vì chỉ ở đó hai kiểu mới khác nhau
    ("hoàn", "khuỷu" thì kiểu nào cũng viết như vậy). Sửa luôn lỗi gõ đặt dấu
    lên chữ u sau q: "qủa" -> "quả".
    """

    def to_first_vowel(match: re.Match) -> str:
        first = match.group(1) or match.group(3)
        second = match.group(2) or match.group(4)
        base, tone = _TONE_OF[second.lower()]  # nguyên âm sau và thanh nó đang mang
        toned_first = _TONES[first.lower()][tone]
        return _with_case(toned_first, first) + _with_case(base, second)

    def to_vowel_after_qu(match: re.Match) -> str:
        q, u, vowel = match.groups()
        _, tone = _TONE_OF[u.lower()]
        return q + _with_case("u", u) + _with_case(_TONES[vowel.lower()][tone], vowel)

    text = _TONE_ON_SECOND.sub(to_first_vowel, text)
    return _TONE_ON_QU.sub(to_vowel_after_qu, text)


def remove_links(text: str) -> str:
    """Bỏ đường link, email và tên miền trần; giữ lại dấu câu đứng sau chúng."""

    def keep_tail(match: re.Match) -> str:
        link = match.group(0)
        tail = link[len(link.rstrip(_LINK_TAIL)) :]
        return tail or " "

    return _LINK.sub(keep_tail, text)


def normalize(text: str) -> str:
    """Làm sạch nhẹ: Unicode dựng sẵn (NFC), bỏ đường link, thống nhất dấu thanh,
    gom khoảng trắng thừa.

    Văn bản dán từ nhiều nguồn có thể viết cùng một chữ theo nhiều cách: "ệ" có
    hai cách mã hóa, "hòa" có hai cách đặt dấu. Không chuẩn hóa thì cùng một từ
    sẽ bị đếm thành hai từ khác nhau.
    """
    text = unicodedata.normalize("NFC", text)
    text = normalize_tone(remove_links(text))
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Tách văn bản thành câu. Mỗi dòng xuống hàng luôn là một ranh giới câu."""
    sentences = []
    for paragraph in normalize(text).splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if HAS_UNDERTHESEA:
            parts = _uts_sent_tokenize(paragraph)
        else:
            parts = _SENTENCE_END.split(paragraph)
        # Bỏ những "câu" không có chữ cái nào (ví dụ chỉ gồm "..." hoặc số).
        sentences.extend(p.strip() for p in parts if any(c.isalpha() for c in p))
    return sentences


def tokenize(sentence: str, stopwords: frozenset[str] | None = None) -> list[str]:
    """Tách một câu thành danh sách từ khóa.

    Các bước: chữ thường -> tách từ -> nối từ ghép bằng "_" -> bỏ dấu câu,
    bỏ số, bỏ từ một ký tự -> bỏ từ dừng.
    """
    if stopwords is None:
        stopwords = load_stopwords()
    sentence = normalize(sentence).lower()
    raw = _uts_word_tokenize(sentence) if HAS_UNDERTHESEA else sentence.split()

    tokens = []
    for word in raw:
        word = _PUNCT_EDGES.sub("", word).replace(" ", "_")
        if len(word) < 2 or not any(c.isalpha() for c in word):
            continue
        if word not in stopwords:
            tokens.append(word)
    return tokens


def preprocess(text: str) -> ProcessedText:
    """Chạy trọn quy trình tiền xử lý cho một văn bản."""
    sentences = split_sentences(text)
    stopwords = load_stopwords()
    return ProcessedText(
        sentences=sentences,
        tokens=[tokenize(s, stopwords) for s in sentences],
    )


def filter_rare_words(tokens: list[list[str]], min_df: int = 2) -> list[list[str]]:
    """Bỏ những từ xuất hiện trong ít hơn `min_df` câu.

    Từ chỉ có mặt ở đúng một câu không nối được câu đó với câu nào khác, nên
    không giúp tìm ra chủ đề chung. Giữ chúng lại thì NMF dễ coi mỗi câu là
    một chủ đề riêng, và từ khóa của chủ đề toàn là từ hiếm, khó hiểu.
    """
    df = Counter(word for sentence in tokens for word in set(sentence))
    return [[word for word in sentence if df[word] >= min_df] for sentence in tokens]

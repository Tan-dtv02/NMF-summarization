"""Tiền xử lý văn bản tiếng Việt: tách câu, chuẩn hóa, tách từ, bỏ từ dừng.

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

# Từ dừng lấy từ hai tệp, đọc theo thứ tự này:
#   1. kho công khai (khoảng 1.900 từ, github.com/stopwords/vietnamese-stopwords);
#   2. tệp của nhóm: thêm từ còn thiếu, và nêu những từ muốn GIỮ LẠI dù kho có.
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PUBLIC_STOPWORDS_PATH = _DATA_DIR / "vietnamese-stopwords.txt"
STOPWORDS_PATH = _DATA_DIR / "stopwords_vi.txt"

# Dùng khi không có underthesea: cắt sau dấu kết thúc câu, trước khoảng trắng.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_PUNCT_EDGES = re.compile(r"^\W+|\W+$")


@dataclass(frozen=True)
class ProcessedText:
    """Kết quả tiền xử lý. Hai danh sách có cùng độ dài: mỗi câu một phần tử."""

    sentences: list[str]  # câu gốc, giữ nguyên chữ hoa và dấu câu
    tokens: list[list[str]]  # từ khóa của từng câu sau khi tách từ, bỏ từ dừng


def tokenizer_name() -> str:
    """Tên bộ tách từ đang dùng, để giao diện báo cho người xem."""
    return "underthesea" if HAS_UNDERTHESEA else "khoảng trắng (dự phòng)"


@lru_cache(maxsize=4)
def load_stopwords(*paths: str | Path) -> frozenset[str]:
    """Đọc từ dừng từ một hay nhiều tệp (mặc định: kho công khai rồi tệp của nhóm).

    Quy ước trong tệp, mỗi dòng một từ:
      - dòng bắt đầu bằng "#" là ghi chú;
      - dòng bắt đầu bằng "!" là NGOẠI LỆ: từ đó được giữ lại, không coi là từ
        dừng, dù tệp nào khác có liệt kê nó (ví dụ "!nước").
    Từ nhiều âm tiết được nối bằng dấu gạch dưới cho khớp kết quả tách từ.
    Tệp không tồn tại thì bỏ qua.
    """
    paths = paths or (PUBLIC_STOPWORDS_PATH, STOPWORDS_PATH)
    words, kept = set(), set()
    for path in map(Path, paths):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = unicodedata.normalize("NFC", line).strip().lower()
            if not line or line.startswith("#"):
                continue
            target = kept if line.startswith("!") else words
            target.add(line.lstrip("!").strip().replace(" ", "_"))
    return frozenset(words - kept)


def normalize(text: str) -> str:
    """Đưa chữ về dạng Unicode dựng sẵn (NFC) và gom khoảng trắng thừa.

    Văn bản dán từ nhiều nguồn có thể viết "ệ" theo hai cách khác nhau về mã
    ký tự. Không chuẩn hóa thì cùng một từ sẽ bị đếm thành hai từ khác nhau.
    """
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[ \t ]+", " ", text).strip()


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

"""Tóm tắt trích xuất một văn bản bằng NMF.

Quy trình:
    văn bản -> tách câu, tách từ (preprocess) -> ma trận V (từ x câu, TF-IDF)
            -> NMF tự viết cho ra W và H -> chọn n câu
            -> xếp lại theo thứ tự trong bài gốc.

Hai con số được tính từ H và dùng ở mọi nơi:

    trọng số chủ đề i  =  tổng hàng i của H / tổng mọi phần tử của H
                          (chủ đề nào được nhiều câu nhắc tới thì quan trọng)
    điểm câu j         =  tổng trên các chủ đề i của (H[i, j] * trọng số chủ đề i)
                          (câu nào chứa nhiều chủ đề quan trọng thì điểm cao)

Có hai cách chọn câu:

  "coverage" (mặc định): đi vòng qua các chủ đề, bắt đầu từ chủ đề quan trọng
      nhất, mỗi lượt lấy câu thuộc chủ đề đó nhiều nhất. Bản tóm tắt nhờ vậy
      nhắc tới nhiều chủ đề khác nhau của bài.
  "score": lấy n câu có điểm câu cao nhất, theo Lee và cộng sự (2009),
      "Automatic generic document summarization based on non-negative matrix
      factorization". Cách này dễ dồn hết các câu vào chủ đề lớn nhất.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.nmf_scratch import build_tfidf_matrix, nmf, normalize_topics
from src.preprocess import ProcessedText, filter_rare_words, preprocess

MIN_SENTENCES = 3  # ít câu hơn mức này thì không có gì để tóm tắt
STRATEGIES = {
    "coverage": "Phủ chủ đề (mỗi chủ đề lần lượt góp một câu)",
    "score": "Điểm tổng (câu chứa nhiều chủ đề quan trọng)",
}


class TextTooShortError(ValueError):
    """Văn bản không đủ nội dung để tóm tắt. Thông báo viết sẵn cho người dùng."""


@dataclass(frozen=True)
class SummaryResult:
    """Mọi thứ giao diện cần để hiển thị và giải thích bản tóm tắt."""

    sentences: list[str]  # n câu gốc
    tokens: list[list[str]]  # từ khóa của từng câu (sau khi lọc từ hiếm, nếu có lọc)
    vocab: list[str]  # m từ, ứng với các hàng của V và W
    V: np.ndarray  # (m x n)
    W: np.ndarray  # (m x k) đã chuẩn hóa: mỗi cột có tổng bằng 1
    H: np.ndarray  # (k x n)
    errors: list[float]  # sai số sau mỗi vòng lặp
    topic_weights: np.ndarray  # (k,) tầm quan trọng của từng chủ đề, tổng bằng 1
    dominant_topic: np.ndarray  # (n,) chủ đề trội của từng câu, -1 nếu câu rỗng
    sentence_scores: np.ndarray  # (n,) điểm của từng câu
    selected: list[int]  # chỉ số các câu được chọn, theo thứ tự bài gốc

    @property
    def k(self) -> int:
        return self.H.shape[0]

    @property
    def summary(self) -> str:
        return " ".join(self.sentences[j] for j in self.selected)

    def top_keywords(self, topic: int, n: int = 6) -> list[tuple[str, float]]:
        """Các từ nặng nhất của một chủ đề (cột `topic` của W), kèm trọng số.

        Chỉ lấy từ có trọng số từ 10% của từ đứng đầu trở lên, để không kéo
        theo những từ gần như không thuộc chủ đề.
        """
        column = self.W[:, topic]
        order = np.argsort(-column, kind="stable")[:n]
        return [(self.vocab[i], float(column[i])) for i in order if column[i] >= 0.1 * column.max()]


def summarize(
    text: str,
    k: int = 3,
    n_sentences: int = 3,
    strategy: str = "coverage",
    seed: int = 42,
    min_df: int = 2,
) -> SummaryResult:
    """Tóm tắt văn bản thô. Báo TextTooShortError nếu văn bản quá ngắn.

    k           số chủ đề
    n_sentences số câu của bản tóm tắt
    strategy    cách chọn câu: "score" hoặc "coverage" (xem STRATEGIES)
    seed        hạt giống khởi tạo NMF, cố định để chạy lại ra cùng kết quả
    min_df      bỏ từ xuất hiện trong ít hơn min_df câu (đặt 1 để giữ mọi từ)
    """
    return summarize_processed(preprocess(text), k, n_sentences, strategy, seed, min_df)


def summarize_processed(
    doc: ProcessedText,
    k: int = 3,
    n_sentences: int = 3,
    strategy: str = "coverage",
    seed: int = 42,
    min_df: int = 2,
) -> SummaryResult:
    """Như summarize() nhưng nhận văn bản đã tiền xử lý.

    Tách riêng để phần thử nghiệm chỉ phải tách từ một lần cho mỗi bài rồi
    chạy lại với nhiều giá trị k (tách từ là bước chậm nhất).
    """
    n = len(doc.sentences)
    if n == 0:
        raise TextTooShortError("Chưa có văn bản. Hãy dán hoặc chọn một bài để tóm tắt.")
    if n < MIN_SENTENCES:
        raise TextTooShortError(
            f"Văn bản chỉ có {n} câu, quá ngắn để tóm tắt (cần ít nhất {MIN_SENTENCES} câu)."
        )
    if n <= n_sentences:
        raise TextTooShortError(
            f"Văn bản có {n} câu, không dài hơn bản tóm tắt {n_sentences} câu. "
            "Hãy giảm số câu tóm tắt hoặc dùng văn bản dài hơn."
        )

    tokens = choose_tokens(doc.tokens, min_df, k, n_sentences)
    V, vocab = build_tfidf_matrix(tokens)
    if V.sum() == 0:
        raise TextTooShortError(
            "Không tìm thấy từ khóa nào giúp phân biệt các câu "
            "(các câu quá giống nhau hoặc chỉ gồm từ dừng)."
        )

    k = max(1, min(k, n - 1, len(vocab)))  # không thể có nhiều chủ đề hơn số câu
    result = nmf(V, k, seed=seed)
    W, H = normalize_topics(result.W, result.H)

    topic_weights = H.sum(axis=1) / H.sum()
    sentence_scores = topic_weights @ H
    has_content = V.sum(axis=0) > 0
    dominant_topic = np.where(has_content, H.argmax(axis=0), -1)

    if strategy == "score":
        selected = select_top(sentence_scores, n_sentences)
    else:
        selected = select_by_coverage(H, topic_weights, n_sentences)

    return SummaryResult(
        sentences=doc.sentences,
        tokens=tokens,
        vocab=vocab,
        V=V,
        W=W,
        H=H,
        errors=result.errors,
        topic_weights=topic_weights,
        dominant_topic=dominant_topic,
        sentence_scores=sentence_scores,
        selected=selected,
    )


def choose_tokens(
    tokens: list[list[str]], min_df: int, k: int, n_sentences: int
) -> list[list[str]]:
    """Lọc từ hiếm nếu sau khi lọc vẫn còn đủ nội dung, ngược lại giữ nguyên.

    Với bài rất ngắn, gần như từ nào cũng chỉ xuất hiện một lần. Lọc đi thì
    không còn gì để phân tích, nên khi đó dùng toàn bộ từ.
    """
    if min_df <= 1:
        return tokens
    filtered = filter_rare_words(tokens, min_df)
    n_words = len({word for sentence in filtered for word in sentence})
    n_nonempty = sum(1 for sentence in filtered if sentence)
    enough = n_words > k and n_nonempty > max(n_sentences, MIN_SENTENCES - 1)
    return filtered if enough else tokens


def select_top(scores: np.ndarray, n_sentences: int) -> list[int]:
    """Lấy n câu điểm cao nhất rồi xếp lại theo thứ tự trong bài gốc."""
    best = np.argsort(-scores, kind="stable")[:n_sentences]  # bằng điểm thì ưu tiên câu đứng trước
    return sorted(int(j) for j in best)


def select_by_coverage(H: np.ndarray, topic_weights: np.ndarray, n_sentences: int) -> list[int]:
    """Đi vòng qua các chủ đề, từ quan trọng nhất; mỗi lượt lấy câu mạnh nhất.

    Cách này bảo đảm bản tóm tắt nhắc tới nhiều chủ đề khác nhau, thay vì
    dồn hết vào chủ đề lớn nhất.
    """
    k, n = H.shape
    topics = np.argsort(-topic_weights, kind="stable")
    chosen: list[int] = []
    for turn in range(k * n):  # chặn trên số lượt, để vòng lặp chắc chắn kết thúc
        if len(chosen) == min(n_sentences, n):
            break
        topic = topics[turn % k]
        for j in np.argsort(-H[topic], kind="stable"):
            if H[topic, j] <= 0:
                break  # chủ đề này không còn câu nào liên quan
            if int(j) not in chosen:
                chosen.append(int(j))
                break
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Phần thêm: tóm tắt theo một chủ đề hoặc một từ khóa do người dùng chọn
# ---------------------------------------------------------------------------


def summarize_by_topic(result: SummaryResult, topic: int, n_sentences: int = 3) -> list[int]:
    """Các câu thuộc chủ đề đã chọn nhiều nhất (hàng `topic` của H)."""
    scores = np.where(result.H[topic] > 0, result.H[topic], -1.0)
    return [j for j in select_top(scores, n_sentences) if scores[j] > 0]


def keyword_topic_profile(result: SummaryResult, keyword: str) -> np.ndarray | None:
    """Từ khóa người dùng gõ thuộc về các chủ đề theo tỉ lệ nào (tổng bằng 1).

    Cách 1: tìm các từ trong bảng từ vựng khớp với từ khóa rồi cộng các hàng
    tương ứng của W.
    Cách 2 (khi từ khóa không có trong bảng từ vựng, ví dụ đã bị lọc vì hiếm):
    tìm các câu có chứa từ khóa rồi cộng các cột tương ứng của H.
    Trả về None nếu bài không nhắc tới từ khóa.
    """
    wanted = " ".join(keyword.lower().replace("_", " ").split())
    if not wanted:
        return None
    padded = f" {wanted} "
    rows = [
        i
        for i, word in enumerate(result.vocab)
        # khớp cả hai chiều: gõ "pin" khớp từ "pin"; gõ "giá xăng" khớp "giá_xăng",
        # "giá", "xăng"; gõ "điện" khớp "điện_thoại"
        if f" {word.replace('_', ' ')} " in padded or padded in f" {word.replace('_', ' ')} "
    ]
    if rows:
        # W đã chuẩn hóa theo cột, nên nhân với trọng số chủ đề để so sánh được
        # "từ này thuộc chủ đề nào" giữa các chủ đề lớn nhỏ khác nhau.
        profile = (result.W[rows] * result.topic_weights).sum(axis=0)
    else:
        columns = [j for j, s in enumerate(result.sentences) if wanted in s.lower()]
        profile = result.H[:, columns].sum(axis=1)
    return profile / profile.sum() if profile.sum() > 0 else None


def summarize_by_keyword(
    result: SummaryResult, keyword: str, n_sentences: int = 3
) -> tuple[list[int], np.ndarray] | None:
    """Tóm tắt xoay quanh một từ khóa: đổi từ khóa thành tỉ lệ chủ đề, rồi
    chấm điểm câu theo tỉ lệ đó. Nhờ đi qua chủ đề, câu không chứa đúng từ
    khóa nhưng nói cùng chủ đề vẫn được chọn.
    """
    profile = keyword_topic_profile(result, keyword)
    if profile is None:
        return None
    scores = profile @ result.H
    scores = np.where(scores > 0, scores, -1.0)
    return [j for j in select_top(scores, n_sentences) if scores[j] > 0], profile


# ---------------------------------------------------------------------------
# Phần thêm: gợi ý số chủ đề k
# ---------------------------------------------------------------------------


def error_by_k(V: np.ndarray, k_values: list[int], seed: int = 42) -> list[float]:
    """Sai số cuối cùng của NMF với từng giá trị k."""
    return [nmf(V, k, seed=seed).errors[-1] for k in k_values]


def suggest_k(k_values: list[int], errors: list[float]) -> int:
    """Gợi ý k theo "điểm khuỷu tay" của đường sai số.

    Tăng k thì sai số luôn giảm, nên không thể chọn k có sai số nhỏ nhất. Thay
    vào đó, nối điểm đầu và điểm cuối của đường sai số bằng một đoạn thẳng rồi
    chọn k nằm xa đoạn thẳng đó nhất: đó là chỗ đường cong gập lại, thêm chủ đề
    nữa cũng không giảm sai số được bao nhiêu. Đây là gợi ý theo kinh nghiệm,
    không phải đáp án duy nhất.
    """
    if len(k_values) < 3:
        return k_values[0]
    x = np.asarray(k_values, dtype=float)
    y = np.asarray(errors, dtype=float)
    # đưa hai trục về đoạn [0, 1] để khoảng cách không phụ thuộc đơn vị
    x = (x - x[0]) / (x[-1] - x[0])
    y = (y - y[-1]) / (y[0] - y[-1]) if y[0] != y[-1] else np.zeros_like(y)
    distance_below_line = (1 - x) - y
    return k_values[int(np.argmax(distance_below_line))]

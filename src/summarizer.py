"""Tóm tắt trích xuất một văn bản bằng NMF.

Quy trình:
    văn bản -> làm sạch, tách câu, tách từ (preprocess)
            -> ma trận TF-IDF từ x câu theo công thức của slide
            -> chuẩn hóa độ dài từng câu: mỗi cột dài 1, được ma trận V
            -> NMF tự viết, chạy từ 5 điểm khởi tạo, giữ lần sai số nhỏ nhất
            -> chọn n câu từ H -> xếp lại theo thứ tự trong bài gốc.

Hai bước nằm giữa TF-IDF và việc chọn câu, mỗi bước chữa một điểm yếu:

  1. Chuẩn hóa độ dài câu. Với TF-IDF thô, câu ngắn và câu chứa từ hiếm có
     giá trị rất lớn, NMF dồn sức khớp vài câu đó nên chủ đề bị lệch. Khi mọi
     câu dài bằng nhau thì câu nào cũng có tiếng nói ngang nhau.
  2. Nhiều điểm khởi tạo. NMF chỉ tới được điểm dừng cục bộ, kết quả đổi theo
     seed. Chạy 5 seed cố định rồi giữ lần khớp V tốt nhất thì ổn định hơn.

Đã thử thêm một bước nữa là nhân "lượng nội dung" của câu vào H trước khi chọn,
để ưu tiên câu dài. Trên 1000 bài báo thật, bước đó làm ROUGE-1 giảm từ 0,255
xuống 0,244 vì bản tóm tắt dài ra trong khi bản tham chiếu ngắn, nên không dùng.

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

from src.nmf_scratch import (
    best_rank_k_error,
    init_factors,
    nmf,
    nmf_best_of,
    normalize_columns,
    normalize_topics,
    sklearn_reference,
    tfidf_parts,
)
from src.preprocess import ProcessedText, filter_rare_words, preprocess

MIN_SENTENCES = 3  # ít câu hơn mức này thì không có gì để tóm tắt
STRATEGIES = {
    "coverage": "Phủ chủ đề (mỗi chủ đề lần lượt góp một câu)",
    "score": "Điểm tổng (câu chứa nhiều chủ đề quan trọng)",
}

# Cách chạy NMF khi tóm tắt. Màn mô phỏng dùng mặc định riêng của hàm nmf.
SEEDS = (0, 1, 2, 3, 4)  # 5 điểm khởi tạo cố định, nên chạy lại vẫn ra kết quả cũ
MAX_ITER = 2000
TOL = 1e-6  # chặt hơn màn mô phỏng: chạy tới khi sai số gần như đứng yên hẳn


class TextTooShortError(ValueError):
    """Văn bản không đủ nội dung để tóm tắt. Thông báo viết sẵn cho người dùng."""


@dataclass(frozen=True)
class SummaryResult:
    """Mọi thứ giao diện cần để hiển thị và giải thích bản tóm tắt."""

    sentences: list[str]  # n câu gốc
    tokens: list[list[str]]  # từ khóa của từng câu (sau khi lọc từ hiếm, nếu có lọc)
    vocab: list[str]  # m từ, ứng với các hàng của V và W
    V: np.ndarray  # (m x n) TF-IDF đã chuẩn hóa độ dài từng câu
    W: np.ndarray  # (m x k) đã chuẩn hóa: mỗi cột có tổng bằng 1
    H: np.ndarray  # (k x n) kết quả NMF: W.H xấp xỉ V; dùng để chọn câu
    errors: list[float]  # sai số sau mỗi vòng lặp của lần chạy được giữ lại
    seed: int  # seed của lần chạy được giữ lại
    floor: float  # sai số nhỏ nhất mà một xấp xỉ hạng k có thể đạt (tính bằng SVD)
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

    @property
    def n_iter(self) -> int:
        return len(self.errors) - 1

    @property
    def explained(self) -> float:
        """Phần của V mà k chủ đề giải thích được: 1 - (sai số / ||V||)^2.

        Bằng 1 nếu W.H tái tạo trọn V, bằng 0 nếu W.H không khớp được gì.
        Đây là mức NÉN THÔNG TIN, không phải thước đo chất lượng bản tóm tắt.
        """
        total = float(np.linalg.norm(self.V))
        return 1.0 - (self.errors[-1] / total) ** 2 if total > 0 else 0.0

    @property
    def floor_gap(self) -> float | None:
        """Sai số của NMF cao hơn mức sàn bao nhiêu: (sai số - sàn) / sàn.

        Gần 0 nghĩa là NMF đã khớp V gần bằng mức tốt nhất mà k chủ đề cho
        phép. Trả về None khi sàn bằng 0 (k đủ lớn để tái tạo trọn V), vì khi
        đó không chia được.
        """
        total = float(np.linalg.norm(self.V))
        if self.floor <= 1e-9 * max(total, 1.0):
            return None
        return (self.errors[-1] - self.floor) / self.floor

    def top_keywords(self, topic: int, n: int = 6) -> list[tuple[str, float]]:
        """Các từ nặng nhất của một chủ đề (cột `topic` của W), kèm trọng số.

        Chỉ lấy từ có trọng số từ 10% của từ đứng đầu trở lên, để không kéo
        theo những từ gần như không thuộc chủ đề.
        """
        column = self.W[:, topic]
        order = np.argsort(-column, kind="stable")[:n]
        return [(self.vocab[i], float(column[i])) for i in order if column[i] >= 0.1 * column.max()]

    def label_keywords(self, topic: int, n: int = 3, pool: int = 6) -> list[str]:
        """Vài từ khóa dùng để gọi tên một chủ đề trên giao diện.

        Luôn giữ từ nặng nhất của chủ đề. Các chỗ còn lại lấy trong `pool` từ
        nặng nhất, ưu tiên từ ghép (từ 2 âm tiết trở lên như "đồng_hồ",
        "thí_sinh") vì đứng riêng chúng rõ nghĩa hơn từ một âm tiết ("kỳ",
        "hòa"); không đủ từ ghép thì lấy thêm từ một âm tiết theo thứ tự trọng số.
        Đây chỉ là cách hiển thị: W, H và cách chọn câu không đổi.
        """
        words = [word for word, _ in self.top_keywords(topic, pool)]
        compound = [word for word in words[1:] if "_" in word]
        single = [word for word in words[1:] if "_" not in word]
        return [*words[:1], *compound, *single][:n]


def summarize(
    text: str,
    k: int = 3,
    n_sentences: int = 3,
    strategy: str = "coverage",
    min_df: int = 2,
    seeds: tuple[int, ...] = SEEDS,
) -> SummaryResult:
    """Tóm tắt văn bản thô. Báo TextTooShortError nếu văn bản quá ngắn.

    k           số chủ đề
    n_sentences số câu của bản tóm tắt
    strategy    cách chọn câu: "score" hoặc "coverage" (xem STRATEGIES)
    min_df      bỏ từ xuất hiện trong ít hơn min_df câu (đặt 1 để giữ mọi từ)
    seeds       các điểm khởi tạo NMF sẽ thử; giữ lần có sai số nhỏ nhất
    """
    return summarize_processed(preprocess(text), k, n_sentences, strategy, min_df, seeds)


def summarize_processed(
    doc: ProcessedText,
    k: int = 3,
    n_sentences: int = 3,
    strategy: str = "coverage",
    min_df: int = 2,
    seeds: tuple[int, ...] = SEEDS,
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
    parts = tfidf_parts(tokens)
    if parts.V.sum() == 0:
        raise TextTooShortError(
            "Không tìm thấy từ khóa nào giúp phân biệt các câu "
            "(các câu quá giống nhau hoặc chỉ gồm từ dừng)."
        )
    V = normalize_columns(parts.V)

    k = max(1, min(k, n - 1, len(parts.vocab)))  # không thể có nhiều chủ đề hơn số câu
    result, seed = nmf_best_of(V, k, seeds, MAX_ITER, TOL)
    W, H = normalize_topics(result.W, result.H)
    topic_weights, sentence_scores, selected = choose_sentences(H, strategy, n_sentences)

    has_content = V.sum(axis=0) > 0
    dominant_topic = np.where(has_content, H.argmax(axis=0), -1)

    return SummaryResult(
        sentences=doc.sentences,
        tokens=tokens,
        vocab=parts.vocab,
        V=V,
        W=W,
        H=H,
        errors=result.errors,
        seed=seed,
        floor=best_rank_k_error(V, k),
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


def choose_sentences(
    H: np.ndarray, strategy: str, n_sentences: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Từ H: trả về (trọng số chủ đề, điểm câu, các câu được chọn)."""
    topic_weights = H.sum(axis=1) / H.sum()
    sentence_scores = topic_weights @ H
    if strategy == "score":
        selected = select_top(sentence_scores, n_sentences)
    else:
        selected = select_by_coverage(H, topic_weights, n_sentences)
    return topic_weights, sentence_scores, selected


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
    row = result.H[topic]
    scores = np.where(row > 0, row, -1.0)
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
# Phần thêm: đối chiếu với scikit-learn trên chính văn bản đang tóm tắt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SklearnCheck:
    """Kết quả so NMF tự viết với NMF của scikit-learn trên cùng một bài."""

    n_iter: int  # số vòng lặp mà cả hai cùng chạy
    error_own: float  # sai số của bản tự viết
    error_sklearn: float  # sai số của scikit-learn
    max_diff_W: float  # ô lệch nhiều nhất giữa hai ma trận W
    max_diff_H: float  # ô lệch nhiều nhất giữa hai ma trận H
    selected_sklearn: list[int]  # các câu được chọn nếu dùng W, H của scikit-learn
    same_summary: bool  # hai bản tóm tắt có trùng nhau không


def compare_with_sklearn(
    result: SummaryResult, n_sentences: int, strategy: str = "coverage"
) -> SklearnCheck:
    """Chạy NMF của scikit-learn trên đúng bài này rồi so với bản tự viết.

    Hai bên dùng cùng ma trận V, cùng điểm khởi tạo (seed của lần chạy được
    giữ lại) và cùng số vòng lặp, nên nếu bản tự viết cài đúng công thức thì
    W, H phải gần như trùng và hai bản tóm tắt phải giống nhau.
    Báo ImportError nếu máy chưa cài scikit-learn.
    """
    own = nmf(result.V, result.k, seed=result.seed, max_iter=MAX_ITER, tol=TOL)
    W0, H0 = init_factors(result.V, result.k, result.seed)
    W_sk, H_sk, error_sk = sklearn_reference(result.V, W0, H0, own.n_iter)

    _, H_sk_normalized = normalize_topics(W_sk, H_sk)
    _, _, selected_sk = choose_sentences(H_sk_normalized, strategy, n_sentences)
    return SklearnCheck(
        n_iter=own.n_iter,
        error_own=own.errors[-1],
        error_sklearn=error_sk,
        max_diff_W=float(np.abs(own.W - W_sk).max()),
        max_diff_H=float(np.abs(own.H - H_sk).max()),
        selected_sklearn=selected_sk,
        same_summary=selected_sk == result.selected,
    )

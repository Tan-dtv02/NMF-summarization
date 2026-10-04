"""NMF tự viết bằng NumPy và cách dựng ma trận V từ các câu.

Ký hiệu theo slide chương 4 (mục "Diễn giải NMF"):

    V (m x n): ma trận từ - câu. m từ (hàng), n câu (cột), trọng số TF-IDF.
    W (m x k): mỗi CỘT là một chủ đề, cho biết từ nào quan trọng với chủ đề đó.
    H (k x n): mỗi CỘT ứng với một câu, cho biết câu đó chứa mỗi chủ đề bao nhiêu.
    k        : số chủ đề, do người dùng chọn.

Mục tiêu: tìm W và H không âm sao cho W.H xấp xỉ V tốt nhất, tức là làm nhỏ
sai số Frobenius ||V - W.H||.

Cách giải là quy tắc cập nhật nhân tỉ lệ (multiplicative update) của Lee và
Seung. Mỗi vòng lặp làm hai việc, theo đúng thứ tự này:

    H <- H * (W^T V) / (W^T W H)
    W <- W * (V H^T) / (W H H^T)

Phép nhân * và phép chia / ở đây là theo TỪNG Ô. Vì chỉ nhân và chia các số
không âm nên W và H luôn không âm, không cần ép thêm điều kiện nào.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

EPS = 1e-9  # cộng vào mẫu số để không bao giờ chia cho 0


# ---------------------------------------------------------------------------
# Phần A. Từ các câu đã tách từ đến ma trận V
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TfidfParts:
    """Mọi bước trung gian khi dựng V, giữ lại để màn mô phỏng hiển thị."""

    vocab: list[str]  # m từ, theo thứ tự xuất hiện lần đầu
    counts: np.ndarray  # (m x n) số lần từ i xuất hiện trong câu j
    tf: np.ndarray  # (m x n) counts chia cho số từ của câu j
    df: np.ndarray  # (m,)   số câu có chứa từ i
    idf: np.ndarray  # (m,)   ln(n / df)
    V: np.ndarray  # (m x n) tf * idf


def build_vocab(tokens: list[list[str]]) -> list[str]:
    """Danh sách từ không trùng lặp, xếp theo thứ tự xuất hiện lần đầu."""
    return list(dict.fromkeys(word for sentence in tokens for word in sentence))


def tfidf_parts(tokens: list[list[str]]) -> TfidfParts:
    """Dựng ma trận V (từ x câu) theo công thức TF-IDF trong slide.

        TF(t, d)  = số lần t xuất hiện trong câu d / số từ của câu d
        IDF(t)    = ln(N / df(t)), N là số câu, df(t) là số câu chứa t
        V[t, d]   = TF(t, d) * IDF(t)

    Ở đây mỗi CÂU đóng vai một "tài liệu". Từ nào xuất hiện ở mọi câu thì
    IDF = ln(1) = 0, tức là không giúp phân biệt câu nào với câu nào.
    """
    vocab = build_vocab(tokens)
    index = {word: i for i, word in enumerate(vocab)}
    m, n = len(vocab), len(tokens)

    counts = np.zeros((m, n))
    for j, sentence in enumerate(tokens):
        for word in sentence:
            counts[index[word], j] += 1

    lengths = counts.sum(axis=0)  # số từ của từng câu
    tf = counts / np.maximum(lengths, 1)  # câu rỗng: chia cho 1 để ra cột 0
    df = (counts > 0).sum(axis=1)
    idf = np.log(n / np.maximum(df, 1))
    return TfidfParts(vocab, counts, tf, df, idf, tf * idf[:, None])


def build_tfidf_matrix(tokens: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    """Trả về (V, vocab). Dùng khi không cần xem các bước trung gian."""
    parts = tfidf_parts(tokens)
    return parts.V, parts.vocab


# ---------------------------------------------------------------------------
# Phần B. Thuật toán NMF
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NMFStep:
    """Trạng thái sau một vòng lặp. Vòng 0 là trạng thái khởi tạo ngẫu nhiên."""

    iteration: int
    W: np.ndarray
    H: np.ndarray
    error: float


@dataclass(frozen=True)
class NMFResult:
    """Kết quả cuối cùng của NMF."""

    W: np.ndarray
    H: np.ndarray
    errors: list[float]  # errors[i] là sai số sau vòng i (errors[0]: lúc khởi tạo)
    n_iter: int  # số vòng đã chạy
    converged: bool  # True nếu dừng vì sai số gần như hết giảm


def init_factors(V: np.ndarray, k: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Khởi tạo W, H bằng số dương ngẫu nhiên. Cùng seed thì cùng kết quả.

    Nhân với sqrt(trung bình của V / k) để tích W.H ngay từ đầu có độ lớn
    cùng cỡ với V, nhờ đó thuật toán hội tụ nhanh hơn.
    """
    m, n = V.shape
    rng = np.random.default_rng(seed)
    scale = np.sqrt(V.mean() / k)
    W = scale * rng.random((m, k)) + EPS
    H = scale * rng.random((k, n)) + EPS
    return W, H


def update_H(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """H <- H * (W^T V) / (W^T W H), tính theo từng ô."""
    return H * (W.T @ V) / (W.T @ W @ H + EPS)


def update_W(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """W <- W * (V H^T) / (W H H^T), tính theo từng ô."""
    return W * (V @ H.T) / (W @ H @ H.T + EPS)


def nmf_step(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Một vòng lặp: cập nhật H trước, rồi dùng H mới để cập nhật W."""
    H = update_H(V, W, H)
    W = update_W(V, W, H)
    return W, H


def reconstruction_error(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """Sai số Frobenius ||V - W.H||: căn của tổng bình phương độ lệch từng ô."""
    return float(np.linalg.norm(V - W @ H))


def has_converged(previous: float, current: float, first: float, tol: float) -> bool:
    """Sai số gần như hết giảm: mức giảm vòng này < tol lần sai số ban đầu."""
    return first == 0 or (previous - current) / first < tol


def nmf_steps(
    V: np.ndarray,
    k: int,
    seed: int = 42,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> Iterator[NMFStep]:
    """Chạy NMF TỪNG VÒNG MỘT: mỗi lần lấy ra một NMFStep.

    Dừng khi một trong hai điều xảy ra trước:
      - đã chạy đủ max_iter vòng;
      - mức giảm sai số của vòng vừa rồi nhỏ hơn tol lần sai số ban đầu
        (thuật toán gần như đứng yên, chạy tiếp cũng không cải thiện đáng kể).

    Ví dụ:
        for step in nmf_steps(V, k=2):
            print(step.iteration, step.error)
    """
    W, H = init_factors(V, k, seed)
    first_error = previous_error = reconstruction_error(V, W, H)
    yield NMFStep(0, W, H, first_error)

    for iteration in range(1, max_iter + 1):
        W, H = nmf_step(V, W, H)
        error = reconstruction_error(V, W, H)
        yield NMFStep(iteration, W, H, error)
        if has_converged(previous_error, error, first_error, tol):
            return
        previous_error = error


def nmf(
    V: np.ndarray,
    k: int,
    seed: int = 42,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> NMFResult:
    """Chạy NMF MỘT MẠCH đến khi dừng, trả về W, H và sai số sau mỗi vòng."""
    errors = []
    for step in nmf_steps(V, k, seed, max_iter, tol):
        errors.append(step.error)
    return NMFResult(
        W=step.W,
        H=step.H,
        errors=errors,
        n_iter=step.iteration,
        converged=len(errors) > 1 and has_converged(errors[-2], errors[-1], errors[0], tol),
    )


def normalize_topics(W: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chuẩn hóa để mỗi cột của W có tổng bằng 1, tích W.H giữ nguyên.

    NMF không quy định độ lớn riêng của W và H: nhân một cột của W với 10 rồi
    chia hàng tương ứng của H cho 10 thì W.H vẫn y như cũ. Muốn so sánh các
    chủ đề với nhau qua H thì phải đưa W về cùng một thang đo trước.
    Sau bước này, mỗi cột W đọc được như "tỉ lệ các từ trong chủ đề".
    """
    column_sums = W.sum(axis=0)
    column_sums = np.where(column_sums > 0, column_sums, 1.0)
    return W / column_sums, H * column_sums[:, None]


# ---------------------------------------------------------------------------
# Phần C. Đối chiếu với scikit-learn (chỉ để kiểm tra, không dùng khi tóm tắt)
# ---------------------------------------------------------------------------


def sklearn_reference(
    V: np.ndarray, W0: np.ndarray, H0: np.ndarray, n_iter: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Chạy NMF của scikit-learn từ CÙNG điểm khởi tạo, CÙNG số vòng lặp.

    Chú ý về chiều ma trận: scikit-learn đặt tài liệu theo HÀNG, còn slide đặt
    câu theo CỘT. Vì vậy phải đưa vào V^T, và kết quả của thư viện là:

        fit_transform(V^T)  ->  H^T  (theo ký hiệu slide)
        components_         ->  W^T  (theo ký hiệu slide)

    Hàm này đã chuyển vị lại, nên W, H trả về so sánh trực tiếp được với
    kết quả của bản tự viết.
    """
    from sklearn.decomposition import NMF
    from sklearn.exceptions import ConvergenceWarning

    model = NMF(
        n_components=W0.shape[1],
        init="custom",  # dùng đúng W0, H0 của mình thay vì để thư viện tự khởi tạo
        solver="mu",  # multiplicative update, cùng quy tắc với bản tự viết
        beta_loss="frobenius",
        max_iter=n_iter,
        tol=0,  # tắt điều kiện dừng sớm để chạy đủ n_iter vòng
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        H_t = model.fit_transform(
            np.ascontiguousarray(V.T),
            W=np.ascontiguousarray(H0.T),
            H=np.ascontiguousarray(W0.T),
        )
    return model.components_.T, H_t.T, float(model.reconstruction_err_)

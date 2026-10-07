"""Kiểm tra NMF tự viết: đúng công thức, đúng tính chất, khớp scikit-learn."""

import numpy as np
import pytest

from src.nmf_scratch import (
    best_rank_k_error,
    init_factors,
    nmf,
    nmf_best_of,
    nmf_step,
    nmf_steps,
    normalize_columns,
    normalize_topics,
    reconstruction_error,
    sklearn_reference,
    tfidf_parts,
)

# Ví dụ nhỏ của màn mô phỏng, sau khi tách từ và bỏ từ dừng.
MINI = [
    ["pin", "camera", "điện_thoại"],  # Pin và camera của điện thoại
    ["camera", "điện_thoại"],  # Camera của điện thoại
    ["phở", "bún"],  # Phở và bún
    ["bún", "phở"],  # Bún và phở
]


@pytest.fixture
def V():
    return tfidf_parts(MINI).V


def test_tfidf_khop_tinh_tay():
    """V[t, d] = (số lần t trong câu d / số từ của câu d) * ln(số câu / số câu chứa t)."""
    parts = tfidf_parts(MINI)
    assert parts.vocab == ["pin", "camera", "điện_thoại", "phở", "bún"]
    assert parts.V.shape == (5, 4)  # 5 từ x 4 câu

    ln2, ln4 = np.log(2), np.log(4)
    expected = np.array(
        [
            [ln4 / 3, 0, 0, 0],  # pin: chỉ ở câu 1 (df = 1), câu 1 có 3 từ
            [ln2 / 3, ln2 / 2, 0, 0],  # camera: ở câu 1 và câu 2 (df = 2)
            [ln2 / 3, ln2 / 2, 0, 0],  # điện_thoại
            [0, 0, ln2 / 2, ln2 / 2],  # phở
            [0, 0, ln2 / 2, ln2 / 2],  # bún
        ]
    )
    np.testing.assert_allclose(parts.V, expected)


def test_tu_co_o_moi_cau_thi_idf_bang_0():
    parts = tfidf_parts([["a", "x"], ["a", "y"], ["a", "z"]])
    assert parts.idf[parts.vocab.index("a")] == 0


def test_mot_vong_khop_cong_thuc_viet_bang_vong_lap(V):
    """Tính lại một vòng cập nhật theo từng ô, không dùng phép nhân ma trận gộp."""
    W0, H0 = init_factors(V, k=2, seed=0)
    W1, H1 = nmf_step(V, W0, H0)

    m, n = V.shape
    k = 2
    H_tay = np.zeros((k, n))
    for a in range(k):
        for j in range(n):
            tu = sum(W0[i, a] * V[i, j] for i in range(m))  # (W^T V)[a, j]
            mau = sum(W0[i, a] * (W0 @ H0)[i, j] for i in range(m))  # (W^T W H)[a, j]
            H_tay[a, j] = H0[a, j] * tu / mau
    W_tay = np.zeros((m, k))
    for i in range(m):
        for a in range(k):
            tu = sum(V[i, j] * H_tay[a, j] for j in range(n))  # (V H^T)[i, a]
            mau = sum((W0 @ H_tay)[i, j] * H_tay[a, j] for j in range(n))  # (W H H^T)[i, a]
            W_tay[i, a] = W0[i, a] * tu / mau

    np.testing.assert_allclose(H1, H_tay, rtol=1e-6)
    np.testing.assert_allclose(W1, W_tay, rtol=1e-6)


def test_w_h_luon_khong_am_va_sai_so_khong_tang(V):
    result = nmf(V, k=2, seed=42)
    assert (result.W >= 0).all() and (result.H >= 0).all()
    assert len(result.errors) == result.n_iter + 1  # tính cả sai số lúc khởi tạo
    for truoc, sau in zip(result.errors, result.errors[1:]):
        assert sau <= truoc + 1e-12


def test_cung_seed_thi_cung_ket_qua(V):
    a, b = nmf(V, k=2, seed=7), nmf(V, k=2, seed=7)
    np.testing.assert_array_equal(a.W, b.W)
    np.testing.assert_array_equal(a.H, b.H)


def test_chay_tung_vong_giong_chay_mot_mach(V):
    steps = list(nmf_steps(V, k=2, seed=42))
    result = nmf(V, k=2, seed=42)
    assert steps[0].iteration == 0
    assert [s.error for s in steps] == result.errors
    np.testing.assert_array_equal(steps[-1].W, result.W)


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("n_iter", [1, 10, 50])
def test_khop_scikit_learn(V, seed, n_iter):
    """Cùng điểm khởi tạo, cùng số vòng thì bản tự viết phải ra gần như đúng
    các con số của sklearn NMF(solver="mu")."""
    W0, H0 = init_factors(V, k=2, seed=seed)
    W, H = W0, H0
    for _ in range(n_iter):
        W, H = nmf_step(V, W, H)

    W_sk, H_sk, error_sk = sklearn_reference(V, W0, H0, n_iter)

    np.testing.assert_allclose(W, W_sk, atol=1e-5)
    np.testing.assert_allclose(H, H_sk, atol=1e-5)
    assert reconstruction_error(V, W, H) == pytest.approx(error_sk, abs=1e-6)


def test_khop_scikit_learn_tren_ma_tran_ngau_nhien():
    rng = np.random.default_rng(0)
    V = rng.random((30, 12))
    W0, H0 = init_factors(V, k=4, seed=3)
    W, H = W0, H0
    for _ in range(100):
        W, H = nmf_step(V, W, H)
    W_sk, H_sk, _ = sklearn_reference(V, W0, H0, 100)
    np.testing.assert_allclose(W, W_sk, atol=1e-5)
    np.testing.assert_allclose(H, H_sk, atol=1e-5)


@pytest.mark.parametrize("seed", range(10))
def test_vi_du_nho_tach_dung_hai_chu_de(V, seed):
    """Dù khởi tạo thế nào, hai câu điện thoại về một chủ đề, hai câu ẩm thực về chủ đề kia."""
    result = nmf(V, k=2, seed=seed)
    chu_de = result.H.argmax(axis=0)
    assert chu_de[0] == chu_de[1]
    assert chu_de[2] == chu_de[3]
    assert chu_de[0] != chu_de[2]


def test_chuan_hoa_giu_nguyen_tich(V):
    result = nmf(V, k=2)
    W, H = normalize_topics(result.W, result.H)
    np.testing.assert_allclose(W.sum(axis=0), 1.0)
    np.testing.assert_allclose(W @ H, result.W @ result.H)


def test_vi_du_man_1_khong_doi(V):
    """Ví dụ mặc định của màn mô phỏng: dừng sau 8 vòng, sai số 0,339."""
    result = nmf(V, k=2, seed=42)
    assert result.n_iter == 8
    assert round(result.errors[-1], 3) == 0.339


# --- Ba hàm phụ của phần tóm tắt -----------------------------------------------


def test_chuan_hoa_do_dai_tung_cot():
    V = np.array([[3.0, 0.0, 1.0], [4.0, 0.0, 1.0]])
    chuan = normalize_columns(V)
    np.testing.assert_allclose(chuan[:, 0], [0.6, 0.8])  # (3, 4) dài 5
    np.testing.assert_allclose(chuan[:, 1], [0.0, 0.0])  # cột toàn 0 giữ nguyên
    np.testing.assert_allclose(np.linalg.norm(chuan, axis=0), [1.0, 0.0, 1.0])
    np.testing.assert_allclose(V, [[3.0, 0.0, 1.0], [4.0, 0.0, 1.0]])  # không sửa V gốc


def test_chay_nhieu_seed_giu_sai_so_nho_nhat(V):
    seeds = (0, 1, 2, 3, 4)
    tot_nhat, seed = nmf_best_of(V, k=2, seeds=seeds)
    tung_lan = [nmf(V, k=2, seed=s, max_iter=2000, tol=1e-6) for s in seeds]
    assert tot_nhat.errors[-1] == min(r.errors[-1] for r in tung_lan)
    assert seed in seeds
    lan_hai, seed_hai = nmf_best_of(V, k=2, seeds=seeds)  # chạy lại ra đúng kết quả cũ
    assert seed_hai == seed
    np.testing.assert_array_equal(lan_hai.W, tot_nhat.W)


def test_san_cua_xap_xi_hang_k(V):
    """Không phép xấp xỉ hạng k nào khớp V tốt hơn mức sàn tính bằng SVD."""
    for k in (1, 2, 3):
        san = best_rank_k_error(V, k)
        assert nmf(V, k, seed=0, max_iter=2000, tol=1e-8).errors[-1] >= san - 1e-9
    assert best_rank_k_error(V, 1) > best_rank_k_error(V, 2) > best_rank_k_error(V, 3)
    assert best_rank_k_error(V, min(V.shape)) == pytest.approx(0.0, abs=1e-9)
    # ma trận hạng 1 thì xấp xỉ hạng 1 không còn sai số
    assert best_rank_k_error(np.outer([1.0, 2.0, 3.0], [1.0, 0.5]), 1) == pytest.approx(0.0, abs=1e-9)


# --- TF-IDF tự viết so với scikit-learn ------------------------------------------

CAU_THU = [
    ["pin", "pin", "camera", "điện_thoại"],
    ["camera", "điện_thoại", "giá"],
    ["phở", "bún", "giá", "giá"],
    ["bún", "phở"],
    ["giá", "pin"],
]


def test_tfidf_khop_scikit_learn_sau_khi_quy_doi():
    """Hai công thức khác nhau ở hai chỗ đã biết, quy đổi xong thì phải trùng:

    scikit-learn (smooth_idf=False, norm=None):  số lần đếm * (ln(N / df) + 1)
    slide và bản tự viết                      :  (số lần đếm / số từ của câu) * ln(N / df)

    Tức là bỏ phần "+1" (trừ đi số lần đếm) rồi chia cho số từ của câu.
    scikit-learn xếp câu theo hàng nên phải chuyển vị ở hai đầu.
    """
    from sklearn.feature_extraction.text import TfidfTransformer

    parts = tfidf_parts(CAU_THU)
    thu_vien = TfidfTransformer(norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False)
    cua_sklearn = thu_vien.fit_transform(parts.counts.T).toarray().T
    so_tu_moi_cau = parts.counts.sum(axis=0)

    np.testing.assert_allclose(thu_vien.idf_ - 1, parts.idf)  # IDF chỉ lệch nhau đúng 1
    np.testing.assert_allclose((cua_sklearn - parts.counts) / so_tu_moi_cau, parts.V)


def test_chuan_hoa_l2_khop_scikit_learn():
    """Bước chuẩn hóa độ dài câu cho đúng kết quả của hàm normalize trong thư viện."""
    from sklearn.preprocessing import normalize

    V = tfidf_parts(CAU_THU).V
    np.testing.assert_allclose(normalize_columns(V), normalize(V.T, norm="l2").T)

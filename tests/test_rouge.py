"""Kiểm tra ROUGE tự viết trên các ví dụ tính tay được."""

import pytest

from src.evaluate import lead_n, random_n, rouge_all, rouge_l, rouge_n, rouge_tokens

THAM_CHIEU = "Con mèo ngồi trên thảm."  # 5 âm tiết: con, mèo, ngồi, trên, thảm
MAY_TAO = "Con mèo nằm trên thảm"  # khác đúng một âm tiết: nằm thay cho ngồi


def test_giu_nguyen_chu_co_dau():
    """Lỗi của thư viện rouge-score: "điện thoại" bị cắt thành i, n, tho, i."""
    assert rouge_tokens("Điện thoại có pin trâu, camera ĐẸP!") == [
        "điện", "thoại", "có", "pin", "trâu", "camera", "đẹp",
    ]  # fmt: skip


def test_rouge_1_tinh_tay():
    # Trùng 4 âm tiết (con, mèo, trên, thảm) trên 5 âm tiết mỗi bên.
    score = rouge_n(MAY_TAO, THAM_CHIEU, 1)
    assert score["p"] == pytest.approx(4 / 5)
    assert score["r"] == pytest.approx(4 / 5)
    assert score["f"] == pytest.approx(4 / 5)


def test_rouge_2_tinh_tay():
    # Mỗi bên có 4 cặp liền nhau; trùng 2 cặp: (con, mèo) và (trên, thảm).
    score = rouge_n(MAY_TAO, THAM_CHIEU, 2)
    assert score["p"] == pytest.approx(2 / 4)
    assert score["r"] == pytest.approx(2 / 4)


def test_rouge_l_tinh_tay():
    # Dãy con chung dài nhất: con, mèo, trên, thảm (dài 4).
    score = rouge_l(MAY_TAO, THAM_CHIEU)
    assert score["f"] == pytest.approx(4 / 5)


def test_precision_va_recall_khac_nhau_khi_do_dai_khac_nhau():
    # Bản máy tạo 2 âm tiết, đều có trong bản tham chiếu 5 âm tiết.
    score = rouge_n("con mèo", THAM_CHIEU, 1)
    assert score["p"] == pytest.approx(1.0)
    assert score["r"] == pytest.approx(2 / 5)
    assert score["f"] == pytest.approx(2 * 1.0 * 0.4 / 1.4)


def test_tu_lap_lai_chi_tinh_so_lan_co_o_ca_hai_ben():
    # "pin" có 3 lần ở bản máy tạo nhưng chỉ 1 lần ở tham chiếu: tính trùng 1.
    score = rouge_n("pin pin pin", "pin tốt", 1)
    assert score["p"] == pytest.approx(1 / 3)
    assert score["r"] == pytest.approx(1 / 2)


def test_rouge_l_giu_thu_tu():
    # Cùng các âm tiết nhưng đảo thứ tự: ROUGE-1 vẫn tối đa, ROUGE-L thì không.
    assert rouge_n("thảm trên mèo", "mèo trên thảm", 1)["f"] == pytest.approx(1.0)
    assert rouge_l("thảm trên mèo", "mèo trên thảm")["f"] == pytest.approx(1 / 3)


def test_hai_ban_giong_het_va_khac_het():
    assert rouge_all(THAM_CHIEU, THAM_CHIEU)["rougeL_f"] == pytest.approx(1.0)
    assert rouge_all("phở bún", THAM_CHIEU)["rouge1_f"] == 0.0
    assert rouge_all("", THAM_CHIEU)["rouge2_f"] == 0.0  # không chia cho 0


def test_doi_chung():
    cau = ["a", "b", "c", "d", "e"]
    assert lead_n(cau, 3) == [0, 1, 2]
    assert random_n(cau, 3, seed=1) == random_n(cau, 3, seed=1)  # cùng seed, cùng kết quả
    assert len(set(random_n(cau, 3, seed=1))) == 3
    assert lead_n(cau, 10) == [0, 1, 2, 3, 4]

"""Kiểm tra ROUGE tự viết và các mốc so sánh trên các ví dụ tính tay được."""

import itertools

import pytest

from src.evaluate import (
    lead_n,
    oracle_label,
    oracle_n,
    random_n,
    rouge_all,
    rouge_l,
    rouge_n,
    rouge_tokens,
)

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


def test_muc_tran_tinh_tay():
    """Tham chiếu "con mèo đuổi chuột" (4 âm tiết), chọn 2 trong 3 câu."""
    cau = ["Con mèo ngủ.", "Trời mưa to.", "Mèo đuổi chuột."]
    # Lượt 1: câu 2 trùng 3 trên 3 âm tiết, F1 = 6/7; câu 0 trùng 2, F1 = 4/7; câu 1 trùng 0.
    # Lượt 2: thêm câu 0 thì trùng 4 trên 6 âm tiết (F1 = 0,8); thêm câu 1 thì vẫn trùng 3.
    assert oracle_n(cau, "Con mèo đuổi chuột", 1) == [2]
    assert oracle_n(cau, "Con mèo đuổi chuột", 2) == [0, 2]  # trả về theo thứ tự trong bài
    assert oracle_label(2) == "2 câu tốt nhất (mức trần)"


def test_muc_tran_lay_dung_so_cau():
    """Cùng số câu với các phương pháp khác, kể cả khi câu thêm vào làm điểm giảm."""
    cau = ["Con mèo đuổi chuột.", "Trời mưa to.", "Phở rất ngon."]
    assert oracle_n(cau, "Con mèo đuổi chuột", 1) == [0]
    assert len(oracle_n(cau, "Con mèo đuổi chuột", 2)) == 2
    assert oracle_n(cau, "Con mèo đuổi chuột", 10) == [0, 1, 2]  # bài không đủ câu thì lấy hết
    assert oracle_n(cau, "Con mèo đuổi chuột", 2) == oracle_n(cau, "Con mèo đuổi chuột", 2)


def test_muc_tran_khong_thua_bo_cau_nao_tren_vi_du_nho():
    """Chọn tham lam trùng với cách thử hết mọi bộ 2 câu trên ví dụ nhỏ này."""
    cau = ["Con mèo ngủ trên thảm.", "Con chó đuổi con mèo.", "Trời mưa to.", "Mèo bắt chuột."]
    tham_chieu = "Con mèo bắt chuột trên thảm"

    def diem(bo_cau) -> float:
        return rouge_n(" ".join(cau[j] for j in bo_cau), tham_chieu, 1)["f"]

    tot_nhat = max(diem(bo) for bo in itertools.combinations(range(len(cau)), 2))
    assert diem(oracle_n(cau, tham_chieu, 2)) == pytest.approx(tot_nhat)
    assert diem(oracle_n(cau, tham_chieu, 2)) >= diem(lead_n(cau, 2))

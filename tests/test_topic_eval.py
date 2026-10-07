"""Kiểm tra thử nghiệm "tách chủ đề": hai phép đo và một lần chạy nhỏ."""

import numpy as np
import pytest

from src.topic_eval import (
    combined_documents,
    covers_all_parts,
    demo_documents,
    evaluate_documents,
    sentence_accuracy,
    summarize_runs,
)


def test_cau_dung_nhom_khong_phu_thuoc_cach_danh_so_chu_de():
    nhan = np.array([0, 0, 0, 1, 1, 1])
    assert sentence_accuracy(np.array([0, 0, 0, 1, 1, 1]), nhan) == 1.0
    # NMF gọi mảng 0 là "chủ đề 1" và ngược lại: vẫn là tách đúng hoàn toàn.
    assert sentence_accuracy(np.array([1, 1, 1, 0, 0, 0]), nhan) == 1.0
    assert sentence_accuracy(np.array([0, 0, 1, 1, 1, 1]), nhan) == pytest.approx(5 / 6)


def test_cau_dung_nhom_tinh_cau_khong_con_tu_khoa_la_sai():
    nhan = np.array([0, 0, 1, 1])
    assert sentence_accuracy(np.array([0, -1, 1, 1]), nhan) == pytest.approx(3 / 4)
    # Dồn hết câu về một chủ đề: mỗi mảng chỉ được ghép với một chủ đề nên chỉ đúng một nửa.
    assert sentence_accuracy(np.array([0, 0, 0, 0]), nhan) == pytest.approx(1 / 2)


def test_phu_du_mang():
    nhan = np.array([0, 0, 1, 1, 2, 2])
    assert covers_all_parts([0, 2, 5], nhan)
    assert not covers_all_parts([0, 1, 2], nhan)  # thiếu mảng 2
    assert covers_all_parts([0, 1, 2, 3, 4], nhan)


def test_so_bai_ghep_va_nhan():
    """6 bài ngắn cho 15 cặp, 20 bộ ba, 15 bộ bốn; nhãn đánh theo bài gốc."""
    bai_ghep = combined_documents()
    assert [sum(d.n_parts == size for d in bai_ghep) for size in (2, 3, 4)] == [15, 20, 15]
    for bai in bai_ghep:
        assert len(bai.labels) == len(bai.doc.sentences) == len(bai.doc.tokens)
        assert list(bai.labels) == sorted(bai.labels)  # các bài gốc nối tiếp nhau


def test_nhan_cua_bai_demo_theo_doan_van():
    bai_demo = demo_documents()
    assert [d.n_parts for d in bai_demo] == [3, 3, 3, 3, 2]
    for bai in bai_demo:
        assert len(bai.labels) == len(bai.doc.sentences)


def test_chay_thu_tren_bai_ghep_2_mang():
    """Một lần chạy nhỏ: đủ cột, đủ dòng, và pipeline mới không kém pipeline cũ."""
    luot_chay = evaluate_documents(combined_documents(sizes=(2,)))
    assert len(luot_chay) == 15 * 2 * 2  # 15 bài x 2 pipeline x 2 chế độ lọc từ hiếm
    cot_do = [c for c in luot_chay.columns if c.startswith("phu_du_")]
    assert sorted(cot_do) == [
        "phu_du_diem_tong_2k_cau", "phu_du_diem_tong_k_cau",
        "phu_du_phu_chu_de_2k_cau", "phu_du_phu_chu_de_k_cau",
    ]  # fmt: skip
    assert luot_chay[["cau_dung_nhom", *cot_do]].apply(lambda cot: cot.between(0, 1)).all().all()

    bang = summarize_runs(luot_chay).set_index(["loc_tu_hiem", "pipeline"])
    assert (bang["so_van_ban"] == 15).all()
    for loc in ("bật", "tắt"):
        cu, moi = bang.loc[(loc, "cũ")], bang.loc[(loc, "mới")]
        assert moi["cau_dung_nhom"] >= cu["cau_dung_nhom"]
        assert moi["phu_du_phu_chu_de_k_cau"] >= cu["phu_du_phu_chu_de_k_cau"]
        assert moi["phu_du_phu_chu_de_k_cau"] == 1.0  # Phủ chủ đề: luôn đủ cả hai mảng

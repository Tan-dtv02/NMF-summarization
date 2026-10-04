"""Kiểm tra tiền xử lý, tóm tắt và phần thử nghiệm trên dữ liệu mẫu."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import SAMPLE_DIR, evaluate_dataset, is_sample_data, load_dataset
from src.preprocess import filter_rare_words, load_stopwords, preprocess, split_sentences
from src.summarizer import (
    TextTooShortError,
    error_by_k,
    suggest_k,
    summarize,
    summarize_by_keyword,
    summarize_by_topic,
)

BAI_MAU = SAMPLE_DIR / "bai_mau.csv"


@pytest.fixture(scope="module")
def bai_dien_thoai() -> str:
    return load_dataset(BAI_MAU)["text"][0]


# --- Tiền xử lý -------------------------------------------------------------


def test_tach_cau():
    cau = split_sentences("Trời mưa to. Đường ngập!\nXe có đi được không? Có.")
    assert cau == ["Trời mưa to.", "Đường ngập!", "Xe có đi được không?", "Có."]


def test_bo_tu_dung_va_dau_cau():
    doc = preprocess("Pin của máy rất tốt, và giá là 7,9 triệu!")
    tu = doc.tokens[0]
    assert "pin" in tu and "giá" in tu
    assert not {"của", "và", "là", "rất"} & set(tu)  # từ dừng đã bị bỏ
    assert all(any(c.isalpha() for c in t) for t in tu)  # không còn số, dấu câu


def test_danh_sach_tu_dung_doc_duoc():
    tu_dung = load_stopwords()
    assert len(tu_dung) > 1500  # kho công khai đã được nạp
    assert {"và", "của", "tuy_nhiên", "lớn", "giờ"} <= tu_dung  # từ của kho công khai
    assert "kéo_dài" in tu_dung  # từ nhóm thêm
    assert not {"nước", "bán"} & tu_dung  # từ nhóm giữ lại bằng dấu "!"


def test_dau_cham_than_la_ngoai_le(tmp_path):
    kho = tmp_path / "kho.txt"
    kho.write_text("và\nnước\ntuy nhiên\n", encoding="utf-8")
    nhom = tmp_path / "nhom.txt"
    nhom.write_text("# ghi chú\n!nước\nkéo dài\n", encoding="utf-8")
    assert load_stopwords(kho, nhom) == {"và", "tuy_nhiên", "kéo_dài"}
    assert load_stopwords(kho, tmp_path / "khong_co.txt") == {"và", "nước", "tuy_nhiên"}


def test_loc_tu_hiem():
    assert filter_rare_words([["a", "b"], ["a", "c"], ["c", "c"]]) == [
        ["a"],
        ["a", "c"],
        ["c", "c"],
    ]


# --- Tóm tắt ----------------------------------------------------------------


def test_tom_tat_dung_so_cau_va_giu_thu_tu_goc(bai_dien_thoai):
    kq = summarize(bai_dien_thoai, k=3, n_sentences=3)
    assert len(kq.selected) == 3
    assert kq.selected == sorted(kq.selected)
    assert all(kq.sentences[j] in bai_dien_thoai for j in kq.selected)  # trích nguyên văn
    assert kq.summary == " ".join(kq.sentences[j] for j in kq.selected)


def test_kich_thuoc_ma_tran_theo_quy_uoc_slide(bai_dien_thoai):
    kq = summarize(bai_dien_thoai, k=3, n_sentences=3)
    m, n = len(kq.vocab), len(kq.sentences)
    assert kq.V.shape == (m, n)  # từ x câu
    assert kq.W.shape == (m, 3)  # từ x chủ đề
    assert kq.H.shape == (3, n)  # chủ đề x câu
    assert kq.topic_weights.sum() == pytest.approx(1.0)
    assert len(kq.sentence_scores) == n and len(kq.dominant_topic) == n


def test_cau_ve_camera_cung_mot_chu_de(bai_dien_thoai):
    """Bài mẫu có 4 câu liền nhau nói về camera (câu 5 đến 8)."""
    kq = summarize(bai_dien_thoai, k=3, n_sentences=3)
    assert len(set(kq.dominant_topic[4:8])) == 1
    chu_de = kq.dominant_topic[4]
    assert "camera" in [tu for tu, _ in kq.top_keywords(chu_de, 3)]


def test_hai_cach_chon_cau_deu_chay(bai_dien_thoai):
    for cach in ("score", "coverage"):
        assert len(summarize(bai_dien_thoai, 3, 3, strategy=cach).selected) == 3


def test_cung_dau_vao_thi_cung_ket_qua(bai_dien_thoai):
    assert summarize(bai_dien_thoai, 3, 3).selected == summarize(bai_dien_thoai, 3, 3).selected


@pytest.mark.parametrize(
    "van_ban",
    ["", "   \n ", "Chỉ có một câu.", "Câu một. Câu hai.", "Pin tốt. Camera đẹp. Giá rẻ."],
)
def test_van_ban_qua_ngan_bao_loi_ro_rang(van_ban):
    with pytest.raises(TextTooShortError) as loi:
        summarize(van_ban, k=3, n_sentences=3)
    assert str(loi.value)  # có thông báo tiếng Việt để hiện cho người dùng


def test_k_lon_hon_so_cau_thi_tu_giam():
    kq = summarize("Pin tốt lắm. Pin bền. Camera đẹp. Camera nét.", k=10, n_sentences=2)
    assert kq.k <= 3


# --- Phần thêm --------------------------------------------------------------


def test_tom_tat_theo_chu_de(bai_dien_thoai):
    kq = summarize(bai_dien_thoai, 3, 3)
    for chu_de in range(kq.k):
        chon = summarize_by_topic(kq, chu_de, 2)
        assert 1 <= len(chon) <= 2
        assert all(kq.H[chu_de, j] > 0 for j in chon)


def test_tom_tat_theo_tu_khoa(bai_dien_thoai):
    kq = summarize(bai_dien_thoai, 3, 3)
    chon, ti_le = summarize_by_keyword(kq, "pin", 2)
    assert ti_le.sum() == pytest.approx(1.0)
    assert all("pin" in kq.sentences[j].lower() for j in chon)
    assert summarize_by_keyword(kq, "bóng đá", 2) is None  # bài không nhắc tới
    assert summarize_by_keyword(kq, "  ", 2) is None


def test_goi_y_k(bai_dien_thoai):
    kq = summarize(bai_dien_thoai, 3, 3)
    ks = [2, 3, 4, 5, 6]
    sai_so = error_by_k(kq.V, ks)
    assert all(b <= a + 1e-6 for a, b in zip(sai_so, sai_so[1:]))  # k tăng thì sai số giảm
    assert suggest_k(ks, sai_so) in ks
    assert suggest_k([2, 3, 4, 5], [10.0, 2.0, 1.5, 1.0]) == 3  # đường gập rõ ở k = 3


# --- Thử nghiệm -------------------------------------------------------------


def test_du_lieu_mau_dung_dinh_dang():
    df = load_dataset(BAI_MAU)
    assert list(df.columns) == ["text", "summary"]
    assert len(df) >= 5
    assert is_sample_data(BAI_MAU)


def test_bao_loi_khi_thieu_cot(tmp_path):
    tep = tmp_path / "sai.csv"
    pd.DataFrame({"noi_dung": ["a"], "tom_tat": ["b"]}).to_csv(tep, index=False)
    with pytest.raises(ValueError, match="thiếu cột"):
        load_dataset(tep)
    assert len(load_dataset(tep, text_col="noi_dung", summary_col="tom_tat")) == 1


def test_chay_thu_nghiem_tren_du_lieu_mau():
    df = load_dataset(BAI_MAU)
    bang, so_bai = evaluate_dataset(df, k_values=(2, 3), n_sentences=3)
    assert so_bai == len(df)
    assert list(bang["phuong_phap"]) == ["NMF (k=2)", "NMF (k=3)", "3 câu đầu", "3 câu ngẫu nhiên"]
    diem = bang.drop(columns="phuong_phap").to_numpy()
    assert np.isfinite(diem).all() and (diem >= 0).all() and (diem <= 1).all()

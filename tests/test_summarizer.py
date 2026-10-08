"""Kiểm tra tiền xử lý, tóm tắt và phần thử nghiệm trên dữ liệu mẫu."""

import itertools

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    SAMPLE_DIR,
    detect_columns,
    evaluate_dataset,
    is_sample_data,
    load_dataset,
)
from src.nmf_scratch import nmf
from src.preprocess import (
    STOPWORDS_PATH,
    filter_rare_words,
    load_stopwords,
    normalize,
    normalize_tone,
    preprocess,
    split_sentences,
)
from src.summarizer import (
    MAX_ITER,
    SEEDS,
    TOL,
    TextTooShortError,
    compare_with_sklearn,
    summarize,
    summarize_by_keyword,
    summarize_by_topic,
)

BAI_MAU = SAMPLE_DIR / "bai_mau.csv"  # 5 bản tin tổng hợp, mỗi bài 2 hoặc 3 mảng
BAI_NGAN = SAMPLE_DIR / "bai_ngan_mot_chu_de.csv"  # 6 bài ngắn, mỗi bài một chủ đề
SO_MANG = [3, 3, 3, 3, 2]  # số mảng tin của từng bài trong BAI_MAU


@pytest.fixture(scope="module")
def bai_dien_thoai() -> str:
    return load_dataset(BAI_NGAN)["text"][0]


@pytest.fixture(scope="module")
def cac_ban_tin() -> list[str]:
    return list(load_dataset(BAI_MAU)["text"])


def nhan_theo_doan(van_ban: str) -> np.ndarray:
    """Nhãn mảng tin của từng câu: mỗi đoạn văn (một dòng) là một mảng."""
    doan = [d for d in van_ban.splitlines() if d.strip()]
    return np.array([m for m, d in enumerate(doan) for _ in split_sentences(d)])


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


@pytest.mark.parametrize(
    ("van_ban", "mong_doi"),
    [
        ("Xem tại https://tin.vn/bai-1.html. Giá xăng tăng.", ["Xem tại.", "Giá xăng tăng."]),
        ("Theo www.tin-tuc.vn/abc, giá vàng giảm.", ["Theo, giá vàng giảm."]),
        ("Nguồn tin-tuc.vn đưa tin sáng nay.", ["Nguồn đưa tin sáng nay."]),
        ("Gửi thư về hop-thu@truong.edu.vn nhé. Hết rồi.", ["Gửi thư về nhé.", "Hết rồi."]),
        # số thập phân và chữ viết tắt có dấu chấm không phải đường link
        ("Giá 7,9 triệu. TP.HCM mưa to.", ["Giá 7,9 triệu.", "TP.HCM mưa to."]),
    ],
)
def test_bo_duong_link(van_ban, mong_doi):
    """Dấu chấm trong đường link không còn cắt một câu thành nhiều mảnh."""
    assert split_sentences(van_ban) == mong_doi


@pytest.mark.parametrize(
    ("kieu_khac", "kieu_chuan"),
    [
        ("hoà bình", "hòa bình"),
        ("Uỷ ban và UỶ BAN", "Ủy ban và ỦY BAN"),
        ("khoẻ, xoá, thuỷ sản, doạ", "khỏe, xóa, thủy sản, dọa"),
        ("qủa táo, qúy vị", "quả táo, quý vị"),  # dấu thanh đặt nhầm lên chữ u sau q
        # những chữ này kiểu nào cũng viết như vậy, phải giữ nguyên
        ("hoàn toàn, khuỷu tay, quý khách, quả, hòa", "hoàn toàn, khuỷu tay, quý khách, quả, hòa"),
    ],
)
def test_thong_nhat_cach_dat_dau_thanh(kieu_khac, kieu_chuan):
    assert normalize_tone(kieu_khac) == kieu_chuan


def test_hai_cach_viet_ra_cung_mot_tu():
    """ "hoà bình" và "hòa bình" phải cho cùng danh sách từ khóa."""
    assert normalize("Thuỷ sản  xuất khẩu") == "Thủy sản xuất khẩu"
    a = preprocess("Hoà bình giúp thuỷ sản phát triển.").tokens
    b = preprocess("Hòa bình giúp thủy sản phát triển.").tokens
    assert a == b and a[0]


def test_kho_tu_dung_mot_tep():
    tu_dung = load_stopwords()
    assert STOPWORDS_PATH.name == "stopwords_vi.txt"
    assert len(tu_dung) == 1961  # đúng bằng tập từ dừng trước khi gộp hai tệp
    assert {"và", "của", "tuy_nhiên", "lớn", "giờ"} <= tu_dung  # từ của kho công khai
    assert {"kéo_dài", "bên_cạnh_đó", "tuần"} <= tu_dung  # từ nhóm bổ sung
    assert not {"nước", "bán"} & tu_dung  # hai từ nhóm bỏ khỏi kho vì mang nội dung
    assert "tòa" in tu_dung and "toà" not in tu_dung  # cùng kiểu đặt dấu thanh với văn bản


def test_doc_tep_tu_dung(tmp_path):
    kho = tmp_path / "kho.txt"
    kho.write_text("# ghi chú\nvà\n\nTuy Nhiên\nkéo dài\n", encoding="utf-8")
    assert load_stopwords(kho) == {"và", "tuy_nhiên", "kéo_dài"}
    assert load_stopwords(tmp_path / "khong_co.txt") == frozenset()


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


def test_moi_cau_duoc_chuan_hoa_ve_do_dai_1(bai_dien_thoai):
    """Mỗi cột của V dài đúng bằng 1, trừ cột của câu không còn từ khóa nào."""
    kq = summarize(bai_dien_thoai, k=3, n_sentences=3)
    do_dai = np.linalg.norm(kq.V, axis=0)
    co_tu = np.array([bool(t) for t in kq.tokens])
    np.testing.assert_allclose(do_dai[co_tu], 1.0)
    np.testing.assert_allclose(do_dai[~co_tu], 0.0)
    np.testing.assert_allclose(kq.W.sum(axis=0), 1.0)  # mỗi chủ đề là một tỉ lệ từ


def test_giu_lan_chay_co_sai_so_nho_nhat(bai_dien_thoai):
    kq = summarize(bai_dien_thoai, k=3, n_sentences=3)
    assert kq.seed in SEEDS
    sai_so = [nmf(kq.V, 3, seed=s, max_iter=MAX_ITER, tol=TOL).errors[-1] for s in SEEDS]
    assert kq.errors[-1] == pytest.approx(min(sai_so))
    assert SEEDS[int(np.argmin(sai_so))] == kq.seed


def test_muc_giai_thich_va_khoang_cach_toi_san(bai_dien_thoai):
    kq = summarize(bai_dien_thoai, k=3, n_sentences=3)
    assert 0 < kq.explained < 1
    assert kq.floor <= kq.errors[-1] + 1e-9  # NMF không thể thấp hơn sàn của xấp xỉ hạng k
    assert 0 <= kq.floor_gap < 0.05  # và ở bài này chỉ cao hơn sàn chưa tới 5%
    # k tăng thì khớp hơn
    assert summarize(bai_dien_thoai, k=5, n_sentences=3).explained > kq.explained


def test_cau_ve_camera_cung_mot_chu_de(bai_dien_thoai):
    """Bài điện thoại có 4 câu liền nhau nói về camera (câu 5 đến 8)."""
    kq = summarize(bai_dien_thoai, k=3, n_sentences=3)
    assert len(set(kq.dominant_topic[4:8])) == 1
    chu_de = kq.dominant_topic[4]
    assert "camera" in [tu for tu, _ in kq.top_keywords(chu_de, 3)]


def test_tu_khoa_goi_ten_chu_de():
    """Ba từ: luôn có từ nặng nhất, các chỗ còn lại ưu tiên từ ghép trong 6 từ đứng đầu."""
    kq = summarize(load_dataset(BAI_MAU)["text"][1], k=3, n_sentences=3)
    for chu_de in range(kq.k):
        sau_tu = [tu for tu, _ in kq.top_keywords(chu_de, 6)]
        ba_tu = kq.label_keywords(chu_de)
        assert len(ba_tu) == 3 and set(ba_tu) <= set(sau_tu)
        assert ba_tu[0] == sau_tu[0]  # từ nặng nhất không bao giờ bị đẩy ra
        tu_ghep = [tu for tu in sau_tu[1:] if "_" in tu]
        assert ba_tu[1:] == (tu_ghep + [tu for tu in sau_tu[1:] if "_" not in tu])[:2]
    # Chủ đề du lịch biển của bài này: "biển" nặng nhất, rồi tới hai từ ghép; "bãi" nhường chỗ.
    bien = next(c for c in range(kq.k) if kq.top_keywords(c, 1)[0][0] == "biển")
    assert kq.label_keywords(bien) == ["biển", "du_khách", "thành_phố"]


def test_hai_cach_chon_cau_deu_chay(bai_dien_thoai):
    for cach in ("score", "coverage"):
        assert len(summarize(bai_dien_thoai, 3, 3, strategy=cach).selected) == 3


def test_cung_dau_vao_thi_cung_ket_qua(bai_dien_thoai):
    a, b = summarize(bai_dien_thoai, 3, 3), summarize(bai_dien_thoai, 3, 3)
    assert a.selected == b.selected and a.seed == b.seed
    np.testing.assert_array_equal(a.H, b.H)


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


# --- Bản tin nhiều mảng: mô hình có tìm đúng các mảng không -------------------


@pytest.mark.parametrize("bai", range(5))
def test_ban_tin_tach_dung_cac_mang(cac_ban_tin, bai):
    """Với k = số mảng: phần lớn câu được xếp đúng mảng của nó, và bản tóm tắt
    mỗi mảng một câu (cách Phủ chủ đề) nhắc đủ mọi mảng."""
    van_ban, so_mang = cac_ban_tin[bai], SO_MANG[bai]
    nhan = nhan_theo_doan(van_ban)
    kq = summarize(van_ban, k=so_mang, n_sentences=so_mang)
    assert len(nhan) == len(kq.sentences)

    # ghép mỗi chủ đề với một mảng sao cho số câu khớp là nhiều nhất
    dung = max(
        sum(int(((kq.dominant_topic == t) & (nhan == m)).sum()) for t, m in enumerate(hoan_vi))
        for hoan_vi in itertools.permutations(range(so_mang))
    )
    assert dung / len(nhan) >= 0.8
    assert {int(nhan[j]) for j in kq.selected} == set(range(so_mang))


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


@pytest.mark.parametrize("cach", ["coverage", "score"])
def test_khop_scikit_learn_tren_bai_that(cac_ban_tin, cach):
    """Cùng V, cùng điểm khởi tạo, cùng số vòng: scikit-learn phải ra cùng sai số
    và cùng bản tóm tắt. Từng ô có thể lệch ở vài chữ số cuối vì hai bản xử lý mẫu
    số bằng 0 khác nhau."""
    kq = summarize(cac_ban_tin[0], k=3, n_sentences=3, strategy=cach)
    so = compare_with_sklearn(kq, 3, cach)
    assert so.n_iter == kq.n_iter
    assert so.error_own == pytest.approx(kq.errors[-1])
    assert so.error_sklearn == pytest.approx(so.error_own, abs=1e-5)
    assert max(so.max_diff_W, so.max_diff_H) < 1e-3
    assert so.same_summary and so.selected_sklearn == kq.selected


# --- Dữ liệu mẫu và thử nghiệm ------------------------------------------------


def test_bai_mau_la_ban_tin_nhieu_mang(cac_ban_tin):
    assert is_sample_data(BAI_MAU)
    assert list(load_dataset(BAI_MAU).columns) == ["text", "summary"]
    assert len(cac_ban_tin) == 5
    for van_ban, so_mang in zip(cac_ban_tin, SO_MANG):
        nhan = nhan_theo_doan(van_ban)
        assert 30 <= len(nhan) <= 36  # mỗi bài 30 đến 36 câu
        assert len(set(nhan)) == so_mang  # mỗi mảng là một đoạn văn


def test_sau_bai_ngan_cu_van_con():
    df = load_dataset(BAI_NGAN)
    assert len(df) == 6 and is_sample_data(BAI_NGAN)
    assert all(10 <= len(split_sentences(t)) <= 13 for t in df["text"])


def test_tu_nhan_ten_cot(tmp_path):
    """Hai cặp tên cột quen thuộc được tự nhận, không cần gõ tên cột."""
    assert detect_columns(["title", "article", "abstract"]) == ("article", "abstract")
    assert detect_columns(["id", "text", "summary"]) == ("text", "summary")
    assert detect_columns(["noi_dung", "tom_tat"]) is None

    tep = tmp_path / "vietnews.jsonl"
    pd.DataFrame(
        {"guid": [1], "title": ["t"], "abstract": ["tóm tắt"], "article": ["nội dung"]}
    ).to_json(tep, orient="records", lines=True, force_ascii=False)
    df = load_dataset(tep)
    assert list(df.columns) == ["text", "summary"]
    assert df.loc[0, "text"] == "nội dung" and df.loc[0, "summary"] == "tóm tắt"


def test_bao_loi_khi_thieu_cot(tmp_path):
    tep = tmp_path / "sai.csv"
    pd.DataFrame({"noi_dung": ["a"], "tom_tat": ["b"]}).to_csv(tep, index=False)
    with pytest.raises(ValueError, match="Không tự nhận được tên cột"):
        load_dataset(tep)
    with pytest.raises(ValueError, match="thiếu cột"):
        load_dataset(tep, text_col="noi_dung", summary_col="khong_co")
    with pytest.raises(ValueError, match="cả tên cột"):
        load_dataset(tep, text_col="noi_dung")
    assert len(load_dataset(tep, text_col="noi_dung", summary_col="tom_tat")) == 1


def test_chay_thu_nghiem_tren_du_lieu_mau():
    df = load_dataset(BAI_MAU)
    bang, so_bai = evaluate_dataset(df, k_values=(2, 3), n_sentences=3)
    assert so_bai == len(df)
    assert list(bang["phuong_phap"]) == [
        "NMF (k=2)", "NMF (k=3)", "3 câu đầu", "3 câu ngẫu nhiên", "3 câu tốt nhất (mức trần)",
    ]  # fmt: skip
    diem = bang.drop(columns="phuong_phap").to_numpy()
    assert np.isfinite(diem).all() and (diem >= 0).all() and (diem <= 1).all()
    # Mức trần được xem trước bản tham chiếu nên không phương pháp nào vượt được nó.
    rouge1 = bang.set_index("phuong_phap")["rouge1_f"]
    assert rouge1["3 câu tốt nhất (mức trần)"] == rouge1.max()

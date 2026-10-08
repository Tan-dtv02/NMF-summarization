"""Chạy thử giao diện Streamlit mà không cần mở trình duyệt.

Mỗi bài kiểm tra mở ứng dụng, bấm vài nút như người trình bày sẽ bấm, rồi
xác nhận không màn nào báo lỗi.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
APP = str(APP_DIR / "streamlit_app.py")


def open_screen(name: str) -> AppTest:
    """Mở ứng dụng rồi bấm một mục trên thanh điều hướng ngang."""
    app = AppTest.from_file(APP, default_timeout=60).run()
    dieu_huong = next(r for r in app.radio if r.label == "Màn hình")
    dieu_huong.set_value(name).run()
    assert not app.exception, app.exception
    return app


def test_thanh_dieu_huong_co_ba_man():
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    dieu_huong = next(r for r in app.radio if r.label == "Màn hình")
    assert dieu_huong.options == ["Mô phỏng NMF", "Tóm tắt văn bản", "Thử nghiệm"]
    assert dieu_huong.value == "Mô phỏng NMF"  # mở ứng dụng là vào màn 1
    assert not app.sidebar  # cột bên trái đã bỏ: mọi thứ nằm ở thanh ngang và phần chính


def button(app: AppTest, label: str):
    return next(b for b in app.button if b.label == label)


def metric(app: AppTest, label: str) -> str:
    return next(m.value for m in app.metric if m.label == label)


def test_man_1_mo_phong_tung_vong():
    app = open_screen("Mô phỏng NMF")
    assert metric(app, "Vòng lặp") == "0"
    sai_so_dau = float(metric(app, "Sai số ||V − W·H||"))

    button(app, "Vòng tiếp theo").click().run()
    assert metric(app, "Vòng lặp") == "1"
    assert float(metric(app, "Sai số ||V − W·H||")) < sai_so_dau

    button(app, "Chạy đến khi dừng").click().run()
    assert int(metric(app, "Vòng lặp")) > 1
    assert any("Đã dừng" in s.value for s in app.success)

    button(app, "So với scikit-learn").click().run()
    assert not app.exception
    assert float(metric(app, "Chênh lệch lớn nhất giữa các ô")) < 1e-4

    button(app, "Làm lại từ đầu").click().run()
    assert metric(app, "Vòng lặp") == "0"


def test_man_2_tom_tat_bai_mau():
    app = open_screen("Tóm tắt văn bản")
    tieu_de = [h.value for h in app.header]
    assert "Bản tóm tắt" in tieu_de and "Các chủ đề tìm được" in tieu_de
    assert [t.label for t in app.tabs] == ["Tóm tắt theo chủ đề và từ khóa", "Bản đồ chủ đề"]
    assert not app.number_input  # không còn ô Seed
    assert 0 < float(metric(app, "V được giải thích").rstrip("%")) < 100
    assert float(metric(app, "Khoảng cách tới sàn").rstrip("%")) < 5

    app.slider[0].set_value(4).run()  # đổi số chủ đề k
    assert not app.exception
    cach_chon = next(r for r in app.radio if r.label == "Cách chọn câu")
    cach_chon.set_value("score").run()  # đổi sang cách Điểm tổng
    assert not app.exception


def test_man_2_so_voi_scikit_learn():
    app = open_screen("Tóm tắt văn bản")
    assert "So với scikit-learn" in [h.value for h in app.header]
    assert not [m for m in app.metric if m.label == "Bản tóm tắt"]  # chưa bấm thì chưa hiện

    button(app, "So với scikit-learn").click().run()
    assert not app.exception
    assert metric(app, "Bản tóm tắt") == "Trùng nhau"
    assert float(metric(app, "Chênh lệch lớn nhất giữa các ô")) < 1e-3
    tu_viet = float(metric(app, "Sai số, bản tự viết"))
    assert tu_viet == pytest.approx(float(metric(app, "Sai số, scikit-learn")), abs=1e-5)


@pytest.mark.parametrize("van_ban", ["", "Chỉ một câu."])
def test_man_2_van_ban_qua_ngan_khong_bao_loi_do(van_ban):
    app = open_screen("Tóm tắt văn bản")
    app.selectbox[0].set_value("Tự dán văn bản").run()
    app.text_area[0].set_value(van_ban).run()
    assert not app.exception
    assert len(app.info) == 1  # thông báo nhẹ nhàng, không phải lỗi


def test_man_3_mo_duoc():
    app = open_screen("Thử nghiệm")
    assert not app.exception


def test_man_3_chay_lai_chi_can_chon_so_bai(tmp_path, monkeypatch):
    """Khung chạy lại không còn ô tên cột, có ô chọn cách chọn câu, chạy được cả Điểm tổng."""
    app = open_screen("Thử nghiệm")
    assert not app.text_input  # tên cột được tự nhận
    so_bai = next(n for n in app.number_input if n.label == "Số bài")
    assert so_bai.max == 10_000
    cach_chon = next(r for r in app.radio if r.label == "Cách chọn câu")
    assert cach_chon.options == ["Phủ chủ đề", "Điểm tổng"] and cach_chon.value == "coverage"

    # Ghi kết quả vào thư mục tạm để bài kiểm tra không để lại tệp trong results/.
    import src.evaluate

    monkeypatch.setattr(src.evaluate, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys.modules["screen_experiment"], "RESULTS_DIR", tmp_path)
    tep = next(s for s in app.selectbox if s.label == "Tệp dữ liệu")
    tep.select_index(next(i for i, o in enumerate(tep.options) if o.startswith("bai_mau.csv")))
    so_bai.set_value(2)
    cach_chon.set_value("score")
    button(app, "Chạy thử nghiệm").click().run()
    assert not app.exception
    # Chỉ có cảnh báo "dữ liệu mẫu", không có cảnh báo thiếu cột.
    assert [w for w in app.warning if "CHẠY THỬ" not in w.value] == []

    (ket_qua,) = tmp_path.glob("rouge_*.csv")
    bang = pd.read_csv(ket_qua, encoding="utf-8-sig")
    assert set(bang["cach_chon_cau"]) == {"score"} and set(bang["so_bai"]) == {2}
    # Chạy xong thì màn hình chuyển ngay sang lần vừa chạy.
    lan_chay = next(s for s in app.selectbox if s.label == "Lần chạy (mới nhất ở trên)")
    assert lan_chay.value == ket_qua.name
    assert any("Đã lưu kết quả" in s.value for s in app.success)


def test_man_3_bieu_do_co_dong_muc_tran():
    """Dòng mức trần được nhận ra và xếp vào nhóm màu riêng, không lẫn với NMF hay đối chứng."""
    sys.path.insert(0, str(APP_DIR))  # các tệp giao diện gọi nhau theo tên tệp
    from ui_helpers import is_ceiling, rouge_chart

    bang = pd.DataFrame(
        {
            "phuong_phap": ["NMF (k=3)", "3 câu đầu", "3 câu tốt nhất (mức trần)"],
            "rouge1_f": [0.26, 0.25, 0.46],
            "rouge2_f": [0.13, 0.14, 0.29],
            "rougeL_f": [0.18, 0.17, 0.34],
        }
    )
    assert [is_ceiling(ten) for ten in bang["phuong_phap"]] == [False, False, True]
    nhom = rouge_chart(bang).to_dict()["spec"]["layer"][0]["encoding"]["color"]["scale"]["domain"]
    assert nhom == ["NMF", "Đối chứng", "Mức trần"]
    # Tệp kết quả cũ (trước khi có mức trần) vẫn vẽ được, chỉ còn hai nhóm.
    cu = rouge_chart(bang.iloc[:2]).to_dict()["spec"]["layer"][0]["encoding"]["color"]["scale"]
    assert cu["domain"] == ["NMF", "Đối chứng"]

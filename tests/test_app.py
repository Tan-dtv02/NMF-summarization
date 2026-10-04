"""Chạy thử giao diện Streamlit mà không cần mở trình duyệt.

Mỗi bài kiểm tra mở ứng dụng, bấm vài nút như người trình bày sẽ bấm, rồi
xác nhận không màn nào báo lỗi.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def open_screen(name: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.sidebar.radio[0].set_value(name).run()
    assert not app.exception, app.exception
    return app


def button(app: AppTest, label: str):
    return next(b for b in app.button if b.label == label)


def metric(app: AppTest, label: str) -> str:
    return next(m.value for m in app.metric if m.label == label)


def test_man_1_mo_phong_tung_vong():
    app = open_screen("1. Mô phỏng NMF")
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
    app = open_screen("2. Tóm tắt văn bản")
    tieu_de = [h.value for h in app.header]
    assert "Bản tóm tắt" in tieu_de and "Các chủ đề tìm được" in tieu_de

    app.slider[0].set_value(4).run()  # đổi số chủ đề k
    assert not app.exception


@pytest.mark.parametrize("van_ban", ["", "Chỉ một câu."])
def test_man_2_van_ban_qua_ngan_khong_bao_loi_do(van_ban):
    app = open_screen("2. Tóm tắt văn bản")
    app.selectbox[0].set_value("Tự dán văn bản").run()
    app.text_area[0].set_value(van_ban).run()
    assert not app.exception
    assert len(app.info) == 1  # thông báo nhẹ nhàng, không phải lỗi


def test_man_3_mo_duoc():
    app = open_screen("3. Thử nghiệm")
    assert not app.exception

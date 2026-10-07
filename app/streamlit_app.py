"""Điểm vào của ứng dụng: chọn một trong ba màn.

Chạy từ thư mục gốc của dự án:

    python -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Cho phép `from src...` dù Streamlit khởi động từ thư mục app/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import screen_experiment  # noqa: E402
import screen_simulation  # noqa: E402
import screen_summary  # noqa: E402
import styles  # noqa: E402

from src.preprocess import HAS_UNDERTHESEA, tokenizer_name  # noqa: E402

SCREENS = {
    "Mô phỏng NMF": screen_simulation.render,
    "Tóm tắt văn bản": screen_summary.render,
    "Thử nghiệm": screen_experiment.render,
}


def main() -> None:
    st.set_page_config(page_title="NMF - Tóm tắt văn bản", layout="wide")
    st.html(styles.CSS)
    screen = styles.navbar(list(SCREENS))

    if not HAS_UNDERTHESEA:
        st.warning(
            "Chưa cài được underthesea nên đang tách từ theo khoảng trắng. Từ ghép như "
            "“điện thoại” sẽ bị tách đôi. Hãy chạy `pip install -r requirements.txt`."
        )
    SCREENS[screen]()
    styles.footer(f"Bộ tách từ: {tokenizer_name()}")


main()

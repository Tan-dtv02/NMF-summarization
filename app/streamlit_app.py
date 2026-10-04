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

from src.preprocess import HAS_UNDERTHESEA, tokenizer_name  # noqa: E402

SCREENS = {
    "1. Mô phỏng NMF": screen_simulation.render,
    "2. Tóm tắt văn bản": screen_summary.render,
    "3. Thử nghiệm": screen_experiment.render,
}

# Chữ và bảng lớn hơn mặc định để người ngồi cuối lớp vẫn đọc được qua máy chiếu.
CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1500px; }
.matrix-wrap { overflow-x: auto; }
table.matrix { border-collapse: separate; border-spacing: 3px; font-size: 1rem; }
table.matrix th { font-weight: 600; color: #52514e; padding: 4px 6px; text-align: right;
                  white-space: nowrap; }
table.matrix thead th { text-align: center; }
table.matrix td { padding: 6px; text-align: right; border-radius: 6px; color: #0b0b0b;
                  font-variant-numeric: tabular-nums; white-space: nowrap;
                  border: 1px solid #e4e3df; }
table.matrix .mark { display: inline-block; width: 1.4em; text-align: center; font-size: 0.65em;
                     color: #52514e; }
.chip { display: inline-block; padding: 1px 9px; border-radius: 5px; font-size: 0.85em;
        font-weight: 600; color: #0b0b0b; }
.article { font-size: 1.15rem; line-height: 2.1; color: #0b0b0b; }
.article .sentence { padding: 4px 5px; border-radius: 5px; box-decoration-break: clone;
                     -webkit-box-decoration-break: clone; }
.index { color: #52514e; font-size: 0.75em; font-weight: 600; }
ul.summary { list-style: none; padding: 0; margin: 0; font-size: 1.15rem; line-height: 1.6;
             color: #0b0b0b; }
ul.summary li { margin-bottom: 14px; padding: 10px 14px; background: #f6f6f4; border-radius: 8px; }
.topic-card { padding: 10px 14px; background: #f6f6f4; border-radius: 8px; font-size: 1.1rem;
              line-height: 1.6; color: #0b0b0b; }
</style>
"""


def main() -> None:
    st.set_page_config(page_title="NMF - Tóm tắt văn bản", layout="wide")
    st.html(CSS)

    with st.sidebar:
        st.markdown("### NMF và tóm tắt dựa trên chủ đề")
        screen = st.radio("Màn hình", list(SCREENS), label_visibility="collapsed")
        st.divider()
        st.caption(f"Bộ tách từ: {tokenizer_name()}")
        if not HAS_UNDERTHESEA:
            st.warning(
                "Chưa cài được underthesea nên đang tách từ theo khoảng trắng. Từ ghép như "
                "“điện thoại” sẽ bị tách đôi. Hãy chạy `pip install -r requirements.txt`."
            )

    SCREENS[screen]()


main()

"""Diện mạo chung của ứng dụng: CSS và vài khối dựng trang dùng cho cả ba màn.

Tệp này chỉ lo phần "trông ra sao": thanh điều hướng, tiêu đề trang, thẻ nội
dung, chân trang. Không có thuật toán hay số liệu nào ở đây.

Màu, phông và bo góc của các thành phần có sẵn của Streamlit (nút, thanh trượt,
ô nhập, bảng) đặt ở .streamlit/config.toml. CSS dưới đây lo phần còn lại và
dùng đúng các màu đó, khai báo một lần ở `:root`.
"""

from __future__ import annotations

import html
from contextlib import contextmanager

import streamlit as st

CSS = """
<style>
:root {
  --bg: #f4f6fa;            /* nền trang */
  --surface: #ffffff;       /* nền thẻ */
  --line: #e2e6ee;          /* viền thẻ */
  --line-soft: #eef1f6;     /* đường kẻ mờ bên trong thẻ */
  --ink: #141b2d;           /* chữ chính */
  --ink-2: #3b4558;         /* chữ thân bài */
  --muted: #667085;         /* chữ phụ */
  --accent: #1e3a8a;        /* màu nhấn duy nhất: xanh đậm */
  --accent-soft: #e9eefb;
  --radius: 14px;
  --shadow: 0 1px 2px rgba(16, 24, 40, .04), 0 2px 8px rgba(16, 24, 40, .04);
  --nav-h: 60px;
  --page-w: 1240px;
}

/* ----- Khung trang ------------------------------------------------------ */
.stApp { background: var(--bg); }
header[data-testid="stHeader"] { display: none; }
.stMainBlockContainer {
  max-width: var(--page-w);
  padding: calc(var(--nav-h) + 30px) 28px 40px;
}
h1, h2, h3 { letter-spacing: -0.01em; color: var(--ink); }
h1 { padding: 0 0 .15rem !important; }
h2 { padding: 0 0 .35rem !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
  color: var(--muted) !important; opacity: 1 !important;
}
[data-testid="stHeadingWithActionElements"] a { display: none; }  /* biểu tượng móc xích */

/* ----- Thanh điều hướng ngang, cố định ở đầu trang ------------------------ */
.st-key-navbar {
  position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
  height: var(--nav-h);
  padding: 0 max(28px, calc((100vw - var(--page-w)) / 2 + 28px));
  background: rgba(255, 255, 255, .94);
  backdrop-filter: saturate(180%) blur(10px);
  -webkit-backdrop-filter: saturate(180%) blur(10px);
  border-bottom: 1px solid var(--line);
  display: flex; flex-direction: row !important; flex-wrap: nowrap;
  align-items: center; justify-content: space-between; gap: 16px;
}
.st-key-navbar > div { width: auto !important; flex: 0 0 auto; }
.brand { display: flex; align-items: center; gap: 11px; white-space: nowrap; }
.brand-mark {
  display: grid; grid-template-columns: repeat(2, 9px); gap: 3px;
  padding: 7px; border-radius: 9px; background: var(--accent);
}
.brand-mark i { width: 9px; height: 9px; border-radius: 2.5px; background: #fff; }
.brand-mark i:nth-child(2) { opacity: .55; }
.brand-mark i:nth-child(3) { opacity: .55; }
.brand-name { font-weight: 700; font-size: 1.02rem; color: var(--ink); }
.brand-sub { color: var(--muted); font-size: .9rem; }
.brand-sub::before { content: "·"; margin: 0 9px 0 0; color: #b6bdcb; }

.st-key-navbar [role="radiogroup"] { flex-wrap: nowrap; gap: 4px; }
.st-key-navbar [role="radiogroup"] label {
  margin: 0; padding: 7px 14px; border-radius: 9px; cursor: pointer;
  transition: background .12s ease;
}
.st-key-navbar [role="radiogroup"] label > div > div:first-child:not([data-testid]) {
  display: none;  /* chấm tròn của nút chọn */
}
.st-key-navbar [role="radiogroup"] label p {
  font-size: .94rem; font-weight: 500; color: var(--ink-2); white-space: nowrap;
}
.st-key-navbar [role="radiogroup"] label:hover { background: var(--line-soft); }
.st-key-navbar [role="radiogroup"] label[data-selected="true"] { background: var(--accent-soft); }
.st-key-navbar [role="radiogroup"] label[data-selected="true"] p {
  color: var(--accent); font-weight: 600;
}

/* ----- Tiêu đề trang ------------------------------------------------------ */
.page-head { margin-bottom: 4px; }
.page-eyebrow {
  font-size: .76rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 8px;
}
.page-head h1 {
  font-size: 1.85rem; font-weight: 700; line-height: 1.25; margin: 0 0 8px;
  padding: 0 !important;
}
.page-lead { color: var(--ink-2); font-size: 1.02rem; line-height: 1.65; max-width: 60rem;
             margin: 0; }
.page-lead b { color: var(--ink); font-weight: 600; }

/* ----- Thẻ nội dung ------------------------------------------------------- */
div[class*="st-key-card-"] {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 22px 24px 24px;
}
/* Thẻ "Bản tóm tắt" bám theo màn hình khi cuộn đọc bài gốc. Phải đặt lên khối bọc ngay
   ngoài thẻ: khối cha của nó cao bằng cả cột nên mới có chỗ để trượt. */
[data-testid="stLayoutWrapper"]:has(> .st-key-card-summary) {
  position: sticky; top: calc(var(--nav-h) + 16px);
}

/* ----- Thành phần có sẵn của Streamlit ------------------------------------ */
[data-testid="stMetric"] {
  background: #f8f9fc; border: 1px solid var(--line-soft); border-radius: 12px;
  padding: 12px 16px 10px;
}
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: .86rem; }
[data-testid="stMetricValue"] { color: var(--ink); font-variant-numeric: tabular-nums; }
[data-testid="stExpander"] details {
  border: 1px solid var(--line); border-radius: 12px; background: var(--surface);
}
[data-testid="stExpander"] summary { font-weight: 500; }
.stButton button, .stFormSubmitButton button { font-weight: 500; }
[data-testid="stAlert"] { border-radius: 12px; }
.stTabs [data-baseweb="tab"] p { font-weight: 500; }

/* ----- Nhãn chủ đề, thẻ chủ đề -------------------------------------------- */
.chip {
  display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px 2px 8px;
  border-radius: 999px; font-size: .8rem; font-weight: 600; color: #0b0b0b;
  white-space: nowrap; vertical-align: .08em;
}
.chip i { width: 7px; height: 7px; border-radius: 50%; }
.topic-grid { display: grid; gap: 14px;
              grid-template-columns: repeat(var(--cols, 3), minmax(0, 1fr)); }
.topic-card {
  position: relative; padding: 14px 16px 15px; border-radius: 12px;
  background: var(--surface); border: 1px solid var(--line); overflow: hidden;
}
.topic-card::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px;
                      background: var(--c); }
.tc-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
.tc-name { font-weight: 600; color: var(--ink); }
.tc-pct { font-size: 1.35rem; font-weight: 700; color: var(--ink);
          font-variant-numeric: tabular-nums; }
.tc-bar { height: 6px; border-radius: 999px; background: var(--line-soft); margin: 8px 0 12px; }
.tc-bar span { display: block; height: 100%; border-radius: 999px; background: var(--c); }
.tc-words { display: flex; flex-wrap: wrap; gap: 6px; }
.tc-words span {
  padding: 2px 10px; border-radius: 999px; font-size: .86rem; color: var(--ink-2);
  background: #f3f5f9; border: 1px solid var(--line-soft);
}

/* ----- Bài gốc và bản tóm tắt --------------------------------------------- */
.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: .92rem; color: var(--ink-2);
          margin: 2px 0 6px; }
.article { font-size: 1.04rem; line-height: 2.05; color: var(--ink-2); }
.article .sentence {
  padding: .16em .34em; border-radius: 6px;
  box-decoration-break: clone; -webkit-box-decoration-break: clone;
}
.index { font-size: .7em; font-weight: 600; white-space: nowrap; color: #3b4558; }
.index .dot { display: inline-block; width: .62em; height: .62em; border-radius: 50%;
              margin-right: 4px; vertical-align: .05em; }
ul.summary { list-style: none; padding: 0; margin: 0; }
ul.summary li {
  position: relative; margin: 0 0 12px; padding: 12px 14px 13px 18px;
  background: #f8f9fc; border: 1px solid var(--line-soft); border-radius: 12px;
  font-size: 1.02rem; line-height: 1.6; color: var(--ink); overflow: hidden;
}
ul.summary li:last-child { margin-bottom: 0; }
ul.summary li::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px;
                        background: var(--c, #c9cfdb); }
ul.summary .meta { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
ul.summary .index { font-size: .8rem; font-weight: 500; color: var(--muted); }

/* ----- Bảng ma trận (màn 1) ----------------------------------------------- */
.matrix-wrap { overflow-x: auto; }
table.matrix { border-collapse: separate; border-spacing: 4px; font-size: .98rem; }
table.matrix th { font-weight: 500; font-size: .86rem; color: var(--muted); padding: 4px 8px;
                  text-align: right; white-space: nowrap; }
table.matrix thead th { text-align: center; }
table.matrix td { padding: 7px 10px; text-align: right; border-radius: 8px; color: #0b0b0b;
                  font-variant-numeric: tabular-nums; white-space: nowrap;
                  box-shadow: inset 0 0 0 1px var(--line-soft); }
table.matrix .mark { display: inline-block; width: 1.3em; text-align: center; font-size: .62em;
                     color: #52514e; }
.token-list { display: grid; gap: 8px; margin-top: 2px; }
.token-row { display: flex; align-items: baseline; gap: 10px; }
.token-row .label { flex: 0 0 3.6rem; color: var(--muted); font-size: .9rem; }
.token-row .tokens { display: flex; flex-wrap: wrap; gap: 6px; }
.token { padding: 2px 10px; border-radius: 999px; font-size: .88rem; background: var(--accent-soft);
         color: var(--accent); font-weight: 500; }
.token.none { background: #f3f5f9; color: var(--muted); font-weight: 400; }
.matrix-title { font-weight: 600; color: var(--ink); margin: 0 0 2px; }
.matrix-title span { font-weight: 400; color: var(--muted); }

/* ----- Ô số liệu tóm lược và thẻ thông tin lần chạy (màn 3) ---------------- */
.kpi-grid { display: grid; gap: 14px;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
.kpi {
  position: relative; padding: 14px 16px 13px; border-radius: 12px; overflow: hidden;
  background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow);
}
.kpi::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 4px;
               background: var(--c); }
.kpi-group { font-size: .74rem; font-weight: 600; letter-spacing: .07em;
             text-transform: uppercase; color: var(--muted); }
.kpi-name { font-weight: 600; color: var(--ink); margin: 2px 0 6px; }
.kpi-value { font-size: 1.7rem; font-weight: 700; color: var(--ink); line-height: 1.15;
             font-variant-numeric: tabular-nums; }
.kpi-unit { font-size: .82rem; color: var(--muted); margin-top: 2px; }
.meta-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 4px; }
.meta-pill { padding: 4px 12px; border-radius: 999px; font-size: .86rem; color: var(--ink-2);
             background: #f3f5f9; border: 1px solid var(--line-soft); }
.meta-pill b { font-weight: 600; color: var(--ink); }

/* ----- Chân trang --------------------------------------------------------- */
.app-foot { margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--line);
            color: var(--muted); font-size: .84rem; display: flex; flex-wrap: wrap;
            justify-content: space-between; gap: 6px 24px; }

/* ----- Màn hình hẹp ------------------------------------------------------- */
@media (max-width: 920px) {
  .brand-sub { display: none; }
  .stMainBlockContainer { padding-left: 16px; padding-right: 16px; }
  .st-key-navbar { padding: 0 16px; }
  div[class*="st-key-card-"] { padding: 16px; }
  .topic-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""

BRAND = (
    '<div class="brand"><span class="brand-mark"><i></i><i></i><i></i><i></i></span>'
    '<span class="brand-name">NMF</span>'
    '<span class="brand-sub">Tóm tắt văn bản dựa trên chủ đề</span></div>'
)


def navbar(screens: list[str]) -> str:
    """Vẽ thanh điều hướng ngang ở đầu trang, trả về tên màn đang được chọn.

    Các mục là một nhóm nút chọn (radio) của Streamlit được CSS vẽ lại thành
    thẻ ngang, nên luôn có đúng một mục được chọn và không cần thư viện nào thêm.
    """
    with st.container(key="navbar"):
        st.html(BRAND)
        return st.radio(
            "Màn hình", screens, horizontal=True, label_visibility="collapsed", key="screen"
        )


def page_header(eyebrow: str, title: str, lead: str) -> None:
    """Phần mở đầu của một màn: dòng nhãn nhỏ, tiêu đề, một đoạn giới thiệu ngắn.

    `lead` là HTML (được phép dùng <b> để nhấn tên ma trận), nên chỉ truyền chữ
    do chương trình viết sẵn, không truyền chữ người dùng nhập.
    """
    st.html(
        f'<div class="page-head"><div class="page-eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1>{html.escape(title)}</h1><p class="page-lead">{lead}</p></div>'
    )


@contextmanager
def card(key: str):
    """Một thẻ nền trắng bo góc chứa một phần nội dung. `key` phải khác nhau trong một màn."""
    with st.container(key=f"card-{key}") as container:
        yield container


def footer(note: str) -> None:
    st.html(
        '<div class="app-foot"><span> '
        f"NMF VÀ TÓM TẮT DỰA TRÊN CHỦ ĐỀ</span><span>{html.escape(note)}</span></div>"
    )

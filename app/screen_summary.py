"""Màn 2: tóm tắt một văn bản và giải thích vì sao chọn những câu đó.

Gồm hai phần khai thác thêm từ W và H (tóm tắt theo chủ đề hoặc từ khóa, bản đồ
chủ đề) và phần đối chiếu NMF tự viết với scikit-learn trên chính bài đang xem.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st
from styles import card, page_header
from ui_helpers import TOPIC_INK, tint, topic_chip, topic_color, topic_map_chart

from src.evaluate import SAMPLE_DIR
from src.summarizer import (
    SEEDS,
    STRATEGIES,
    SklearnCheck,
    SummaryResult,
    TextTooShortError,
    compare_with_sklearn,
    summarize,
    summarize_by_keyword,
    summarize_by_topic,
)

PASTE_OPTION = "Tự dán văn bản"


@st.cache_data
def _load_samples() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "bai_mau.csv", encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def _summarize(text: str, k: int, n: int, strategy: str, min_df: int) -> SummaryResult:
    """Lưu lại kết quả theo đầu vào, để đổi qua lại giữa các tab không phải tính lại."""
    return summarize(text, k, n, strategy, min_df)


@st.cache_data(show_spinner=False)
def _sklearn_check(text: str, k: int, n: int, strategy: str, min_df: int) -> SklearnCheck:
    return compare_with_sklearn(_summarize(text, k, n, strategy, min_df), n, strategy)


def _sentences_html(result: SummaryResult, highlighted: list[int]) -> str:
    """Bài gốc: mỗi câu có nền theo màu chủ đề trội, câu được chọn thì in đậm, có ★.

    Câu không còn từ khóa nào (chỉ gồm từ dừng) không thuộc chủ đề nào nên để
    trơn, không tô màu. Câu có màu luôn dùng chữ sẫm trên nền nhạt, nên đọc rõ ở
    cả giao diện sáng lẫn tối. Ô vuông màu đậm ở đầu câu giúp phân biệt hai màu
    nền nhạt gần nhau (ví dụ cam và vàng).
    """
    spans = []
    for j, sentence in enumerate(result.sentences):
        topic = int(result.dominant_topic[j])
        chosen = j in highlighted
        style, dot = "", ""
        if topic >= 0:
            # Câu thường nền rất nhạt để cả bài dễ đọc; câu được chọn nền đậm hơn nên nổi lên.
            shade = 0.7 if chosen else 0.86
            style = f"background:{tint(topic_color(topic), shade)};color:{TOPIC_INK};"
            dot = f'<span class="dot" style="background:{topic_color(topic)}"></span>'
        if chosen:
            style += "font-weight:600;"
            if topic >= 0:
                style += f"box-shadow:inset 0 -3px 0 {topic_color(topic)};"
        label = f"{'★&nbsp;' if chosen else ''}{j + 1}"
        spans.append(  # &nbsp; giữ số thứ tự dính liền với chữ đầu câu khi xuống dòng
            f'<span class="sentence" style="{style}">'
            f'<sup class="index">{dot}{label}</sup>&nbsp;{html.escape(sentence)}</span>'
        )
    return f'<div class="article">{" ".join(spans)}</div>'


def _keywords_text(result: SummaryResult, topic: int) -> str:
    """Ba từ khóa gọi tên chủ đề, viết lại có dấu cách ("đồng_hồ" thành "đồng hồ")."""
    return ", ".join(word.replace("_", " ") for word in result.label_keywords(topic))


def _legend_html(result: SummaryResult) -> str:
    """Chú giải màu: nhãn của từng chủ đề kèm ba từ khóa gọi tên nó."""
    items = []
    for topic in range(result.k):
        words = _keywords_text(result, topic)
        items.append(f"<span>{topic_chip(topic)} {html.escape(words)}</span>")
    return f'<div class="legend">{"".join(items)}</div>'


def _summary_box(result: SummaryResult, indices: list[int]) -> str:
    items = []
    for j in indices:
        topic = int(result.dominant_topic[j])
        color = f' style="--c:{topic_color(topic)}"' if topic >= 0 else ""
        items.append(
            f'<li{color}><div class="meta">{topic_chip(topic)}'
            f'<span class="index">câu {j + 1}</span></div>{html.escape(result.sentences[j])}</li>'
        )
    return f'<ul class="summary">{"".join(items)}</ul>'


def _topic_cards_html(result: SummaryResult) -> str:
    """Mỗi chủ đề một thẻ: tên, tỉ trọng (số và thanh ngang), ba từ khóa gọi tên."""
    cards = []
    for topic in range(result.k):
        share = float(result.topic_weights[topic])
        words = "".join(
            f"<span>{html.escape(word.replace('_', ' '))}</span>"
            for word in result.label_keywords(topic)
        )
        cards.append(
            f'<div class="topic-card" style="--c:{topic_color(topic)}">'
            f'<div class="tc-head"><span class="tc-name">Chủ đề {topic + 1}</span>'
            f'<span class="tc-pct">{share:.0%}</span></div>'
            f'<div class="tc-bar"><span style="width:{share:.1%}"></span></div>'
            f'<div class="tc-words">{words}</div></div>'
        )
    # Tối đa 5 thẻ một hàng; 6 chủ đề thì chia đều thành hai hàng ba thẻ.
    columns = result.k if result.k <= 5 else 3
    return f'<div class="topic-grid" style="--cols:{columns}">{"".join(cards)}</div>'


def _topic_name(result: SummaryResult, topic: int) -> str:
    return f"Chủ đề {topic + 1}: {_keywords_text(result, topic)}"


def _fit_quality(result: SummaryResult) -> None:
    """Hai chỉ số cho biết k chủ đề khớp ma trận V tới đâu, thay cho con số sai số thô."""
    c_explained, c_gap, c_note = st.columns([4, 5, 9], gap="medium")
    # Công thức nằm trong dấu "?" cạnh con số: màn hình gọn, cần thì rê chuột là thấy.
    c_explained.metric(
        "V được giải thích",
        f"{result.explained:.1%}",
        help="Công thức: 1 − (sai số ÷ ‖V‖)², với sai số = ‖V − W·H‖.",
    )
    gap = result.floor_gap
    c_gap.metric(
        "Khoảng cách tới sàn",
        "sàn = 0" if gap is None else f"{gap:.2%}",
        help=(
            "Sàn là sai số nhỏ nhất mà mọi cách xấp xỉ V bằng k thành phần có thể đạt, "
            "tính bằng SVD. Khoảng cách = (sai số − sàn) ÷ sàn."
        ),
    )
    c_note.caption(
        f"“V được giải thích”: phần nội dung của văn bản mà {result.k} chủ đề tái tạo lại "
        "được. “Khoảng cách tới sàn”: NMF còn cách mức khớp tốt nhất có thể bao xa, càng gần "
        "0 càng tốt. Hai con số này đo mức nén thông tin, không đo chất lượng bản tóm tắt."
    )


def _sklearn_panel(text: str, k: int, n: int, strategy: str, min_df: int) -> None:
    """Đối chiếu NMF tự viết với scikit-learn trên đúng bài đang tóm tắt."""
    st.header("So với scikit-learn")
    st.caption(
        "Chạy NMF của thư viện scikit-learn trên đúng ma trận V của bài này, từ cùng điểm "
        "khởi tạo và cùng số vòng lặp với lần chạy được giữ lại, rồi so với bản tự viết."
    )
    # Ghi nhớ bài nào đã bấm nút: đổi bài hay đổi k thì phần so sánh cũ tự ẩn đi.
    current = (text, k, n, strategy, min_df)
    if st.button("So với scikit-learn"):
        st.session_state.sklearn_checked = current
    if st.session_state.get("sklearn_checked") != current:
        return

    try:
        check = _sklearn_check(text, k, n, strategy, min_df)
    except ImportError:
        st.info(
            "Máy này chưa cài scikit-learn nên chưa đối chiếu được. Hãy chạy "
            "`pip install -r requirements.txt`. Bản tóm tắt ở trên không dùng tới thư viện này."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sai số, bản tự viết", f"{check.error_own:.6f}")
    c2.metric("Sai số, scikit-learn", f"{check.error_sklearn:.6f}")
    c3.metric("Chênh lệch lớn nhất giữa các ô", f"{max(check.max_diff_W, check.max_diff_H):.1e}")
    c4.metric("Bản tóm tắt", "Trùng nhau" if check.same_summary else "Khác nhau")
    chosen = ", ".join(str(j + 1) for j in check.selected_sklearn)
    st.caption(
        f"Cả hai chạy {check.n_iter} vòng. Dùng W, H của scikit-learn thì các câu được chọn "
        f"là: {chosen}. Hai bản xử lý mẫu số bằng 0 hơi khác nhau (bản tự viết luôn cộng "
        "thêm 1e-9), nên từng ô có thể lệch chút ít, rõ hơn khi ma trận rất thưa (tắt lọc từ "
        "hiếm). Sai số và bản tóm tắt là hai thứ cần khớp."
    )


def _more_panel(result: SummaryResult, n_sentences: int) -> None:
    """Hai cách khai thác thêm W và H: tóm tắt xoay quanh một chủ đề hay một từ khóa, và
    bản đồ chủ đề của cả bài."""
    st.header("Khai thác thêm từ W và H")
    tab_focus, tab_map = st.tabs(["Tóm tắt theo chủ đề và từ khóa", "Bản đồ chủ đề"])

    with tab_focus:
        mode = st.radio("Tóm tắt xoay quanh", ["Một chủ đề", "Một từ khóa"], horizontal=True)
        if mode == "Một chủ đề":
            topic = st.selectbox(
                "Chủ đề", range(result.k), format_func=lambda t: _topic_name(result, t)
            )
            chosen = summarize_by_topic(result, topic, n_sentences)
            st.caption(f"Các câu có trọng số lớn nhất ở hàng {topic + 1} của H.")
            st.html(_summary_box(result, chosen))
        else:
            keyword = st.text_input("Từ khóa", placeholder="ví dụ: giá, học sinh, biển")
            focused = summarize_by_keyword(result, keyword, n_sentences) if keyword else None
            if not keyword:
                st.info("Gõ một từ khóa có trong bài rồi nhấn Enter.")
            elif focused is None:
                st.info(f"Bài không nhắc tới “{keyword}”. Hãy thử một từ khác.")
            else:
                chosen, profile = focused
                share = " · ".join(
                    f"Chủ đề {t + 1}: {profile[t]:.0%}"
                    for t in range(result.k)
                    if profile[t] >= 0.005
                )
                st.caption(
                    f"“{keyword}” thuộc về: {share}. Câu được chấm điểm theo tỉ lệ này, nên câu "
                    "không chứa đúng từ khóa nhưng cùng chủ đề vẫn có thể được chọn."
                )
                st.html(_summary_box(result, chosen))

    with tab_map:
        st.caption(
            "Mỗi cột là một câu theo thứ tự trong bài, mỗi hàng là một chủ đề. Ô càng đậm thì "
            "câu càng thuộc chủ đề đó. ★ đánh dấu câu được chọn. Rê chuột vào ô để đọc câu."
        )
        st.altair_chart(
            topic_map_chart(result.H, result.sentences, result.selected), width="stretch"
        )


def render() -> None:
    page_header(
        "Màn 2 · Ứng dụng",
        "Tóm tắt văn bản bằng NMF",
        "Chọn một bài mẫu hoặc dán văn bản. NMF tìm các chủ đề trong bài, rồi chọn "
        "những câu tiêu biểu nhất của từng chủ đề làm bản tóm tắt.",
    )

    # ----- Đầu vào --------------------------------------------------------
    samples = _load_samples()
    with card("input"):
        source_col, k_col, n_col = st.columns([3, 1, 1], gap="large")
        source = source_col.selectbox("Văn bản", [*samples["title"], PASTE_OPTION])
        k = k_col.slider("Số chủ đề k", 2, 6, 3)
        n_sentences = n_col.slider("Số câu tóm tắt", 1, 6, 3)

        if source == PASTE_OPTION:
            with st.form("paste"):
                text = st.text_area("Dán văn bản vào đây", height=220)
                st.form_submit_button("Tóm tắt", type="primary")
        else:
            text = samples.loc[samples["title"] == source, "text"].iloc[0]
            st.caption("Bài mẫu soạn riêng để chạy thử, không phải bài báo thật.")

        with st.expander("Tùy chọn nâng cao"):
            a, b = st.columns(2)
            strategy = a.radio("Cách chọn câu", list(STRATEGIES), format_func=STRATEGIES.get)
            drop_rare = b.checkbox(
                "Bỏ từ chỉ xuất hiện trong một câu",
                value=True,
                help="Từ chỉ có ở một câu không nối được các câu với nhau, nên không giúp "
                "tìm chủ đề.",
            )
    min_df = 2 if drop_rare else 1

    try:
        result = _summarize(text, k, n_sentences, strategy, min_df)
    except TextTooShortError as message:
        st.info(str(message))
        return
    if result.k < k:
        st.caption(
            f"Bài chỉ có {len(result.sentences)} câu nên số chủ đề được giảm còn {result.k}."
        )

    # ----- Chủ đề ---------------------------------------------------------
    with card("topics"):
        st.header("Các chủ đề tìm được")
        st.caption(
            f"{len(result.sentences)} câu · {len(result.vocab)} từ khóa · NMF chạy từ "
            f"{len(SEEDS)} điểm khởi tạo và giữ lần khớp V tốt nhất (dừng sau {result.n_iter} "
            "vòng). Phần trăm là mức độ chủ đề bao phủ bài viết."
        )
        st.html(_topic_cards_html(result))
        _fit_quality(result)

    # ----- Bài gốc và bản tóm tắt ------------------------------------------
    article_col, summary_col = st.columns([3, 2], gap="medium")
    with article_col, card("article"):
        st.header("Bài gốc")
        st.caption(
            "Màu nền: chủ đề trội của câu. Chữ đậm có ★: câu được chọn vào bản tóm tắt. "
            "Câu để trơn là câu không còn từ khóa nào."
        )
        st.html(_legend_html(result))
        st.html(_sentences_html(result, result.selected))
    with summary_col, card("summary"):
        st.header("Bản tóm tắt")
        st.caption(
            {
                "coverage": "Mỗi chủ đề lần lượt góp câu tiêu biểu nhất của mình, bắt đầu từ chủ "
                "đề quan trọng nhất. Các câu được xếp theo thứ tự trong bài gốc.",
                "score": "Các câu có điểm tổng cao nhất, xếp theo thứ tự trong bài gốc.",
            }[strategy]
        )
        st.html(_summary_box(result, result.selected))

    with st.expander("Vì sao chọn những câu này: bảng H và điểm của từng câu"):
        n = len(result.sentences)
        table = pd.DataFrame({"Câu": range(1, n + 1)})
        table["Chủ đề trội"] = [f"CĐ {t + 1}" if t >= 0 else "–" for t in result.dominant_topic]
        for topic in range(result.k):
            table[f"CĐ {topic + 1}"] = result.H[topic]
        table["Điểm tổng"] = result.sentence_scores
        table["Được chọn"] = ["★" if j in result.selected else "" for j in range(n)]
        table["Nội dung"] = result.sentences
        numeric = [f"CĐ {t + 1}" for t in range(result.k)] + ["Điểm tổng"]
        st.dataframe(
            table.style.format({c: "{:.3f}" for c in numeric}).highlight_max(
                subset=numeric, props="font-weight:700;background-color:#cde2fb;color:#0b0b0b"
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Các cột CĐ 1, CĐ 2... là các hàng của H: câu thuộc mỗi chủ đề nhiều hay ít. Ô tô "
            "xanh là câu mạnh nhất của chủ đề đó. Điểm tổng = tổng của (mức độ câu thuộc chủ "
            "đề × tầm quan trọng của chủ đề)."
        )

    # ----- Phần thêm ------------------------------------------------------
    with card("more"):
        _more_panel(result, n_sentences)

    # ----- Đối chiếu với thư viện -------------------------------------------
    with card("sklearn"):
        _sklearn_panel(text, k, n_sentences, strategy, min_df)

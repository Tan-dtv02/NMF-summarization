"""Màn 2: tóm tắt một văn bản và giải thích vì sao chọn những câu đó.

Gồm cả ba phần thêm: tóm tắt theo chủ đề hoặc từ khóa, bản đồ chủ đề, gợi ý k.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st
from ui_helpers import k_error_chart, tint, topic_chip, topic_color, topic_map_chart

from src.evaluate import SAMPLE_DIR
from src.summarizer import (
    STRATEGIES,
    SummaryResult,
    TextTooShortError,
    error_by_k,
    suggest_k,
    summarize,
    summarize_by_keyword,
    summarize_by_topic,
)

PASTE_OPTION = "Tự dán văn bản"


@st.cache_data
def _load_samples() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "bai_mau.csv", encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def _summarize(text: str, k: int, n: int, strategy: str, seed: int, min_df: int) -> SummaryResult:
    """Lưu lại kết quả theo đầu vào, để đổi qua lại giữa các tab không phải tính lại."""
    return summarize(text, k, n, strategy, seed, min_df)


@st.cache_data(show_spinner=False)
def _errors_by_k(V: np.ndarray, k_values: tuple[int, ...], seed: int) -> list[float]:
    return error_by_k(V, list(k_values), seed)


def _sentences_html(result: SummaryResult, highlighted: list[int]) -> str:
    """Bài gốc: mỗi câu có nền theo màu chủ đề trội, câu được chọn thì in đậm, có ★."""
    spans = []
    for j, sentence in enumerate(result.sentences):
        topic = int(result.dominant_topic[j])
        background = tint(topic_color(topic), 0.8) if topic >= 0 else "#f3f3f1"
        chosen = j in highlighted
        style = f"background:{background};"
        if chosen:
            style += f"font-weight:700;box-shadow:inset 0 -4px 0 {topic_color(topic)};"
        label = f"{'★&nbsp;' if chosen else ''}{j + 1}"
        spans.append(  # &nbsp; giữ số thứ tự dính liền với chữ đầu câu khi xuống dòng
            f'<span class="sentence" style="{style}">'
            f'<sup class="index">{label}</sup>&nbsp;{html.escape(sentence)}</span>'
        )
    return f'<div class="article">{" ".join(spans)}</div>'


def _summary_box(result: SummaryResult, indices: list[int]) -> str:
    items = "".join(
        f"<li>{topic_chip(int(result.dominant_topic[j]))} "
        f"<span class='index'>câu {j + 1}</span><br>{html.escape(result.sentences[j])}</li>"
        for j in indices
    )
    return f'<ul class="summary">{items}</ul>'


def _topic_name(result: SummaryResult, topic: int, n_words: int = 3) -> str:
    words = ", ".join(word.replace("_", " ") for word, _ in result.top_keywords(topic, n_words))
    return f"Chủ đề {topic + 1}: {words}"


def render() -> None:
    st.title("Tóm tắt văn bản bằng NMF")

    # ----- Đầu vào --------------------------------------------------------
    samples = _load_samples()
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
        st.caption("Bài mẫu do nhóm tự soạn để chạy thử, không phải bài báo thật.")

    with st.expander("Tùy chọn nâng cao"):
        a, b, c = st.columns([2, 2, 1])
        strategy = a.radio("Cách chọn câu", list(STRATEGIES), format_func=STRATEGIES.get)
        drop_rare = b.checkbox(
            "Bỏ từ chỉ xuất hiện trong một câu",
            value=True,
            help="Từ chỉ có ở một câu không nối được các câu với nhau, nên không giúp tìm chủ đề.",
        )
        seed = c.number_input("Seed", 0, 9999, 42)
    min_df = 2 if drop_rare else 1

    try:
        result = _summarize(text, k, n_sentences, strategy, int(seed), min_df)
    except TextTooShortError as message:
        st.info(str(message))
        return
    if result.k < k:
        st.caption(
            f"Bài chỉ có {len(result.sentences)} câu nên số chủ đề được giảm còn {result.k}."
        )

    # ----- Chủ đề ---------------------------------------------------------
    st.header("Các chủ đề tìm được")
    st.caption(
        f"{len(result.sentences)} câu · {len(result.vocab)} từ khóa · "
        f"NMF dừng sau {len(result.errors) - 1} vòng, sai số {result.errors[-1]:.3f}. "
        "Phần trăm là mức độ chủ đề bao phủ bài viết."
    )
    for topic, column in enumerate(st.columns(result.k)):
        words = ", ".join(w.replace("_", " ") for w, _ in result.top_keywords(topic, 6))
        column.html(
            f'<div class="topic-card" style="border-top:6px solid {topic_color(topic)}">'
            f"<b>Chủ đề {topic + 1}</b> · {result.topic_weights[topic]:.0%}"
            f"<br>{html.escape(words)}</div>"
        )

    # ----- Bài gốc và bản tóm tắt ------------------------------------------
    article_col, summary_col = st.columns([3, 2], gap="large")
    with article_col:
        st.header("Bài gốc")
        st.caption("Màu nền: chủ đề trội của câu. Chữ đậm có ★: câu được chọn vào bản tóm tắt.")
        st.html(_sentences_html(result, result.selected))
    with summary_col:
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
                subset=numeric, props="font-weight:700;background-color:#cde2fb"
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
    st.header("Khai thác thêm từ W và H")
    tab_focus, tab_map, tab_k = st.tabs(
        ["Tóm tắt theo chủ đề hoặc từ khóa", "Bản đồ chủ đề", "Gợi ý số chủ đề k"]
    )

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
            keyword = st.text_input("Từ khóa", placeholder="ví dụ: pin, giá, camera")
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

    with tab_k:
        k_values = tuple(range(2, min(6, len(result.sentences) - 1, len(result.vocab)) + 1))
        if len(k_values) < 3:
            st.info("Bài quá ngắn để so sánh nhiều giá trị k.")
        else:
            errors = _errors_by_k(result.V, k_values, int(seed))
            suggested = suggest_k(list(k_values), errors)
            st.altair_chart(k_error_chart(list(k_values), errors, suggested), width="stretch")
            st.markdown(
                f"**Gợi ý: k = {suggested}.** Tăng k thì sai số luôn giảm, nên không chọn k có "
                "sai số nhỏ nhất. Gợi ý lấy chỗ đường cong gập lại: thêm chủ đề nữa thì sai số "
                "giảm không đáng kể. Đây là cách chọn theo kinh nghiệm, nên xem thêm từ khóa "
                "của các chủ đề có rõ nghĩa không."
            )

"""Màn 1: mô phỏng thuật toán NMF từng bước trên một ví dụ nhỏ."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st
from styles import card, page_header
from ui_helpers import error_chart, matrix_html

from src.nmf_scratch import (
    has_converged,
    init_factors,
    nmf_step,
    reconstruction_error,
    sklearn_reference,
    tfidf_parts,
)
from src.preprocess import preprocess

DEFAULT_TEXT = "Pin và camera của điện thoại. Camera của điện thoại. Phở và bún. Bún và phở."
MAX_ITER = 200
TOL = 1e-4


def _reset(V: np.ndarray, k: int, seed: int, key: tuple) -> None:
    """Đưa mô phỏng về vòng 0: W, H ngẫu nhiên theo seed."""
    W, H = init_factors(V, k, seed)
    st.session_state.sim = {
        "key": key,
        "W0": W,  # giữ lại điểm khởi tạo để đối chiếu với scikit-learn
        "H0": H,
        "W": W,
        "H": H,
        "prev_W": None,  # trạng thái trước lần bấm vừa rồi, để tô màu ô thay đổi
        "prev_H": None,
        "errors": [reconstruction_error(V, W, H)],
        "stopped": False,
        "show_sklearn": False,
    }


def _advance(V: np.ndarray, until_stop: bool) -> None:
    """Chạy thêm một vòng, hoặc chạy liên tục đến khi thỏa điều kiện dừng."""
    sim = st.session_state.sim
    sim["prev_W"], sim["prev_H"] = sim["W"], sim["H"]
    while len(sim["errors"]) - 1 < MAX_ITER:
        sim["W"], sim["H"] = nmf_step(V, sim["W"], sim["H"])
        sim["errors"].append(reconstruction_error(V, sim["W"], sim["H"]))
        errors = sim["errors"]
        if has_converged(errors[-2], errors[-1], errors[0], TOL):
            sim["stopped"] = True
        if sim["stopped"] or not until_stop:
            break


def _show_sklearn() -> None:
    st.session_state.sim["show_sklearn"] = True


def _tokens_html(tokens_per_sentence: list[list[str]]) -> str:
    """Từ khóa còn lại của từng câu, mỗi từ một nhãn tròn."""
    rows = []
    for j, tokens in enumerate(tokens_per_sentence):
        chips = "".join(f'<span class="token">{html.escape(t)}</span>' for t in tokens)
        chips = chips or '<span class="token none">không còn từ nào</span>'
        rows.append(
            f'<div class="token-row"><span class="label">Câu {j + 1}</span>'
            f'<span class="tokens">{chips}</span></div>'
        )
    return f'<div class="token-list">{"".join(rows)}</div>'


def _matrix_title(name: str, note: str) -> None:
    st.html(f'<p class="matrix-title">{name} <span>{html.escape(note)}</span></p>')


def render() -> None:
    page_header(
        "Màn 1 · Thuật toán",
        "Mô phỏng thuật toán NMF",
        "NMF tách ma trận <b>V</b> (từ × câu) thành tích của hai ma trận không âm: <b>W</b> "
        "(từ × chủ đề) và <b>H</b> (chủ đề × câu), sao cho "
        '<b style="white-space:nowrap">W·H ≈ V</b>.',
    )

    # ----- Bước 1: từ văn bản đến ma trận V -------------------------------
    with card("step1"):
        st.header("Bước 1. Từ văn bản đến ma trận V")
        left, right = st.columns([2, 3], gap="large")
        with left:
            text = st.text_area("Nhập văn bản", DEFAULT_TEXT, height=130)
            doc = preprocess(text)
            if len(doc.sentences) < 2 or not any(doc.tokens):
                st.info("Hãy nhập một văn bản có ít nhất hai câu để dựng ma trận V.")
                return
            _matrix_title("Sau khi tách câu, tách từ và bỏ từ dừng", "")
            st.html(_tokens_html(doc.tokens))

        parts = tfidf_parts(doc.tokens)
        V = parts.V
        m, n = V.shape
        sentence_labels = [f"Câu {j + 1}" for j in range(n)]
        with right:
            _matrix_title("Ma trận V", f"({m} từ × {n} câu), trọng số TF-IDF")
            st.html(matrix_html(V, parts.vocab, sentence_labels))
            with st.expander("Xem cách tính TF-IDF"):
                st.latex(
                    r"V[t, d] = \underbrace{\frac{\text{số lần } t \text{ có trong câu } d}"
                    r"{\text{số từ của câu } d}}_{TF}"
                    r"\times \underbrace{\ln\frac{\text{số câu}}{\text{số câu chứa } t}}_{IDF}"
                )
                detail = pd.DataFrame(parts.tf, index=parts.vocab, columns=sentence_labels)
                detail = detail.round(3)
                detail.insert(0, "Số câu chứa từ", parts.df)
                detail.insert(1, "IDF", parts.idf.round(3))
                st.dataframe(detail, width="stretch")
                st.caption("Các cột Câu 1, Câu 2... ở bảng này là TF. Nhân với cột IDF thì ra V.")

        k_max = min(m, n) - 1
        if k_max < 1:
            st.info("Cần ít nhất hai từ khóa và hai câu khác nhau để phân rã.")
            return

    # ----- Bước 2: phân rã ------------------------------------------------
    with card("step2"):
        st.header("Bước 2. Tìm W và H bằng cách lặp")
        st.latex(
            r"H \leftarrow H \odot \frac{W^{\top}V}{W^{\top}WH}"
            r"\qquad \text{rồi} \qquad "
            r"W \leftarrow W \odot \frac{VH^{\top}}{WHH^{\top}}"
        )
        st.latex(
            r"\text{Learning rate:}\quad \eta_H = \frac{H}{W^{\top}WH}"
            r"\qquad \eta_W = \frac{W}{WHH^{\top}}"
        )
        st.caption(
            r"Thay η vào bước gradient descent $H \leftarrow H - \eta_H \odot (W^{\top}WH - "
            r"W^{\top}V)$ thì ra đúng công thức nhân ở trên. Với W cũng vậy."
        )

        option_k, option_seed, option_color = st.columns([1, 1, 2])
        k = option_k.number_input("Số chủ đề k", 1, k_max, min(2, k_max))
        seed = option_seed.number_input("Seed khởi tạo", 0, 9999, 42)
        color_by = option_color.radio(
            "Tô màu ô theo",
            ["change", "value"],
            format_func={"change": "Thay đổi sau lần bấm vừa rồi", "value": "Độ lớn"}.get,
            horizontal=True,
        )

        key = (text, int(k), int(seed))
        if st.session_state.get("sim", {}).get("key") != key:
            _reset(V, int(k), int(seed), key)
        sim = st.session_state.sim

        # Mỗi nút gọi một hàm đổi trạng thái (on_click) trước khi trang vẽ lại.
        b_next, b_run, b_reset, b_sklearn = st.columns(4)
        b_next.button(
            "Vòng tiếp theo",
            type="primary",
            width="stretch",
            disabled=sim["stopped"],
            on_click=_advance,
            args=(V, False),
        )
        b_run.button(
            "Chạy đến khi dừng",
            width="stretch",
            disabled=sim["stopped"],
            on_click=_advance,
            args=(V, True),
        )
        b_reset.button(
            "Làm lại từ đầu", width="stretch", on_click=_reset, args=(V, int(k), int(seed), key)
        )
        b_sklearn.button("So với scikit-learn", width="stretch", on_click=_show_sklearn)

        errors = sim["errors"]
        n_iter = len(errors) - 1
        m_iter, m_error, m_delta = st.columns(3)
        m_iter.metric("Vòng lặp", n_iter)
        m_error.metric("Sai số ||V − W·H||", f"{errors[-1]:.4f}")
        m_delta.metric(
            "Giảm so với vòng trước", f"{errors[-2] - errors[-1]:.4f}" if n_iter else "–"
        )
        if sim["stopped"]:
            st.success(
                f"Đã dừng sau {n_iter} vòng: sai số gần như hết giảm "
                f"(mức giảm của vòng cuối nhỏ hơn {TOL:g} lần sai số ban đầu)."
            )
        elif n_iter == 0:
            st.info("Vòng 0: W và H đang là số ngẫu nhiên. Bấm “Vòng tiếp theo” để bắt đầu.")

        topic_labels = [f"CĐ {t + 1}" for t in range(int(k))]
        col_W, col_H = st.columns([2, 3], gap="large")
        with col_W:
            _matrix_title("W", f"({m} từ × {k} chủ đề)")
            st.html(matrix_html(sim["W"], parts.vocab, topic_labels, sim["prev_W"], color_by))
        with col_H:
            _matrix_title("H", f"({k} chủ đề × {n} câu)")
            st.html(matrix_html(sim["H"], topic_labels, sentence_labels, sim["prev_H"], color_by))

        col_WH, col_V = st.columns([10, 9], gap="small")
        with col_WH:
            _matrix_title("W·H", "(tích của hai ma trận trên)")
            previous_WH = None if sim["prev_W"] is None else sim["prev_W"] @ sim["prev_H"]
            st.html(
                matrix_html(
                    sim["W"] @ sim["H"], parts.vocab, sentence_labels, previous_WH, color_by
                )
            )
        with col_V:
            _matrix_title("V", "(ma trận cần xấp xỉ, để so với W·H)")
            st.html(matrix_html(V, parts.vocab, sentence_labels, color_by="none"))
        if color_by == "change":
            st.caption("Nền xanh và ▲: ô tăng. Nền cam và ▼: ô giảm. Nền trắng: không đổi.")

        _matrix_title("Sai số sau mỗi vòng lặp", "")
        st.altair_chart(error_chart(errors), width="stretch")

    # ----- Bước 3: đọc kết quả --------------------------------------------
    if n_iter:
        with card("step3"):
            st.header("Bước 3. Đọc kết quả")
            read_W, read_H = st.columns(2, gap="large")
            with read_W:
                st.markdown("**Mỗi cột của W là một chủ đề.** Từ nào nặng nhất ở cột đó?")
                for t in range(int(k)):
                    column = sim["W"][:, t]
                    top = [
                        parts.vocab[i]
                        for i in np.argsort(-column)
                        if column[i] > 0.1 * column.max()
                    ]
                    st.markdown(f"- Chủ đề {t + 1}: **{', '.join(top)}**")
            with read_H:
                st.markdown("**Mỗi cột của H là một câu.** Câu thuộc chủ đề nào nhiều nhất?")
                for j, sentence in enumerate(doc.sentences):
                    st.markdown(
                        f"- Câu {j + 1} “{sentence}”: **Chủ đề {sim['H'][:, j].argmax() + 1}**"
                    )

    # ----- Đối chiếu với scikit-learn -------------------------------------
    if sim["show_sklearn"]:
        with card("sklearn"):
            st.header("Đối chiếu với scikit-learn")
            if n_iter == 0:
                st.info("Hãy chạy ít nhất một vòng rồi bấm lại để so sánh.")
                return
            W_sk, H_sk, error_sk = sklearn_reference(V, sim["W0"], sim["H0"], n_iter)
            st.write(f"Chạy sklearn đúng {n_iter} vòng.")
            diff = max(np.abs(sim["W"] - W_sk).max(), np.abs(sim["H"] - H_sk).max())
            c1, c2, c3 = st.columns(3)
            c1.metric("Sai số, bản tự viết", f"{errors[-1]:.6f}")
            c2.metric("Sai số, scikit-learn", f"{error_sk:.6f}")
            c3.metric("Chênh lệch lớn nhất giữa các ô", f"{diff:.1e}")
            sk_W, sk_H = st.columns([2, 3], gap="medium")
            with sk_W:
                _matrix_title("W của scikit-learn", "")
                st.html(matrix_html(W_sk, parts.vocab, topic_labels))
            with sk_H:
                _matrix_title("H của scikit-learn", "")
                st.html(matrix_html(H_sk, topic_labels, sentence_labels))

"""Màn 3: kết quả thử nghiệm trên một tập văn bản, đọc từ thư mục results/."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st
from styles import card, page_header
from ui_helpers import METHOD_GROUP_COLORS, is_ceiling, method_group, rouge_chart

from src.evaluate import RESULTS_DIR, ROOT, SAMPLE_DIR, is_sample_data, run_and_save

DATA_SUFFIXES = {".csv", ".jsonl", ".parquet"}
MAX_ARTICLES = 10_000  # số bài tối đa cho một lần chạy trên giao diện
STRATEGY_NAMES = {"coverage": "Phủ chủ đề", "score": "Điểm tổng"}
SCORE_NAMES = {
    "rouge1_f": "ROUGE-1 F1",
    "rouge2_f": "ROUGE-2 F1",
    "rougeL_f": "ROUGE-L F1",
}


def _data_files() -> list[Path]:
    """Các tệp dữ liệu có thể chạy: dữ liệu thật trong data/raw/, rồi tới dữ liệu mẫu."""
    folders = [ROOT / "data" / "raw", SAMPLE_DIR]
    return [
        path
        for folder in folders
        for path in sorted(folder.glob("*"))
        if path.suffix.lower() in DATA_SUFFIXES
    ]


def _strategy_name(info: pd.Series) -> str:
    """Tên dễ đọc của cách chọn câu đã dùng trong một lần chạy."""
    return STRATEGY_NAMES.get(str(info.get("cach_chon_cau")), "không rõ")


def _result_files() -> list[Path]:
    """Các lần chạy, mới nhất đứng đầu.

    Xếp theo ngày giờ ghi ở cuối tên tệp chứ không theo giờ sửa tệp: sau khi
    `git pull`, mọi tệp đều mang giờ tải về nên giờ sửa tệp không còn cho biết
    lần chạy nào mới hơn.
    """
    return sorted(
        RESULTS_DIR.glob("rouge_*.csv"), key=lambda p: p.stem.rsplit("_", 1)[-1], reverse=True
    )


def _run_panel() -> None:
    """Khung chạy lại thử nghiệm ngay trên giao diện: chọn tệp, số bài và cách chọn câu.

    Tên cột được tự nhận (text/summary hoặc article/abstract), xem `load_dataset`.
    """
    files = _data_files()
    with st.form("run"):
        c_file, c_limit, c_n = st.columns([3, 1, 1])
        data_path = c_file.selectbox(
            "Tệp dữ liệu",
            files,
            format_func=lambda p: f"{p.name} (dữ liệu mẫu)" if is_sample_data(p) else p.name,
            help="Tên cột được tự nhận: text/summary hoặc article/abstract.",
        )
        limit = c_limit.number_input("Số bài", 1, MAX_ARTICLES, 30)
        n_sentences = c_n.number_input("Số câu tóm tắt", 1, 6, 3)
        strategy = st.radio(
            "Cách chọn câu",
            list(STRATEGY_NAMES),
            format_func=STRATEGY_NAMES.get,
            horizontal=True,
        )
        submitted = st.form_submit_button("Chạy thử nghiệm", type="primary")

    if not submitted:
        message = st.session_state.pop("run_message", None)
        if message:
            st.success(message)
        return
    progress = st.progress(0.0, text="Đang tóm tắt và chấm điểm...")
    try:
        out_path = run_and_save(
            data_path,
            limit=int(limit),
            n_sentences=int(n_sentences),
            strategy=strategy,
            on_progress=lambda done, total: progress.progress(
                done / total, text=f"Đã xử lý {done}/{total} bài"
            ),
        )
    except ValueError as message:
        progress.empty()
        st.warning(str(message))
        return
    progress.empty()
    st.session_state.shown_result = out_path.name
    st.session_state.run_message = f"Đã lưu kết quả vào results/{out_path.name}"
    st.rerun()  # vẽ lại cả màn để ô "Lần chạy" và bảng điểm chuyển ngay sang lần vừa chạy


def _meta_html(info: pd.Series) -> str:
    """Thông tin của lần chạy đang xem, mỗi mục một nhãn tròn."""
    items = [
        ("Tệp dữ liệu", info["tep_du_lieu"]),
        ("Số bài", info["so_bai"]),
        ("Bản tóm tắt", f"{info['so_cau_tom_tat']} câu"),
        ("Cách chọn câu", _strategy_name(info)),
        ("Chạy lúc", info["ngay_chay"]),
    ]
    pills = "".join(
        f'<span class="meta-pill">{html.escape(label)}: <b>{html.escape(str(value))}</b></span>'
        for label, value in items
    )
    return f'<div class="meta-row">{pills}</div>'


def _kpi_html(table: pd.DataFrame) -> str:
    """Bốn con số ROUGE-1 để nhìn nhanh: NMF tốt nhất, từng cách đối chứng, mức trần.

    Chỉ đọc lại số trong bảng điểm bên dưới, không tính thêm gì.
    """
    nmf = table[table["phuong_phap"].map(method_group) == "NMF"]
    rows = [nmf.loc[nmf["rouge1_f"].idxmax()]] if len(nmf) else []
    rows += [row for _, row in table.iterrows() if method_group(row["phuong_phap"]) != "NMF"]
    tiles = []
    for row in rows:
        group = method_group(row["phuong_phap"])
        label = "NMF tốt nhất" if group == "NMF" else group
        tiles.append(
            f'<div class="kpi" style="--c:{METHOD_GROUP_COLORS[group]}">'
            f'<div class="kpi-group">{html.escape(label)}</div>'
            f'<div class="kpi-name">{html.escape(str(row["phuong_phap"]))}</div>'
            f'<div class="kpi-value">{row["rouge1_f"]:.3f}</div>'
            '<div class="kpi-unit">ROUGE-1 F1</div></div>'
        )
    return f'<div class="kpi-grid">{"".join(tiles)}</div>'


def render() -> None:
    page_header(
        "Màn 3 · Đánh giá",
        "Thử nghiệm và đánh giá",
        "Mỗi bài trong tập dữ liệu được tóm tắt bằng NMF với k = 2, 3, 4, 5, bằng hai cách "
        "đối chứng (lấy các câu đầu bài, chọn câu ngẫu nhiên) và bằng cách lấy các câu tốt "
        "nhất khi đã biết bản tham chiếu (mức trần). Bản tóm tắt được so với bản tham chiếu "
        "do người viết bằng ROUGE (tỉ lệ từ trùng nhau).",
    )

    with card("run"):
        files = _result_files()
        if files:
            names = [f.name for f in files]
            shown = st.session_state.get("shown_result")
            name = st.selectbox(
                "Lần chạy (mới nhất ở trên)",
                names,
                index=names.index(shown) if shown in names else 0,
            )
        with st.expander("Chạy lại thử nghiệm", expanded=not files):
            _run_panel()
        if not files:
            st.info(
                "Chưa có kết quả nào trong thư mục results/. Hãy bấm “Chạy thử nghiệm” ở trên, "
                "hoặc chạy `python -m src.evaluate --data <tệp dữ liệu>` từ dòng lệnh."
            )
            return

    table = pd.read_csv(RESULTS_DIR / name, encoding="utf-8-sig")
    info = table.iloc[0]

    if "mau" in str(info["loai_du_lieu"]):
        st.warning(
            "Đây là lần CHẠY THỬ trên dữ liệu mẫu soạn riêng, chỉ để kiểm tra chương trình "
            "chạy đúng. Các con số dưới đây không phải kết quả thử nghiệm thật.",
            icon="⚠️",
        )
    st.html(_meta_html(info))
    st.html(_kpi_html(table))

    with card("scores"):
        st.header("Điểm F1 trung bình")
        # Mức trần được xem trước bản tham chiếu nên không tham gia xếp hạng: số in
        # đậm là điểm cao nhất trong các phương pháp tóm tắt thật sự.
        methods = table[~table["phuong_phap"].map(is_ceiling)]
        best_rows = {column: methods[column].idxmax() for column in SCORE_NAMES}
        shown = table[["phuong_phap", *SCORE_NAMES]].rename(
            columns={"phuong_phap": "Phương pháp", **SCORE_NAMES}
        )

        def bold_best(column: pd.Series) -> list[str]:
            key = next((k for k, name in SCORE_NAMES.items() if name == column.name), None)
            return [
                "font-weight:700" if index == best_rows.get(key) else "" for index in column.index
            ]

        st.dataframe(
            shown.style.format({column: "{:.4f}" for column in SCORE_NAMES.values()}).apply(
                bold_best
            ),
            hide_index=True,
            width="stretch",
        )
        best = table.loc[best_rows["rouge1_f"], "phuong_phap"]
        note = (
            "Số in đậm là điểm cao nhất của mỗi cột trong các phương pháp tóm tắt. "
            f"ROUGE-1 cao nhất: {best}."
        )
        if len(methods) < len(table):
            note += (
                " Dòng “mức trần” không phải một phương pháp: nó chọn câu sau khi đã xem bản "
                "tham chiếu, để biết cách trích nguyên câu có thể đạt tới đâu trên tập này."
            )
        st.caption(note)
        st.altair_chart(rouge_chart(table))

        with st.expander("Precision và recall"):
            detail = table[["phuong_phap", *[c for c in table.columns if c.startswith("rouge")]]]
            st.dataframe(
                detail.rename(columns={"phuong_phap": "Phương pháp"}).style.format(precision=4),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "p: precision (bao nhiêu phần của bản máy tạo có trong bản tham chiếu). "
                "r: recall (bao nhiêu phần của bản tham chiếu được bản máy tạo nhắc lại). f: F1."
            )

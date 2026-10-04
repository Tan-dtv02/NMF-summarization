"""Màn 3: kết quả thử nghiệm trên một tập văn bản, đọc từ thư mục results/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from ui_helpers import rouge_chart

from src.evaluate import RESULTS_DIR, ROOT, SAMPLE_DIR, is_sample_data, run_and_save

DATA_SUFFIXES = {".csv", ".jsonl", ".parquet"}
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


def _result_files() -> list[Path]:
    return sorted(RESULTS_DIR.glob("rouge_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def _run_panel() -> None:
    """Khung chạy lại thử nghiệm trên một tập nhỏ, ngay trong buổi trình bày."""
    files = _data_files()
    with st.form("run"):
        c_file, c_limit, c_n = st.columns([3, 1, 1])
        data_path = c_file.selectbox(
            "Tệp dữ liệu",
            files,
            format_func=lambda p: f"{p.name} (dữ liệu mẫu)" if is_sample_data(p) else p.name,
        )
        limit = c_limit.number_input("Số bài", 1, 5000, 30)
        n_sentences = c_n.number_input("Số câu tóm tắt", 1, 6, 3)
        c_text, c_summary = st.columns(2)
        text_col = c_text.text_input("Tên cột nội dung", "text")
        summary_col = c_summary.text_input("Tên cột tóm tắt tham chiếu", "summary")
        submitted = st.form_submit_button("Chạy thử nghiệm", type="primary")

    if not submitted:
        return
    progress = st.progress(0.0, text="Đang tóm tắt và chấm điểm...")
    try:
        out_path = run_and_save(
            data_path,
            text_col,
            summary_col,
            limit=int(limit),
            n_sentences=int(n_sentences),
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
    st.success(f"Đã lưu kết quả vào results/{out_path.name}")


def render() -> None:
    st.title("Thử nghiệm và đánh giá")
    st.write(
        "Mỗi bài trong tập dữ liệu được tóm tắt bằng NMF với k = 2, 3, 4, 5 và bằng hai cách "
        "đối chứng: lấy các câu đầu bài, chọn câu ngẫu nhiên. Bản tóm tắt được so với bản "
        "tham chiếu do người viết bằng ROUGE (tỉ lệ từ trùng nhau)."
    )

    with st.expander("Chạy lại trên một tập nhỏ", expanded=not _result_files()):
        _run_panel()

    files = _result_files()
    if not files:
        st.info(
            "Chưa có kết quả nào trong thư mục results/. Hãy bấm “Chạy thử nghiệm” ở trên, "
            "hoặc chạy `python -m src.evaluate --data <tệp dữ liệu>` từ dòng lệnh."
        )
        return

    names = [f.name for f in files]
    shown = st.session_state.get("shown_result")
    name = st.selectbox(
        "Lần chạy (mới nhất ở trên)", names, index=names.index(shown) if shown in names else 0
    )
    table = pd.read_csv(RESULTS_DIR / name, encoding="utf-8-sig")
    info = table.iloc[0]

    if "mau" in str(info["loai_du_lieu"]):
        st.warning(
            "Đây là lần CHẠY THỬ trên dữ liệu mẫu do nhóm tự soạn, chỉ để kiểm tra chương trình "
            "chạy đúng. Các con số dưới đây không phải kết quả thử nghiệm thật.",
            icon="⚠️",
        )
    st.caption(
        f"Tệp dữ liệu: {info['tep_du_lieu']} · {info['so_bai']} bài · "
        f"bản tóm tắt {info['so_cau_tom_tat']} câu · seed {info['seed']} · chạy lúc {info['ngay_chay']}"
    )

    st.header("Điểm F1 trung bình")
    best = table.loc[table["rouge1_f"].idxmax(), "phuong_phap"]
    st.dataframe(
        table[["phuong_phap", *SCORE_NAMES]]
        .rename(columns={"phuong_phap": "Phương pháp", **SCORE_NAMES})
        .style.format({column: "{:.4f}" for column in SCORE_NAMES.values()})
        .highlight_max(subset=list(SCORE_NAMES.values()), props="font-weight:700"),
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Số in đậm là điểm cao nhất của mỗi cột. ROUGE-1 cao nhất: {best}.")
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

    st.caption(
        "Cách đọc: bản tham chiếu thường do người viết diễn đạt lại chứ không trích nguyên câu, "
        "nên điểm tuyệt đối của mọi phương pháp trích xuất đều không cao. Điều đáng so sánh là "
        "thứ hạng giữa các phương pháp."
    )

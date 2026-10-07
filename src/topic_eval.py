"""Thử nghiệm "tách chủ đề": mô hình có tìm đúng các mảng nội dung của một bài không.

ROUGE (xem evaluate.py) chỉ cho biết bản tóm tắt giống bản tham chiếu tới đâu.
Thử nghiệm này trả lời câu hỏi đứng trước đó: NMF có chia các câu về đúng chủ
đề không. Muốn chấm được thì phải biết trước đáp án, nên dùng văn bản CÓ NHÃN:

  1. Bài ghép: lấy 2, 3 hoặc 4 trong 6 bài ngắn một chủ đề (data/sample/
     bai_ngan_mot_chu_de.csv) nối lại thành một văn bản. Nhãn của mỗi câu là bài
     gốc của nó. Có 15 + 20 + 15 = 50 văn bản như vậy.
  2. Bài demo: 5 bản tin tổng hợp hiện ở màn 2 (data/sample/bai_mau.csv). Nhãn
     của mỗi câu là đoạn văn chứa nó, vì mỗi mảng tin được viết thành một đoạn.

Với mỗi văn bản, đặt k bằng số mảng thật rồi đo:

  - câu đúng nhóm: ghép mỗi chủ đề với một mảng sao cho khớp nhiều nhất, rồi tính
    tỉ lệ câu có chủ đề trội trùng với mảng của nó;
  - phủ đủ mảng: bản tóm tắt k câu (và 2k câu) có nhắc tới mọi mảng hay không,
    với cả hai cách chọn câu.

Mỗi phép đo được làm cho hai pipeline: "cũ" (TF-IDF thô, một seed) và "mới"
(chuẩn hóa độ dài câu, chạy 5 seed tới hội tụ rồi giữ lần sai số nhỏ nhất).
Pipeline cũ chỉ tồn tại ở tệp này, để có bảng "trước và sau" từ lần chạy thật;
ứng dụng không dùng nó.

Chạy từ dòng lệnh (đứng ở thư mục gốc của dự án):

    python -m src.topic_eval

Kết quả in ra màn hình và lưu vào results/tach_chu_de_<ngày giờ>.csv.

LƯU Ý khi đọc kết quả: các bài này đều là bài mẫu soạn để chạy thử, mỗi mảng nói
về một việc khác hẳn nhau. Bài báo thật khó tách hơn. 5 bài demo lại được soạn
sau khi đã có pipeline mới, nên chỉ báo riêng một dòng, không gộp vào bài ghép.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import RESULTS_DIR, SAMPLE_DIR
from src.nmf_scratch import build_tfidf_matrix, nmf, normalize_topics
from src.preprocess import ProcessedText, preprocess, split_sentences
from src.summarizer import (
    TextTooShortError,
    choose_tokens,
    select_by_coverage,
    select_top,
    summarize_processed,
)

SHORT_ARTICLES = SAMPLE_DIR / "bai_ngan_mot_chu_de.csv"
DEMO_ARTICLES = SAMPLE_DIR / "bai_mau.csv"
STRATEGY_NAMES = {"coverage": "phu_chu_de", "score": "diem_tong"}
PIPELINE_NAMES = {"old": "cũ", "new": "mới"}


@dataclass(frozen=True)
class LabeledDocument:
    """Một văn bản đã tách từ, kèm đáp án: mỗi câu thuộc mảng nào."""

    group: str  # "bài ghép 2 mảng", "bài ghép 3 mảng", "bài ghép 4 mảng" hoặc "bài demo"
    name: str
    doc: ProcessedText
    labels: np.ndarray  # (số câu,) chỉ số mảng của từng câu, từ 0

    @property
    def n_parts(self) -> int:
        return int(self.labels.max()) + 1


# ---------------------------------------------------------------------------
# Phần A. Dựng các văn bản có nhãn
# ---------------------------------------------------------------------------


def combined_documents(sizes: tuple[int, ...] = (2, 3, 4)) -> list[LabeledDocument]:
    """Ghép mọi tổ hợp bài ngắn; nhãn của mỗi câu là bài gốc của nó."""
    table = pd.read_csv(SHORT_ARTICLES, encoding="utf-8-sig")
    articles = [preprocess(text) for text in table["text"]]  # tách từ một lần, dùng lại
    documents = []
    for size in sizes:
        for combo in itertools.combinations(range(len(articles)), size):
            sentences = [s for a in combo for s in articles[a].sentences]
            tokens = [t for a in combo for t in articles[a].tokens]
            labels = [part for part, a in enumerate(combo) for _ in articles[a].sentences]
            documents.append(
                LabeledDocument(
                    group=f"bài ghép {size} mảng",
                    name=" + ".join(str(table["id"][a]) for a in combo),
                    doc=ProcessedText(sentences, tokens),
                    labels=np.array(labels),
                )
            )
    return documents


def demo_documents() -> list[LabeledDocument]:
    """5 bản tin của màn 2; nhãn của mỗi câu là đoạn văn (mảng tin) chứa nó."""
    table = pd.read_csv(DEMO_ARTICLES, encoding="utf-8-sig")
    documents = []
    for name, text in zip(table["id"], table["text"]):
        paragraphs = [p for p in text.splitlines() if p.strip()]
        labels = [part for part, p in enumerate(paragraphs) for _ in split_sentences(p)]
        documents.append(LabeledDocument("bài demo", str(name), preprocess(text), np.array(labels)))
    return documents


# ---------------------------------------------------------------------------
# Phần B. Hai pipeline: mỗi cái trả về (chủ đề trội của từng câu, các câu được chọn)
# ---------------------------------------------------------------------------


def run_new_pipeline(
    doc: ProcessedText, k: int, n_sentences: int, strategy: str, min_df: int
) -> tuple[np.ndarray, list[int]]:
    """Pipeline của ứng dụng hiện tại (gọi đúng hàm mà màn 2 và màn 3 dùng)."""
    result = summarize_processed(doc, k, n_sentences, strategy, min_df)
    return result.dominant_topic, result.selected


def run_old_pipeline(
    doc: ProcessedText, k: int, n_sentences: int, strategy: str, min_df: int
) -> tuple[np.ndarray, list[int]]:
    """Pipeline trước khi sửa, dựng lại để so sánh.

    Khác pipeline mới ở hai chỗ: V là TF-IDF thô (không chuẩn hóa độ dài câu),
    và NMF chỉ chạy một lần với seed 42 và ngưỡng dừng mặc định (1e-4, 200 vòng).
    """
    tokens = choose_tokens(doc.tokens, min_df, k, n_sentences)
    V, vocab = build_tfidf_matrix(tokens)
    k = max(1, min(k, len(doc.sentences) - 1, len(vocab)))
    result = nmf(V, k, seed=42)
    _, H = normalize_topics(result.W, result.H)

    topic_weights = H.sum(axis=1) / H.sum()
    if strategy == "score":
        selected = select_top(topic_weights @ H, n_sentences)
    else:
        selected = select_by_coverage(H, topic_weights, n_sentences)
    dominant_topic = np.where(V.sum(axis=0) > 0, H.argmax(axis=0), -1)
    return dominant_topic, selected


PIPELINES = {"old": run_old_pipeline, "new": run_new_pipeline}


# ---------------------------------------------------------------------------
# Phần C. Hai phép đo
# ---------------------------------------------------------------------------


def sentence_accuracy(dominant_topic: np.ndarray, labels: np.ndarray) -> float:
    """Tỉ lệ câu được xếp đúng mảng, theo cách ghép chủ đề với mảng có lợi nhất.

    NMF đánh số chủ đề tùy ý, nên phải thử mọi cách ghép "chủ đề nào ứng với
    mảng nào" (mỗi mảng một chủ đề) rồi lấy cách khớp nhiều câu nhất. Số mảng
    không quá 4 nên chỉ có tối đa 24 cách ghép. Câu không còn từ khóa (chủ đề
    trội = -1) được tính là xếp sai.
    """
    n_parts = int(labels.max()) + 1
    n_topics = max(int(dominant_topic.max()) + 1, n_parts)
    best = max(
        sum(
            int(((dominant_topic == topic) & (labels == part)).sum())
            for part, topic in enumerate(match)
        )
        for match in itertools.permutations(range(n_topics), n_parts)
    )
    return best / len(labels)


def covers_all_parts(selected: list[int], labels: np.ndarray) -> bool:
    """Bản tóm tắt có ít nhất một câu của mỗi mảng hay không."""
    return {int(labels[j]) for j in selected} == set(range(int(labels.max()) + 1))


# ---------------------------------------------------------------------------
# Phần D. Chạy và tổng hợp
# ---------------------------------------------------------------------------


def evaluate_documents(documents: list[LabeledDocument]) -> pd.DataFrame:
    """Chạy hai pipeline, hai chế độ lọc từ hiếm trên từng văn bản. Mỗi dòng là một lần chạy."""
    rows = []
    for item in documents:
        k = item.n_parts
        for pipeline, run in PIPELINES.items():
            for min_df in (2, 1):
                row = {
                    "nhom": item.group,
                    "van_ban": item.name,
                    "pipeline": PIPELINE_NAMES[pipeline],
                    "loc_tu_hiem": "bật" if min_df == 2 else "tắt",
                    "so_mang": k,
                    "so_cau": len(item.labels),
                }
                try:
                    for multiple, strategy in itertools.product((1, 2), STRATEGY_NAMES):
                        dominant_topic, selected = run(item.doc, k, multiple * k, strategy, min_df)
                        if "cau_dung_nhom" not in row:
                            # chủ đề trội không phụ thuộc cách chọn câu: chấm một lần là đủ
                            row["cau_dung_nhom"] = sentence_accuracy(dominant_topic, item.labels)
                        size = "k" if multiple == 1 else "2k"
                        column = f"phu_du_{STRATEGY_NAMES[strategy]}_{size}_cau"
                        row[column] = float(covers_all_parts(selected, item.labels))
                except TextTooShortError:
                    continue
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Trung bình theo nhóm văn bản, pipeline và chế độ lọc từ hiếm."""
    coverage_columns = [c for c in runs.columns if c.startswith("phu_du_")]
    grouped = runs.groupby(["nhom", "loc_tu_hiem", "pipeline"], sort=False)
    table = grouped.agg(
        so_van_ban=("van_ban", "count"),
        cau_dung_nhom=("cau_dung_nhom", "mean"),
        cau_dung_nhom_thap_nhat=("cau_dung_nhom", "min"),
        **{column: (column, "mean") for column in coverage_columns},
    )
    return table.reset_index()


def run_and_save(sizes: tuple[int, ...] = (2, 3, 4)) -> tuple[Path, pd.DataFrame]:
    """Chạy trọn thử nghiệm, lưu bảng tổng hợp vào results/, trả về (đường dẫn, bảng)."""
    runs = evaluate_documents(combined_documents(sizes) + demo_documents())
    table = summarize_runs(runs)
    now = datetime.now()
    table["ngay_chay"] = now.strftime("%Y-%m-%d %H:%M")
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"tach_chu_de_{now:%Y%m%d-%H%M%S}.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path, table


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows: ép UTF-8 để in được tiếng Việt
        sys.stdout.reconfigure(encoding="utf-8")
    print("Đang chạy thử nghiệm tách chủ đề trên 50 bài ghép và 5 bài demo...", flush=True)
    out_path, table = run_and_save()

    shown = table.drop(columns="ngay_chay").rename(
        columns={
            "nhom": "nhóm văn bản",
            "loc_tu_hiem": "lọc từ hiếm",
            "so_van_ban": "số bài",
            "cau_dung_nhom": "câu đúng nhóm",
            "cau_dung_nhom_thap_nhat": "thấp nhất",
            "phu_du_phu_chu_de_k_cau": "phủ đủ: Phủ CĐ, k câu",
            "phu_du_diem_tong_k_cau": "Điểm tổng, k câu",
            "phu_du_phu_chu_de_2k_cau": "Phủ CĐ, 2k câu",
            "phu_du_diem_tong_2k_cau": "Điểm tổng, 2k câu",
        }
    )
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print()
        print(shown.round(3).to_string(index=False))
    print(f"\nĐã lưu: {out_path}")
    print(
        "LƯU Ý: đây là bài mẫu soạn để chạy thử, mỗi mảng nói về một việc khác hẳn nhau. "
        "Bài báo thật khó tách hơn."
    )


if __name__ == "__main__":
    main()

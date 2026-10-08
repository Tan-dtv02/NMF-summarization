"""Thử nghiệm trên một tập văn bản và đánh giá bằng ROUGE.

Tệp này làm ba việc:
  1. Tính ROUGE-1, ROUGE-2, ROUGE-L (tự viết, đếm theo âm tiết tiếng Việt).
  2. Cung cấp ba mốc so sánh: lấy n câu đầu, chọn n câu ngẫu nhiên (hai cách đối
     chứng), và n câu tốt nhất khi biết trước bản tham chiếu (mức trần).
  3. Chạy tất cả phương pháp trên một tập dữ liệu rồi lưu bảng điểm vào results/.

Chạy từ dòng lệnh (đứng ở thư mục gốc của dự án):

    python -m src.evaluate --data data/sample/bai_mau.csv
    python -m src.evaluate --data data/raw/test_desegmented.jsonl --limit 300

ĐỊNH DẠNG DỮ LIỆU: tệp .csv, .jsonl hoặc .parquet, mỗi dòng một bài, có hai cột:
nội dung bài và bản tóm tắt tham chiếu do người viết. Hai cặp tên cột được tự
nhận: text/summary (dữ liệu mẫu) và article/abstract (VietNews). Tệp dùng tên
cột khác thì chỉ ra bằng --text-col và --summary-col.

Vì sao tự viết ROUGE: thư viện `rouge-score` xóa mọi ký tự ngoài a-z và 0-9
trước khi đếm, nên "điện thoại" bị cắt thành "i", "n", "tho", "i". Điểm tính ra
vẫn là một con số nhưng sai hoàn toàn với tiếng Việt có dấu.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.preprocess import preprocess
from src.summarizer import TextTooShortError, summarize_processed

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SAMPLE_DIR = ROOT / "data" / "sample"
SAMPLE_LABEL = "MAU-CHAY-THU"  # gắn vào tên tệp kết quả khi chạy trên dữ liệu mẫu

_NOT_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


# ---------------------------------------------------------------------------
# Phần A. ROUGE
# ---------------------------------------------------------------------------


def rouge_tokens(text: str) -> list[str]:
    """Đổi văn bản thành dãy âm tiết để so khớp: chữ thường, bỏ dấu câu.

    Đếm theo âm tiết (tách bằng khoảng trắng) chứ không theo từ ghép, để điểm
    số không phụ thuộc vào công cụ tách từ. Dấu thanh và dấu mũ được giữ nguyên.
    """
    text = unicodedata.normalize("NFC", text).lower()
    return _NOT_WORD.sub(" ", text).split()


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _prf(overlap: int, n_candidate: int, n_reference: int) -> dict[str, float]:
    """Precision, recall và F1 từ số phần tử trùng nhau."""
    precision = overlap / n_candidate if n_candidate else 0.0
    recall = overlap / n_reference if n_reference else 0.0
    total = precision + recall
    return {"p": precision, "r": recall, "f": 2 * precision * recall / total if total else 0.0}


def rouge_n(candidate: str, reference: str, n: int = 1) -> dict[str, float]:
    """ROUGE-N: tỉ lệ cụm n âm tiết liền nhau trùng giữa hai bản.

    precision = số cụm trùng / số cụm của bản máy tạo
    recall    = số cụm trùng / số cụm của bản tham chiếu
    Một cụm xuất hiện 2 lần ở bản này và 1 lần ở bản kia chỉ tính trùng 1 lần.
    """
    cand = _ngrams(rouge_tokens(candidate), n)
    ref = _ngrams(rouge_tokens(reference), n)
    overlap = sum((cand & ref).values())
    return _prf(overlap, sum(cand.values()), sum(ref.values()))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Độ dài dãy con chung dài nhất (quy hoạch động, chỉ giữ một hàng)."""
    row = [0] * (len(b) + 1)
    for x in a:
        diagonal = 0  # giá trị ô chéo trên-trái của ô đang xét
        for j, y in enumerate(b, start=1):
            above = row[j]
            row[j] = diagonal + 1 if x == y else max(row[j], row[j - 1])
            diagonal = above
    return row[-1]


def rouge_l(candidate: str, reference: str) -> dict[str, float]:
    """ROUGE-L: dựa trên dãy con chung dài nhất (giữ thứ tự, không cần liền nhau)."""
    cand, ref = rouge_tokens(candidate), rouge_tokens(reference)
    return _prf(_lcs_length(cand, ref), len(cand), len(ref))


def rouge_all(candidate: str, reference: str) -> dict[str, float]:
    """Cả ba độ đo, trả về dạng phẳng: rouge1_p, rouge1_r, rouge1_f, rouge2_p, ..."""
    scores = {
        "rouge1": rouge_n(candidate, reference, 1),
        "rouge2": rouge_n(candidate, reference, 2),
        "rougeL": rouge_l(candidate, reference),
    }
    return {f"{name}_{part}": value for name, prf in scores.items() for part, value in prf.items()}


# ---------------------------------------------------------------------------
# Phần B. Các mốc so sánh: hai cách đối chứng và mức trần
# ---------------------------------------------------------------------------


def lead_n(sentences: list[str], n: int) -> list[int]:
    """Lấy n câu đầu của bài. Với tin tức đây là đối thủ rất mạnh."""
    return list(range(min(n, len(sentences))))


def random_n(sentences: list[str], n: int, seed: int) -> list[int]:
    """Chọn ngẫu nhiên n câu (cố định seed để chạy lại ra cùng kết quả)."""
    rng = random.Random(seed)
    return sorted(rng.sample(range(len(sentences)), min(n, len(sentences))))


def oracle_n(sentences: list[str], reference: str, n: int) -> list[int]:
    """Mức trần ước lượng: n câu của bài cho ROUGE-1 F1 cao nhất so với bản tham chiếu.

    Đây không phải một phương pháp tóm tắt, vì nó được "nhìn đáp án". Nó cho
    biết một bản tóm tắt TRÍCH NGUYÊN CÂU có thể đạt tới đâu trên bài này: bản
    tham chiếu do người viết lại bằng lời khác, nên ngay cả chọn câu hoàn hảo
    cũng không đạt điểm 1.

    Cách chọn là tham lam: mỗi lượt thêm câu làm điểm của cả bản tóm tắt cao
    nhất, cho đến khi đủ n câu. Thử hết mọi bộ n câu thì chậm, còn tham lam cho
    kết quả rất gần mức tốt nhất, nên gọi là mức trần ƯỚC LƯỢNG. Bằng điểm thì
    giữ câu đứng trước trong bài.
    """
    reference_counts = Counter(rouge_tokens(reference))
    n_reference = sum(reference_counts.values())
    sentence_counts = [Counter(rouge_tokens(s)) for s in sentences]

    chosen: list[int] = []
    summary_counts: Counter = Counter()
    for _ in range(min(n, len(sentences))):
        best_index, best_f = -1, -1.0
        for j, counts in enumerate(sentence_counts):
            if j in chosen:
                continue
            candidate = summary_counts + counts
            overlap = sum((candidate & reference_counts).values())
            f = _prf(overlap, sum(candidate.values()), n_reference)["f"]
            if f > best_f:
                best_index, best_f = j, f
        chosen.append(best_index)
        summary_counts += sentence_counts[best_index]
    return sorted(chosen)


def oracle_label(n: int) -> str:
    """Tên hiển thị của mức trần, dùng chung cho bảng điểm và giao diện."""
    return f"{n} câu tốt nhất (mức trần)"


# ---------------------------------------------------------------------------
# Phần C. Chạy trên cả tập dữ liệu
# ---------------------------------------------------------------------------


# Các cặp tên cột (nội dung, tóm tắt tham chiếu) được tự nhận, xét theo thứ tự này.
KNOWN_COLUMNS = (("text", "summary"), ("article", "abstract"))


def detect_columns(columns) -> tuple[str, str] | None:
    """Cặp tên cột quen thuộc đầu tiên có đủ trong tệp, hoặc None nếu không có cặp nào."""
    for text_col, summary_col in KNOWN_COLUMNS:
        if text_col in columns and summary_col in columns:
            return text_col, summary_col
    return None


def load_dataset(
    path: str | Path, text_col: str | None = None, summary_col: str | None = None
) -> pd.DataFrame:
    """Đọc tập dữ liệu, trả về bảng có đúng hai cột `text` và `summary`.

    Không chỉ ra tên cột thì tự nhận theo KNOWN_COLUMNS.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    elif suffix in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=suffix == ".jsonl")
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Chưa hỗ trợ định dạng {suffix}. Hãy dùng .csv, .jsonl hoặc .parquet.")

    if text_col is None and summary_col is None:
        found = detect_columns(df.columns)
        if found is None:
            raise ValueError(
                f"Không tự nhận được tên cột của tệp {path.name}: cần có cặp cột text/summary "
                f"hoặc article/abstract. Các cột hiện có: {list(df.columns)}. Tệp có tên cột "
                "khác thì chạy bằng dòng lệnh với --text-col và --summary-col."
            )
        text_col, summary_col = found
    elif text_col is None or summary_col is None:
        raise ValueError("Cần chỉ ra cả tên cột nội dung lẫn tên cột tóm tắt tham chiếu.")

    missing = [c for c in (text_col, summary_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Tệp {path.name} thiếu cột {missing}. Các cột hiện có: {list(df.columns)}. "
            "Hãy chỉ ra tên cột bằng --text-col và --summary-col."
        )
    df = df[[text_col, summary_col]].rename(columns={text_col: "text", summary_col: "summary"})
    return df.dropna().astype(str).reset_index(drop=True)


def evaluate_dataset(
    df: pd.DataFrame,
    k_values: tuple[int, ...] = (2, 3, 4, 5),
    n_sentences: int = 3,
    strategy: str = "coverage",
    seed: int = 42,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Chấm điểm mọi phương pháp trên mọi bài, trả về (bảng điểm trung bình, số bài dùng được).

    Mỗi bài được tóm tắt bằng: NMF với từng giá trị k, n câu đầu, n câu ngẫu
    nhiên, và n câu tốt nhất khi biết trước bản tham chiếu (mức trần).
    Bài nào quá ngắn để tóm tắt thì bỏ qua cho TẤT CẢ phương pháp, để các
    phương pháp luôn được so trên cùng một tập bài.
    """
    rows = []
    n_used = 0
    for index, (text, reference) in enumerate(zip(df["text"], df["summary"])):
        if on_progress:
            on_progress(index + 1, len(df))
        doc = preprocess(text)  # tách từ một lần, dùng lại cho mọi k
        try:
            candidates = {
                f"NMF (k={k})": summarize_processed(doc, k, n_sentences, strategy).selected
                for k in k_values
            }
        except TextTooShortError:
            continue
        candidates[f"{n_sentences} câu đầu"] = lead_n(doc.sentences, n_sentences)
        candidates[f"{n_sentences} câu ngẫu nhiên"] = random_n(
            doc.sentences, n_sentences, seed + index
        )
        candidates[oracle_label(n_sentences)] = oracle_n(doc.sentences, reference, n_sentences)
        n_used += 1
        for method, selected in candidates.items():
            summary = " ".join(doc.sentences[j] for j in selected)
            rows.append({"phuong_phap": method, **rouge_all(summary, reference)})

    if not rows:
        return pd.DataFrame(), 0
    table = pd.DataFrame(rows).groupby("phuong_phap", sort=False).mean().reset_index()
    return table, n_used


def is_sample_data(path: str | Path) -> bool:
    """Tệp nằm trong data/sample/ là dữ liệu mẫu soạn để chạy thử, không phải dữ liệu thật."""
    return SAMPLE_DIR in Path(path).resolve().parents


def run_and_save(
    data_path: str | Path,
    text_col: str | None = None,
    summary_col: str | None = None,
    limit: int | None = None,
    k_values: tuple[int, ...] = (2, 3, 4, 5),
    n_sentences: int = 3,
    strategy: str = "coverage",
    seed: int = 42,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Chạy thử nghiệm rồi lưu bảng điểm thành một tệp CSV trong results/.

    Mỗi dòng của tệp kết quả ghi kèm ngày chạy, tệp dữ liệu, số bài, seed, để
    con số nào trên slide cũng truy ngược được nó sinh ra từ lần chạy nào.
    """
    data_path = Path(data_path)
    df = load_dataset(data_path, text_col, summary_col)
    if limit:
        df = df.head(limit)
    table, n_used = evaluate_dataset(df, k_values, n_sentences, strategy, seed, on_progress)
    if table.empty:
        raise ValueError("Không có bài nào đủ dài để tóm tắt trong tệp dữ liệu này.")

    sample = is_sample_data(data_path)
    now = datetime.now()
    table["so_bai"] = n_used
    table["so_cau_tom_tat"] = n_sentences
    table["cach_chon_cau"] = strategy
    table["seed"] = seed
    table["tep_du_lieu"] = data_path.name
    table["loai_du_lieu"] = "du lieu mau (chay thu)" if sample else "du lieu that"
    table["ngay_chay"] = now.strftime("%Y-%m-%d %H:%M")

    RESULTS_DIR.mkdir(exist_ok=True)
    label = SAMPLE_LABEL if sample else data_path.stem
    out_path = RESULTS_DIR / f"rouge_{label}_{now:%Y%m%d-%H%M%S}.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main() -> None:
    # Trên Windows, khi kết quả in ra được chuyển vào tệp thì bảng mã mặc định
    # không chứa chữ tiếng Việt. Ép về UTF-8 để không lỗi khi in.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Thử nghiệm tóm tắt bằng NMF và chấm ROUGE.")
    parser.add_argument("--data", required=True, help="tệp .csv, .jsonl hoặc .parquet")
    parser.add_argument(
        "--text-col", default=None, help="tên cột nội dung bài (mặc định: tự nhận text/article)"
    )
    parser.add_argument(
        "--summary-col",
        default=None,
        help="tên cột tóm tắt tham chiếu (mặc định: tự nhận summary/abstract)",
    )
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy trên N bài đầu")
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 5], help="các giá trị k")
    parser.add_argument("--n-sentences", type=int, default=3, help="số câu của bản tóm tắt")
    parser.add_argument(
        "--strategy",
        choices=["coverage", "score"],
        default="coverage",
        help="cách chọn câu: coverage = phủ chủ đề, score = điểm tổng",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="hạt giống của cách chọn câu ngẫu nhiên"
    )
    args = parser.parse_args()

    def show_progress(done: int, total: int) -> None:
        if done % 20 == 0 or done == total:
            print(f"  đã xử lý {done}/{total} bài", flush=True)

    out_path = run_and_save(
        args.data,
        args.text_col,
        args.summary_col,
        args.limit,
        tuple(args.k),
        args.n_sentences,
        args.strategy,
        args.seed,
        show_progress,
    )
    table = pd.read_csv(out_path, encoding="utf-8-sig")
    print()
    print(
        table[["phuong_phap", "rouge1_f", "rouge2_f", "rougeL_f"]].round(4).to_string(index=False)
    )
    print(f"\nSố bài dùng được: {table['so_bai'].iloc[0]}. Đã lưu: {out_path}")
    if is_sample_data(args.data):
        print("LƯU Ý: đây là dữ liệu mẫu, chỉ để chạy thử, không phải kết quả thử nghiệm thật.")


if __name__ == "__main__":
    main()

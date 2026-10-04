# NMF và tóm tắt văn bản dựa trên chủ đề (BTL 2)

Bài tập lớn 2, môn Phân tích và khai phá dữ liệu văn bản, đề tài 4.

Chương trình gồm ba phần, ứng với ba màn của ứng dụng:

1. **Mô phỏng NMF**: chạy thuật toán từng vòng trên một ví dụ 4 câu, xem V, W, H và sai số thay đổi.
2. **Tóm tắt văn bản**: dùng chính thuật toán đó để tóm tắt trích xuất một bài bất kỳ, kèm lời giải thích vì sao chọn từng câu.
3. **Thử nghiệm**: chạy trên một tập văn bản, chấm điểm bằng ROUGE, so với hai cách đối chứng.

## Cài đặt

Cần Python 3.12 trở lên. Mở terminal ở thư mục gốc của dự án:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy ứng dụng

```
python -m streamlit run app\streamlit_app.py
```

Trình duyệt tự mở ở `http://localhost:8501`. Chọn màn ở cột bên trái.

## Chạy thử nghiệm từ dòng lệnh

```
python -m src.evaluate --data data\sample\bai_mau.csv
python -m src.evaluate --data data\raw\ten_tep.csv --limit 300 --text-col article --summary-col abstract
```

Kết quả được lưu thành một tệp CSV trong `results/` và hiện ở màn 3. Các tham số:

| Tham số                       | Ý nghĩa                                          | Mặc định          |
| ----------------------------- | ------------------------------------------------ | ----------------- |
| `--data`                      | tệp `.csv`, `.jsonl` hoặc `.parquet`             | bắt buộc          |
| `--text-col`, `--summary-col` | tên cột nội dung và cột tóm tắt tham chiếu       | `text`, `summary` |
| `--limit`                     | chỉ chạy trên N bài đầu                          | tất cả            |
| `--k`                         | các giá trị k, ví dụ `--k 2 3 4 5`               | 2 3 4 5           |
| `--n-sentences`               | số câu của bản tóm tắt                           | 3                 |
| `--strategy`                  | `coverage` (phủ chủ đề) hoặc `score` (điểm tổng) | `coverage`        |
| `--seed`                      | hạt giống ngẫu nhiên                             | 42                |

## Dữ liệu

- `data/sample/bai_mau.csv`: 6 bài mẫu **do nhóm tự soạn** (không phải báo thật), chỉ để chạy thử. Kết quả chạy trên tệp này được ghi nhãn `MAU-CHAY-THU` và không phải kết quả thử nghiệm.
- `data/raw/`: nơi đặt bộ dữ liệu thật. Thư mục này không đưa lên Git. Xem định dạng trong `data/README.md`.

## Kiểm thử

```
python -m pytest
```

Có 63 bài kiểm tra: NMF tự viết so với scikit-learn, TF-IDF và ROUGE so với tính tay, tóm tắt, và ba màn của giao diện.

## Cấu trúc thư mục

```
src/
  preprocess.py        tách câu, tách từ, bỏ từ dừng
  nmf_scratch.py       dựng ma trận V (TF-IDF) và NMF tự viết bằng NumPy
  summarizer.py        chọn câu từ W và H; tóm tắt theo chủ đề, theo từ khóa; gợi ý k
  evaluate.py          ROUGE tự viết, hai cách đối chứng, chạy cả tập
app/
  streamlit_app.py     điểm vào, chọn màn
  screen_simulation.py màn 1
  screen_summary.py    màn 2
  screen_experiment.py màn 3
  ui_helpers.py        màu chủ đề, bảng ma trận, biểu đồ
tests/                 các bài kiểm tra
data/
  vietnamese-stopwords.txt  kho từ dừng công khai (1.942 từ, giấy phép MIT)
  stopwords_vi.txt          phần thêm bớt của nhóm (sửa trực tiếp được)
  sample/              bài mẫu tự soạn
  raw/                 dữ liệu thật (không đưa lên Git)
results/               bảng điểm của các lần chạy thử nghiệm
.streamlit/config.toml giao diện sáng, chữ lớn cho máy chiếu
```

## Ký hiệu

Theo slide chương 4: **V** (m từ × n câu), **W** (m × k, mỗi cột là một chủ đề), **H** (k × n, mỗi cột là một câu). scikit-learn đặt tài liệu theo hàng nên kết quả của thư viện là chuyển vị của các ma trận này; hàm `sklearn_reference` trong `src/nmf_scratch.py` đã đổi lại.

## Các quyết định thiết kế cần biết khi trình bày

- **Ví dụ của màn 1** là "Pin và camera của điện thoại / Camera của điện thoại / Phở và bún / Bún và phở". Nếu câu đầu chỉ là "Pin của điện thoại" thì với TF-IDF, hai cách tách chủ đề cho sai số bằng nhau và kết quả thay đổi theo seed. Với ví dụ hiện tại, 10 seed thử đều tách đúng hai nhóm (có bài kiểm tra).
- **TF-IDF** tính đúng công thức trong slide: TF = số lần / số từ của câu, IDF = ln(số câu / số câu chứa từ). Không dùng `TfidfVectorizer` vì nó làm mịn và chuẩn hóa khác, số sẽ lệch với tính tay.
- **Từ dừng** lấy từ kho công khai `vietnamese-stopwords` (Van-Duyet Le, giấy phép MIT), cộng một tệp nhỏ của nhóm để thêm từ còn thiếu và giữ lại vài từ mang nội dung mà kho coi là từ dừng ("nước", "bán"). Kho lớn làm từ khóa chủ đề sạch hơn, nhưng với bài rất ngắn thì còn lại ít từ.
- **Lọc từ hiếm** khi tóm tắt: bỏ từ chỉ xuất hiện trong một câu, vì chúng không nối được các câu với nhau. Không lọc thì từ khóa của chủ đề toàn từ hiếm. Tắt được ở "Tùy chọn nâng cao" của màn 2 để so sánh.
- **Cách chọn câu** mặc định là phủ chủ đề: mỗi chủ đề lần lượt góp câu mạnh nhất. Cách thứ hai là lấy các câu có điểm tổng cao nhất (Lee và cộng sự, 2009), dễ dồn hết vào một chủ đề.
- **ROUGE** đếm theo âm tiết (tách bằng khoảng trắng), giữ nguyên dấu. Thư viện `rouge-score` không dùng được vì nó xóa mọi ký tự ngoài a-z, 0-9.
- **Điều kiện dừng** của NMF: mức giảm sai số của một vòng nhỏ hơn 1e-4 lần sai số ban đầu, hoặc đủ 200 vòng.

# NMF và tóm tắt văn bản dựa trên chủ đề (BTL 2)

Bài tập lớn 2, môn Phân tích và khai phá dữ liệu văn bản, đề tài 4.

Chương trình gồm ba phần, ứng với ba màn của ứng dụng:

1. **Mô phỏng NMF**: chạy thuật toán từng vòng trên một ví dụ 4 câu, xem V, W, H và sai số thay đổi.
2. **Tóm tắt văn bản**: dùng chính thuật toán đó để tóm tắt trích xuất một bài bất kỳ, tô màu từng câu theo chủ đề, giải thích vì sao chọn từng câu, và đối chiếu với scikit-learn ngay trên bài đang xem.
3. **Thử nghiệm**: chạy trên một tập văn bản, chấm điểm bằng ROUGE, so với hai cách đối chứng (câu đầu bài, câu ngẫu nhiên) và một mức trần (các câu tốt nhất khi biết trước bản tham chiếu).

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

Trình duyệt tự mở ở `http://localhost:8501`. Chọn màn ở thanh điều hướng trên cùng. Nếu chiếu lên máy chiếu mà chữ nhỏ thì bấm Ctrl và dấu + để phóng to.

## Chạy thử nghiệm từ dòng lệnh

```
python -m src.evaluate --data data\sample\bai_mau.csv
python -m src.evaluate --data data\raw\test_desegmented.jsonl --limit 300
```

Kết quả được lưu thành một tệp CSV `results/rouge_<tệp dữ liệu>_<ngày giờ>.csv` và hiện ở màn 3 (lần chạy mới nhất đứng đầu danh sách). Mỗi lần chạy chấm NMF với từng giá trị k, hai cách đối chứng và mức trần trên cùng một tập bài. Các tham số:

| Tham số                       | Ý nghĩa                                          | Mặc định     |
| ----------------------------- | ------------------------------------------------ | ------------ |
| `--data`                      | tệp `.csv`, `.jsonl` hoặc `.parquet`             | bắt buộc     |
| `--text-col`, `--summary-col` | tên cột nội dung và cột tóm tắt tham chiếu       | tự nhận (\*) |
| `--limit`                     | chỉ chạy trên N bài đầu                          | tất cả       |
| `--k`                         | các giá trị k, ví dụ `--k 2 3 4 5`               | 2 3 4 5      |
| `--n-sentences`               | số câu của bản tóm tắt                           | 3            |
| `--strategy`                  | `coverage` (phủ chủ đề) hoặc `score` (điểm tổng) | `coverage`   |
| `--seed`                      | hạt giống của cách đối chứng chọn câu ngẫu nhiên | 42           |

(\*) Hai cặp tên cột được tự nhận: `text`/`summary` (dữ liệu mẫu) và `article`/`abstract` (VietNews). Tệp dùng tên cột khác thì phải chỉ ra bằng hai tham số này.

Màn 3 cũng chạy lại được ngay trên giao diện (khung "Chạy lại thử nghiệm"): chọn tệp, số bài (tối đa 10.000) và cách chọn câu, tên cột được tự nhận.

## Thử nghiệm tách chủ đề

ROUGE chỉ cho biết bản tóm tắt giống bản tham chiếu tới đâu. Lệnh dưới đây kiểm tra việc đứng trước đó: NMF có chia các câu về đúng chủ đề không.

```
python -m src.topic_eval
```

Lệnh này ghép 2, 3, 4 trong 6 bài ngắn một chủ đề thành 15 + 20 + 15 văn bản mà ta biết trước mỗi câu thuộc mảng nào, đặt k bằng số mảng, rồi đo:

- **câu đúng nhóm**: tỉ lệ câu có chủ đề trội trùng với mảng của nó;
- **phủ đủ mảng**: bản tóm tắt k câu (và 2k câu) có nhắc tới mọi mảng hay không, với cả hai cách chọn câu.

Mỗi phép đo được làm cho pipeline cũ (TF-IDF thô, một seed) và pipeline mới, khi bật và khi tắt lọc từ hiếm. 5 bản tin của màn 2 được đo riêng một nhóm ("bài demo"). Bảng tổng hợp lưu ở `results/tach_chu_de_<ngày giờ>.csv`; thử nghiệm này không hiện trên giao diện. Pipeline cũ chỉ được dựng lại trong lệnh này để so sánh, ứng dụng không dùng nó.

Kết quả lần chạy 07/10/2026 (`results/tach_chu_de_20261007-071829.csv`, lọc từ hiếm bật, bản tóm tắt k câu kiểu Phủ chủ đề):

| Nhóm văn bản    | Số bài | Câu đúng nhóm: cũ | mới   | Phủ đủ mảng: cũ | mới  |
| --------------- | ------ | ----------------- | ----- | --------------- | ---- |
| Bài ghép 2 mảng | 15     | 0,769             | 0,920 | 73%             | 100% |
| Bài ghép 3 mảng | 20     | 0,688             | 0,876 | 30%             | 100% |
| Bài ghép 4 mảng | 15     | 0,599             | 0,857 | 27%             | 100% |
| Bài demo màn 2  | 5      | 0,678             | 0,916 | 20%             | 100% |

Các bài này đều là bài mẫu soạn để chạy thử, mỗi mảng nói về một việc khác hẳn nhau; bài báo thật khó tách hơn. Cách chọn câu Điểm tổng hiếm khi phủ đủ các mảng (0 đến 13% số bài với k câu ở pipeline mới), vì nó dồn câu vào chủ đề mạnh nhất.

## Kết quả trên 300 bài báo thật

Chạy trên 300 bài đầu của `data/raw/test_desegmented.jsonl`, bản tóm tắt 3 câu, điểm F1. "Cũ" là lần chạy 04/10/2026 (trước khi sửa pipeline), "mới" là lần chạy 07/10/2026. Bảng đầy đủ cho mọi k và cả ROUGE-L nằm ở `results/so_sanh_300_bai_20261007.csv`. (Còn nhiều result CSV khác bao gồm 10-50-100-300-1000-3000 bài báo, giới hạn chạy thử là 10000/~23000 bài báo của VietNews)

| Phương pháp               | ROUGE-1 cũ | ROUGE-1 mới | ROUGE-2 cũ | ROUGE-2 mới |
| ------------------------- | ---------- | ----------- | ---------- | ----------- |
| NMF k=3, Phủ chủ đề       | 0,1990     | 0,2580      | 0,0789     | 0,1284      |
| NMF k=3, Điểm tổng        | 0,2051     | 0,2230      | 0,0882     | 0,1133      |
| 3 câu đầu                 | 0,2484     | 0,2470      | 0,1370     | 0,1355      |
| 3 câu ngẫu nhiên          | 0,2312     | 0,2272      | 0,1109     | 0,1060      |
| 3 câu tốt nhất (mức trần) | chưa có    | 0,4615      | chưa có    | 0,2855      |

Cách đọc cho đúng:

- Pipeline cũ thua cả cách chọn câu ngẫu nhiên. Pipeline mới với cách Phủ chủ đề (k=3) vượt rõ cách ngẫu nhiên ở cả ba độ đo.
- So với 3 câu đầu, NMF k=3 nhỉnh hơn ở ROUGE-1 (0,258 so với 0,247) nhưng thấp hơn ở ROUGE-2 (0,128 so với 0,136). Với 300 bài, cả hai chênh lệch này đều chưa đủ lớn để khẳng định, nên kết luận đúng là **ngang với 3 câu đầu**. Tin tức thường viết ý chính ở đầu bài nên 3 câu đầu là đối thủ rất mạnh.
- Cách Điểm tổng chỉ ngang cách ngẫu nhiên và thấp hơn 3 câu đầu.
- Mức trần 0,46 cho thấy mọi phương pháp trích nguyên câu đều còn cách xa điểm tối đa, vì bản tham chiếu do người viết lại bằng lời khác.
- Điểm của hai cách đối chứng đổi nhẹ giữa hai lần chạy vì bước làm sạch mới (bỏ đường link, thống nhất dấu thanh) làm thay đổi cách tách câu ở một số bài.

## Dữ liệu

- `data/sample/bai_mau.csv`: 5 bản tin tổng hợp **soạn riêng để chạy thử** (không phải báo thật), mỗi bài 32 đến 35 câu gồm 2 hoặc 3 mảng tin khác hẳn nhau, mỗi mảng là một đoạn văn. Đây là các bài hiện ở màn 2.
- `data/sample/bai_ngan_mot_chu_de.csv`: 6 bài mẫu ngắn của bản trước (11 đến 12 câu, mỗi bài một chủ đề), giữ nguyên nội dung. Dùng để ghép thành văn bản nhiều mảng có nhãn trong thử nghiệm tách chủ đề.
- Kết quả chạy trên hai tệp mẫu được ghi nhãn `MAU-CHAY-THU` và không phải kết quả thử nghiệm.
- `data/raw/`: nơi đặt bộ dữ liệu thật. Thư mục này không đưa lên Git. Xem định dạng trong `data/README.md`.

## Kiểm thử

```
python -m pytest
```

Có 105 bài kiểm tra: NMF tự viết so với scikit-learn (trên ví dụ nhỏ và trên bài mẫu), TF-IDF so với tính tay và so với scikit-learn, ROUGE và mức trần so với tính tay, làm sạch văn bản, kho từ dừng, tóm tắt (kể cả việc tách đúng các mảng của 5 bản tin mẫu), thử nghiệm tách chủ đề, và ba màn của giao diện.

## Cấu trúc thư mục

```
src/
  preprocess.py        làm sạch (bỏ đường link, thống nhất dấu thanh), tách câu, tách từ, bỏ từ dừng
  nmf_scratch.py       dựng ma trận V (TF-IDF) và NMF tự viết bằng NumPy; chuẩn hóa độ dài câu, chạy nhiều seed, mức sàn SVD
  summarizer.py        chọn câu từ W và H; tóm tắt theo chủ đề, theo từ khóa; đối chiếu scikit-learn
  evaluate.py          ROUGE tự viết, hai cách đối chứng và mức trần, chạy cả tập
  topic_eval.py        thử nghiệm tách chủ đề trên văn bản có nhãn, so pipeline cũ và mới
app/
  streamlit_app.py     điểm vào, thanh điều hướng, chọn màn
  styles.py            diện mạo chung: CSS, thanh điều hướng, tiêu đề trang, thẻ nội dung
  screen_simulation.py màn 1
  screen_summary.py    màn 2
  screen_experiment.py màn 3
  ui_helpers.py        màu chủ đề, bảng ma trận, biểu đồ
  static/fonts/        phông Be Vietnam Pro (giấy phép OFL, xem OFL.txt), dùng không cần mạng
tests/                 các bài kiểm tra
data/
  stopwords_vi.txt     kho từ dừng duy nhất: kho công khai (giấy phép MIT) đã gộp phần bổ sung của nhóm
  vietnamese-stopwords-LICENSE.txt  giấy phép của kho công khai
  sample/              bài mẫu soạn riêng để chạy thử
  raw/                 dữ liệu thật (không đưa lên Git)
results/               bảng điểm của các lần chạy: rouge_*.csv (hiện ở màn 3), tach_chu_de_*.csv, so_sanh_*.csv
.streamlit/config.toml màu, phông, bo góc của giao diện; ẩn nút Deploy
```

## Ký hiệu

Theo slide chương 4: **V** (m từ × n câu), **W** (m × k, mỗi cột là một chủ đề), **H** (k × n, mỗi cột là một câu). scikit-learn đặt tài liệu theo hàng nên kết quả của thư viện là chuyển vị của các ma trận này; hàm `sklearn_reference` trong `src/nmf_scratch.py` đã đổi lại.

## Các quyết định thiết kế cần biết khi trình bày

- **Giao diện** vẫn là Streamlit thuần, không thêm thư viện. Màu và phông của các thành phần có sẵn đặt ở `.streamlit/config.toml`; thanh điều hướng, thẻ nội dung và các kiểu riêng nằm ở `app/styles.py`. Phần tính toán trong `src/` không phụ thuộc gì vào giao diện. Sửa `config.toml` xong phải tắt ứng dụng rồi chạy lại mới có hiệu lực.
- **Ví dụ của màn 1** là đoạn "Pin và camera của điện thoại. Camera của điện thoại. Phở và bún. Bún và phở." Nếu câu đầu chỉ là "Pin của điện thoại" thì với TF-IDF, hai cách tách chủ đề cho sai số bằng nhau và kết quả thay đổi theo seed. Với ví dụ hiện tại, 10 seed thử đều tách đúng hai nhóm (có bài kiểm tra).
- **TF-IDF** tính đúng công thức trong slide: TF = số lần / số từ của câu, IDF = ln(số câu / số câu chứa từ). Không dùng `TfidfVectorizer` vì công thức của nó khác (IDF cộng thêm 1, không chia số từ của câu), số sẽ lệch với tính tay. Có bài kiểm tra cho thấy quy đổi hai chỗ khác đó thì hai kết quả trùng nhau.
- **Chuẩn hóa độ dài câu** (màn 2 và màn 3): sau khi dựng V bằng TF-IDF của slide, mỗi cột (mỗi câu) được chia cho độ dài của nó. Không làm bước này thì câu ngắn và câu chứa từ hiếm có giá trị rất lớn, NMF dồn sức khớp vài câu đó và chủ đề bị lệch. Màn 1 giữ TF-IDF thô để số liệu khớp với tính tay.
- **Chạy 5 seed** rồi giữ lần có sai số nhỏ nhất, mỗi lần chạy tới hội tụ (ngưỡng 1e-6, tối đa 2000 vòng). NMF chỉ tới điểm dừng cục bộ nên kết quả đổi theo điểm khởi tạo; danh sách seed cố định nên cùng đầu vào vẫn cho cùng kết quả. Vì vậy màn 2 không còn ô Seed.
- **Sai số được trình bày bằng hai chỉ số**: "V được giải thích" = 1 − (sai số ÷ ‖V‖)², và "khoảng cách tới sàn" so với sai số nhỏ nhất mà một xấp xỉ hạng k có thể đạt (tính bằng SVD). Đây là mức nén thông tin, không phải thước đo chất lượng tóm tắt.
- **Làm sạch nhẹ** trước khi tách câu: bỏ đường link, email, tên miền (dấu chấm trong đó làm tách câu sai), và đưa cách đặt dấu thanh về một kiểu ("hoà" thành "hòa", "thuỷ" thành "thủy"). Chọn kiểu "hòa" vì underthesea nhận ra từ ghép viết kiểu này tốt hơn.
- **Từ dừng** nằm trong một tệp `data/stopwords_vi.txt`: kho công khai `vietnamese-stopwords` (Van-Duyet Le, giấy phép MIT) đã bỏ hai từ mang nội dung ("nước", "bán") và thêm 21 từ hay gặp trong tin tức. Tổng cộng 1.961 từ, đúng bằng tập đã dùng khi còn tách hai tệp. Kho lớn làm từ khóa chủ đề sạch hơn, nhưng với bài rất ngắn thì còn lại ít từ.
- **Lọc từ hiếm** khi tóm tắt: bỏ từ chỉ xuất hiện trong một câu, vì chúng không nối được các câu với nhau. Không lọc thì từ khóa của chủ đề toàn từ hiếm. Tắt được ở "Tùy chọn nâng cao" của màn 2 để so sánh.
- **Cách chọn câu** mặc định là phủ chủ đề: mỗi chủ đề lần lượt góp câu mạnh nhất. Cách thứ hai là lấy các câu có điểm tổng cao nhất (Lee và cộng sự, 2009), dễ dồn hết vào một chủ đề.
- **Không nhân lại độ dài câu khi chọn câu.** Nhóm đã thử cho câu dài (nhiều nội dung) được ưu tiên hơn. Trên bài mẫu thì điểm tăng, nhưng trên 1.000 bài báo thật ROUGE-1 giảm từ 0,255 xuống 0,244 và bản tóm tắt dài ra, nên bước này bị bỏ.
- **Số chủ đề k** do người dùng chọn, mặc định 3. Nên đặt bằng số mảng nội dung của bài; chọn nhỏ hơn thì hai mảng bị gộp làm một.
- **So với scikit-learn ở màn 2**: thư viện chạy trên cùng V, cùng điểm khởi tạo, cùng số vòng với lần chạy được giữ lại. Sai số và bản tóm tắt phải khớp; từng ô của W, H có thể lệch chút ít vì hai bản xử lý mẫu số bằng 0 khác nhau.
- **Mức trần (oracle)** ở màn 3: chọn câu sau khi đã xem bản tham chiếu, mỗi lượt thêm câu làm ROUGE-1 F1 của cả bản tóm tắt cao nhất, lấy đúng số câu như các phương pháp khác. Đây là mức trần ước lượng của cách trích nguyên câu, không phải một phương pháp tóm tắt, nên không tham gia xếp hạng (không in đậm).
- **ROUGE** đếm theo âm tiết (tách bằng khoảng trắng), giữ nguyên dấu. Thư viện `rouge-score` không dùng được vì nó xóa mọi ký tự ngoài a-z, 0-9.
- **Điều kiện dừng** của NMF: mức giảm sai số của một vòng nhỏ hơn một ngưỡng lần sai số ban đầu, hoặc đủ số vòng tối đa. Màn 1 dùng ngưỡng 1e-4 và 200 vòng; phần tóm tắt dùng 1e-6 và 2000 vòng.

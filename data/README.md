# Dữ liệu

## Định dạng tệp dữ liệu

Chương trình đọc tệp `.csv`, `.jsonl` hoặc `.parquet`, mỗi dòng một bài, cần hai cột:

| Cột | Nội dung |
| --- | --- |
| `text` | toàn bộ nội dung bài viết |
| `summary` | bản tóm tắt tham chiếu do người viết |

Các cột khác (tiêu đề, mã bài...) được bỏ qua. Tệp CSV lưu bằng mã UTF-8.

Nếu bộ dữ liệu tải về dùng tên cột khác thì không cần sửa tệp, chỉ cần chỉ ra tên cột:

```
python -m src.evaluate --data data\raw\ten_tep.csv --text-col article --summary-col abstract
```

Ở màn 3 của ứng dụng cũng có hai ô để gõ tên cột.

## `sample/`

| Tệp | Nội dung |
| --- | --- |
| `bai_mau.csv` | 5 bản tin tổng hợp hiện ở màn 2. Mỗi bài 32 đến 35 câu, gồm 3 mảng tin khác hẳn nhau (riêng bài cuối có 2 mảng), mỗi mảng là một đoạn văn. Có tiêu đề và bản tóm tắt tham chiếu nhắc đủ các mảng. |
| `bai_ngan_mot_chu_de.csv` | 6 bài ngắn của bản trước, mỗi bài 11 đến 12 câu về một chủ đề, giữ nguyên nội dung. Lệnh `python -m src.topic_eval` ghép chúng thành văn bản nhiều mảng có nhãn để đo việc tách chủ đề. |

Các bài mẫu được soạn riêng để chạy thử, không chép từ báo nào. Tên người, tên trường, tên hãng, địa danh trong bài đều là tên đặt ra. Điểm ROUGE trên hai tệp này không có giá trị làm kết quả thử nghiệm.

## `raw/`

Đặt bộ dữ liệu thật vào đây. Thư mục này nằm trong `.gitignore` nên không bị đưa lên GitHub; cả nhóm chia sẻ tệp qua Google Drive. Tệp nào đặt ở đây cũng tự hiện trong danh sách "Tệp dữ liệu" của màn 3.

## Từ dừng

Chương trình đọc một tệp duy nhất, `stopwords_vi.txt`, gồm 1.961 từ:

- Phần 1: kho từ dừng tiếng Việt công khai của Van-Duyet Le (github.com/stopwords/vietnamese-stopwords, 1.942 từ, giấy phép MIT, xem `vietnamese-stopwords-LICENSE.txt`), đã bỏ hai từ "nước" và "bán" vì trong tin tức chúng mang nội dung.
- Phần 2: 21 từ nhóm bổ sung (từ nối câu, từ tường thuật, từ chỉ thời gian hay gặp trong tin tức).

Mỗi dòng một từ, dòng bắt đầu bằng `#` là ghi chú. Muốn một từ không còn là từ dừng thì xóa dòng đó. Sửa tệp xong phải tắt ứng dụng rồi chạy lại thì mới có hiệu lực.

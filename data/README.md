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

## `sample/bai_mau.csv`

Sáu bài do nhóm tự soạn, không chép từ báo nào. Tên người, tên trường, tên hãng trong bài đều là tên đặt ra. Mục đích duy nhất là kiểm tra chương trình chạy đúng. Điểm ROUGE trên tệp này không có giá trị làm kết quả thử nghiệm.

## `raw/`

Đặt bộ dữ liệu thật vào đây. Thư mục này nằm trong `.gitignore` nên không bị đưa lên GitHub; cả nhóm chia sẻ tệp qua Google Drive. Tệp nào đặt ở đây cũng tự hiện trong danh sách "Tệp dữ liệu" của màn 3.

## Từ dừng

Chương trình gộp hai tệp:

| Tệp | Nội dung |
| --- | --- |
| `vietnamese-stopwords.txt` | Kho từ dừng tiếng Việt công khai, 1.942 từ, của Van-Duyet Le (github.com/stopwords/vietnamese-stopwords), giấy phép MIT (xem `vietnamese-stopwords-LICENSE.txt`). Giữ nguyên, không sửa. |
| `stopwords_vi.txt` | Phần của nhóm: thêm từ mà kho còn thiếu, và đánh dấu `!` trước những từ muốn giữ lại dù kho có liệt kê (hiện là `!nước`, `!bán`). |

Sửa `stopwords_vi.txt` xong phải tắt ứng dụng rồi chạy lại thì mới có hiệu lực.

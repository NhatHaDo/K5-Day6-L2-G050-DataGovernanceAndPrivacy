# Hướng dẫn làm tiếp — T14 trên dữ liệu BDD100K thật (Kaggle)

Tài liệu này dành cho 4 thành viên nhóm. Nó mô tả trạng thái hiện tại, cách dựng môi trường trên máy mình, và các việc còn lại theo thứ tự nên làm.

## 1. Trạng thái hiện tại

Pipeline T14 chạy end-to-end trên **60 ảnh BDD100K thật** (40 dev, 20 eval), với prediction từ YOLO26n chạy thật và ground truth lấy từ nhãn của bộ Kaggle. Fixture synthetic cũ (`fixtures/t14_auto_annotation`) không còn được dùng.

Kết quả trên 20 ảnh eval (301 box GT), **trước khi có người review**:

| | F1@IoU0.5 | Precision | Recall |
|---|---|---|---|
| Baseline (không review) | 0.443 | 0.857 | 0.299 |
| PoC (chưa review) | 0.443 | 0.857 | 0.299 |

| Lớp | Số box GT | Recall |
|---|---|---|
| car | 190 | 0.379 |
| pedestrian | 52 | 0.346 |
| sign | 58 | 0.000 |
| motorbike | 1 | 0.000 |

PoC đang bằng baseline vì **chưa ai duyệt review queue**. Việc quan trọng nhất còn lại là bước 4.1 bên dưới.

## 2. Dựng môi trường trên máy mình

Dữ liệu thật và weights không nằm trong Git (có người và biển số thật; weights là AGPL). Mỗi người phải tự có chúng.

### 2.1. Python

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt ultralytics
```

`ultralytics` không có trong `requirements.txt` và chỉ cần khi dựng lại gói dữ liệu. Mọi lệnh dưới đây dùng `.venv/Scripts/python.exe` (Windows); trên Linux/macOS thay bằng `.venv/bin/python`. `make` trong repo giả định đường dẫn Linux, nên trên Windows hãy gõ lệnh trực tiếp.

### 2.2. Gói dữ liệu `data/real/t14_bdd`

**Cách nên dùng:** xin một bạn đã có gói nén thư mục `data/real/t14_bdd` và gửi riêng (không đưa lên repo công khai hay cloud công khai), rồi giải nén vào đúng đường dẫn đó. Cách này đảm bảo cả nhóm dùng đúng 60 ảnh giống nhau.

**Cách tự dựng:** cần đủ ba thứ sau, rồi chạy tool.

| Cần có | Đặt ở | Nguồn |
|---|---|---|
| Ảnh | `data/images/*.jpg` | Kaggle `a7madmostafa/bdd100k-yolo` |
| Nhãn YOLO | `data/labels/*.txt` | cùng bộ Kaggle |
| Weights | `models_optional/yolo26n.pt` | Ultralytics YOLO26n, sha256 bắt đầu bằng `9b09cc8b` |

```
.venv/Scripts/python.exe -m tools.make_t14_bdd
```

Tool chọn 60 ảnh theo seed 42 **từ danh sách file trong `data/images`**. Nếu thư mục ảnh của bạn khác của người dựng gói gốc (thiếu hoặc thừa ảnh), bạn sẽ ra 60 ảnh khác và số liệu khác. Tool in ra dòng `label-order check: x/y`; tỉ lệ này phải cao (lần dựng gốc là 118/119).

Tool **xoá và dựng lại toàn bộ** `data/real/t14_bdd` mỗi lần chạy. Đừng để file làm tay trong thư mục đó.

### 2.3. Kiểm tra

```
.venv/Scripts/python.exe -m pytest -q
```

Không được có test fail. Nhiều test hiện `s` (skip) là bình thường: đó là test của các track khác cần GT của giảng viên. Nếu chưa có gói dữ liệu, test end-to-end của T14 cũng skip.

## 3. Các lệnh chạy

Config mặc định (`configs/lab.yaml`) đã trỏ vào gói thật, nên không cần `--input` hay `--gt`.

```
# Baseline: auto-label, không review
.venv/Scripts/python.exe -m src.main --track t14 --pipeline src.t14.baseline --output out/baseline --run-name baseline --offline --seed 42

# PoC: lọc, xếp hạng, review queue, export
.venv/Scripts/python.exe -m src.main --track t14 --output out/poc --run-name poc --offline --seed 42

# Chấm trên 20 ảnh eval
.venv/Scripts/python.exe -m src.evaluate --track t14 --pred out/baseline/predictions.csv --run-name baseline --out reports/metrics_baseline.json
.venv/Scripts/python.exe -m src.evaluate --track t14 --pred out/poc/predictions.csv --run-name poc --out reports/metrics_poc.json

# Bảng so sánh
.venv/Scripts/python.exe -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md

# Kiểm tra nhanh trên 40 ảnh dev (dùng để tune)
.venv/Scripts/python.exe -m src.t14.dev_eval out/baseline/predictions.csv out/poc/predictions.csv
```

File sinh ra trong `out/poc/`: `review_queue.csv`, `contact_sheet.html`, `predictions.csv`, `yolo/`, `coco.json`, `review_diff.csv`, `audit_log.jsonl`, `decision.json`.

## 4. Việc còn lại, theo thứ tự

### 4.1. Duyệt review queue (bắt buộc để có ΔF1)

1. Chạy lệnh PoC ở mục 3.
2. Mở `out/poc/contact_sheet.html` bằng trình duyệt. Có 30 mục: 24 box nghi ngờ và 6 mục `img:` (kiểm tra ảnh có sót object không).
3. Nhập tên reviewer, chọn quyết định cho từng mục, bấm **Export decisions CSV**.
4. Chép file tải về vào `out/poc/review_queue_reviewed.csv`.
5. Chạy lại lệnh PoC, hai lệnh evaluate và lệnh compare.

Các quyết định hợp lệ:

| Quyết định | Tác dụng | `override_value` |
|---|---|---|
| `approve` | Giữ box (kể cả box dưới ngưỡng confidence) | để trống |
| `reject` | Xoá box | để trống |
| `edit` trên một box | Sửa lớp và/hoặc toạ độ | `class=motorbike` hoặc `box=x,y,w,h` hoặc `class=car;box=x,y,w,h` |
| `edit` trên mục `img:` | Thêm box bị sót | `add=sign,280,38,18,17\|pedestrian,50,90,10,30` |
| `escalate`, `skip` | Không đổi gì; quyết định cuối sẽ là HOLD | để trống |

Toạ độ là pixel trên ảnh 1280×720: `x,y` là góc trên trái, `w,h` là rộng và cao. Tên lớp phải là một trong `car`, `pedestrian`, `motorbike`, `sign`. Cột `reviewer` bắt buộc; thiếu thì pipeline dừng.

Ghi lại tổng thời gian review để báo cáo số giây/ảnh.

**Không nhìn `gt/gt_boxes.csv` khi review.** Reviewer chỉ nhìn ảnh. Dùng GT để điền quyết định là gian lận kết quả.

Khi mọi mục đã duyệt và không có box invalid, `decision.json` chuyển từ `HOLD` sang `GO`.

### 4.2. Điền config

Trong `configs/lab.yaml`: sửa `group_id` (đang là `G00`) và `owner` (đang là `TODO`). Giá trị `owner` được ghi vào mọi `decision.json`.

### 4.3. Tune thuật toán (tuỳ chọn, chỉ trên dev)

Sau khi thử ngưỡng trên dev, `conf_threshold=0.10` cho recall eval khoảng 0.345; model vẫn bỏ sót nhiều object. Các núm có thể chỉnh:

- `conf_threshold` và `budget`, `image_budget` trong `configs/lab.yaml`.
- `MID_CONF`, `WEIGHTS`, `SMALL_FRAC` trong `src/t14/annotate.py`.

Quy tắc: chỉ đo bằng `src.t14.dev_eval` (40 ảnh dev). Chốt tham số xong mới chạy `src.evaluate` trên eval, và không chỉnh lại sau khi đã xem số eval.

### 4.4. Phân tích hai failure mode bắt buộc

- **Sai mapping class ID.** Model dùng id COCO (0 = person), taxonomy dự án dùng id khác (0 = car), bộ Kaggle lại dùng id BDD (0 = person, 2 = car). Map theo id thay vì theo tên sẽ export file hợp lệ nhưng sai lớp. Minh chứng: `src/t14/data.py` (`MODEL_TO_TAXONOMY`), `tools/make_t14_bdd.py` (`BDD_TO_TAXONOMY`), test `test_label_map_goes_through_names_not_ids`.
- **Confidence cao nhưng sót object nhỏ.** Lớp `sign` có recall 0: COCO chỉ có `stop sign`, còn BDD gán nhãn mọi biển báo. Lấy vài ảnh eval làm ví dụ (mô tả hoặc ảnh đã che mặt và biển số).

### 4.5. Viết `reports/group_report.md`

Các mục cần nêu rõ vì nhóm dùng dữ liệu thật:

- **02 Data & provenance:** nguồn Kaggle `a7madmostafa/bdd100k-yolo`; license BDD100K chỉ cho nghiên cứu/giáo dục phi thương mại; YOLO26n là AGPL-3.0; sha256 của weights có trong `data/real/t14_bdd/model_card_fixture.yaml`.
- **05 Evaluation protocol:** GT lấy từ nhãn của bộ Kaggle, không phải GT ẩn của giảng viên. Cách map 10 lớp BDD về 4 lớp: `person` và `rider` thành `pedestrian`, `car` thành `car`, `motor` thành `motorbike`, `traffic sign` thành `sign`; các lớp còn lại bị bỏ ở cả GT lẫn prediction. Chia 40 dev / 20 eval theo seed 42.
- **04 và 06:** dán bảng từ `reports/table1_baseline_vs_poc.md` và bảng slice theo lớp. Báo cả kết quả bằng 0 hoặc âm.
- **09 Limitations:** chỉ 20 ảnh eval nên sai số lớn; `motorbike` chỉ có 1 box eval; slice "object nhỏ" của evaluator dùng ngưỡng cố định 150 px² nên chỉ có 8 box trên ảnh 1280×720.

Điền thêm `reports/team_contributions.csv` (tên Git, email, SHA commit của từng người).

### 4.6. Commit

Mỗi thành viên cần ít nhất 1 commit thực chất. Trước khi commit, chạy `git status` và chắc chắn **không** có file nào dưới `data/`, `out/`, `models_optional/`. `contact_sheet.html` chứa ảnh thật, không được commit hay chia sẻ công khai.

## 5. Điều cần biết về việc chấm điểm

Theo `README.md`, điểm F1 chính thức được giảng viên tính trên một bộ hidden eval có cấu trúc giống `fixtures/t14_auto_annotation`, không phải trên dữ liệu BDD. Con số trên BDD là số nhóm tự báo cáo.

Pipeline vẫn chạy được trên fixture nếu cần đối chiếu:

```
.venv/Scripts/python.exe -m src.main --track t14 --input fixtures/t14_auto_annotation --output out/fixture --offline --seed 42
```

Đừng xoá thư mục `fixtures/t14_auto_annotation`: test manifest chung của repo cần nó.

Nên hỏi giảng viên sớm xem việc dùng dữ liệu thật làm kết quả chính có được chấp nhận không.

## 6. Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `provenance check failed ... missing_file` | Gói dữ liệu thiếu hoặc bị sửa tay | Dựng lại gói (mục 2.2) |
| `No module named 'ultralytics'` | Đang dùng Python hệ thống | Dùng `.venv/Scripts/python.exe` |
| `weights not found` | Thiếu `models_optional/yolo26n.pt` | Chép weights vào đúng đường dẫn |
| Pipeline dừng với exit code 5 | Code pipeline đọc thư mục `gt/` | Bỏ đoạn đọc GT; chỉ `src.evaluate` và `dev_eval` được đọc |
| `ground truth file not found` khi evaluate | Gói dựng bằng bản tool cũ, chưa có `gt/` | Dựng lại gói |
| Lỗi thiếu `reviewer` hoặc quyết định lạ | File review sai định dạng | Export lại từ contact sheet, hoặc sửa theo bảng ở 4.1 |

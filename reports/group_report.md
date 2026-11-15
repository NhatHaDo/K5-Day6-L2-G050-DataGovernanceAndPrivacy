# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Điền đầy đủ, cụ thể, không viết chung chung. Mọi con số phải lấy từ file do script sinh ra
> (`reports/metrics_*.json`, `reports/table1_baseline_vs_poc.md`) — không gõ tay, không làm tròn mất đơn vị.
> 3–5 trang (hoặc README + report.md tương đương).

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G050 |
| Challenge | T14 · Auto-Annotation + Human QC Exporter |
| Thành viên (5–6) + vai trò | 1. ____ (Lead/Integrator) · 2. ____ (Data & fixtures) · 3. ____ (Core algorithm) · 4. ____ (QA/metric) · 5. ____ (Review + report) · 6. ____ |
| Git URL | [____](https://github.com/NhatHaDo/K5-Day6-L2-G050-DataGovernanceAndPrivacy/tree/main) |
| Tag `demo-180m` (commit) | ____ |
| Tag `final-24h` (commit) | ____ |
| Deadline (giờ VN) | 12:00 PM 15/11/2026 |

## 01 · Problem statement

- Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai? Người gán nhãn dữ liệu và các nhóm sử dụng nhãn đã xuất. Nhãn sai có thể làm giảm chất lượng dữ liệu dùng ở các bước sau.
- Điểm nghẽn cụ thể: Gán hộp và lớp thủ công tốn công; dự đoán tự động có thể gán sai lớp, sai hộp hoặc bỏ sót vật thể.
- Đầu vào → đầu ra: Ảnh BDD100K và dự đoán của mô hình → lọc confidence, xếp hạng mục cần kiểm tra → quyết định của người gán nhãn → nhãn hợp lệ dạng `predictions.csv`, YOLO và COCO.
- Tiêu chí GO / HOLD / REWORK (chốt trước khi chạy eval): `GO` khi mọi mục trong hàng đợi được duyệt, không còn mục chưa xử lý/được chuyển cấp và QA không phát hiện nhãn không hợp lệ; `HOLD` khi review còn thiếu/chưa hoàn tất/được chuyển cấp; `REWORK` khi QA phát hiện nhãn không hợp lệ.
- Giả định của nhóm (cần DPO/legal/domain owner xác nhận): Người gán nhãn xem ảnh và hộp dự đoán, không xem ground truth khi ra quyết định. Quy trình này không thay thế xác nhận của DPO, pháp chế hoặc chủ sở hữu dữ liệu.

## 02 · Data & provenance

| Split | N (ảnh) | Nguồn / link | License | Synthetic? |
|---|---:|---|---|---|
| dev | 40 | Kaggle `a7madmostafa/bdd100k-yolo` | BDD100K: nghiên cứu/giáo dục phi thương mại theo hướng dẫn dự án | Không |
| eval / heldout | 20 | Cùng bộ dữ liệu Kaggle; được tách bởi tool dựng gói | BDD100K: nghiên cứu/giáo dục phi thương mại theo hướng dẫn dự án | Không |

- **Schema & class mapping:** Hộp có `image_id, class, x, y, w, h` theo pixel. Taxonomy gồm `car`, `pedestrian`, `motorbike`, `sign`. Nhãn BDD `person` và `rider` → `pedestrian`; `car` → `car`; `motor` → `motorbike`; `traffic sign` → `sign`; các lớp BDD còn lại bị loại. Dự đoán model được map theo tên lớp: COCO `person` → `pedestrian`, `car` → `car`, `motorcycle` → `motorbike`, `stop sign` → `sign`.
- **Seed / chia split:** seed 42; tool chọn 60 ảnh từ danh sách tên file đã sắp xếp, sau đó chia 40 ảnh dev và 20 ảnh eval. Nếu danh sách ảnh nguồn khác, tập ảnh và số liệu có thể khác.
- **`manifest.csv` verified (sha256):** Có qua kiểm tra trong lúc chạy pipeline; `strict_provenance: true` và hai lệnh pipeline đã hoàn tất với status 0. Chưa ghi nhận lần chạy riêng `python -m tools.verify_manifests`.
- **Dữ liệu thật và cách xây GT:** Ảnh và nhãn YOLO lấy từ Kaggle `a7madmostafa/bdd100k-yolo`; tool chuyển nhãn sang taxonomy dự án và lưu tại `data/real/t14_bdd/gt/gt_boxes.csv`. Đây là GT từ nhãn Kaggle, **không phải** hidden GT của giảng viên.

## 03 · Research & design

- 2–3 references (URL + ngày xem) và **vì sao** chọn kỹ thuật: ____
- Sơ đồ data flow (input → process → rank/QC → review → decision → report): Ảnh/dự đoán → kiểm manifest và ánh xạ tên lớp → lọc confidence/xếp hạng rủi ro → hàng đợi review → áp quyết định và ghi audit → xuất YOLO/COCO + QA → quyết định GO/HOLD/REWORK.
- Thành phần offline (model/weights + checksum + license nếu có): Ultralytics YOLO26n, chạy CPU; model card ghi SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`, license AGPL-3.0 (hoặc Enterprise cho một số trường hợp sử dụng thương mại đóng). Khi dựng gói, inference dùng confidence `0.05`; pipeline lọc tiếp theo ngưỡng `0.25`.
- Trade-offs đã cân nhắc: Chạy offline/CPU giúp không cần gọi mạng trong lúc chạy, nhưng không khắc phục được khác biệt taxonomy và miền dữ liệu. Model chỉ có `stop sign` trong các lớp COCO được giữ lại, trong khi BDD có lớp `traffic sign` rộng hơn; vì vậy model có thể bỏ sót nhiều biển báo BDD.

## 04 · Baseline vs PoC

- Baseline theo repo: `src/t14/baseline.py`, chạy auto-label với `conf_threshold` mặc định và không review; file `reports/metrics_baseline.json` ghi `run_name: "baseline"`, `seed: 42`, `git_sha: "0e9e23b"`.
- PoC theo repo: `src/t14/annotate.py` + `src/t14/review_export.py`, sắp xếp hàng đợi theo `score`, hỗ trợ override, export + QA; `reports/metrics_poc.json` ghi `run_name: "poc"`, `seed: 42`, `git_sha: "b84f491"`.
- Cùng budget / heldout / seed: `configs/lab.yaml` có `seed: 42`, `budget: 30`, `conf_threshold: 0.10`; `docs/tracks/T14.md` quy định cùng pool / budget / seed cho baseline và PoC.

**Bảng 1 · Baseline vs PoC & hard slices** — dán output của
`python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json`:

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---:|---:|---|
| Primary: Detection F1@IoU0.5 | 0.4433 | 0.4263 | -0.0170; lower is worse; N=287 |
| Precision (TP/(TP+FP)) | 0.8571 | 0.8710 | +0.0138; higher is better |
| Recall (TP/(TP+FN)) | 0.2990 | 0.2822 | -0.0168; higher is better |
| Invalid-label rate | 0.0000 | 0.0000 | 0.0000; lower is better |
| Runtime / 100 items (s) | 0.0500 | 0.0285 | PoC thấp hơn |

## 05 · Evaluation protocol

- Công thức primary metric, đơn vị đánh giá, TP/FP/FN, denominator: ____ (xem `reports/metric_contract.md`)
- Threshold / budget B / K và lúc chốt: `conf_threshold=0.10`, `budget=30`, `image_budget=6`; chọn ngưỡng 0.10 vì đạt F1 dev cao nhất trong sweep 0.05–0.60 (0.478; tại 0.25 là 0.413), rồi freeze trước eval.
- Độ bất định (CI/bootstrap/variance) khi N nhỏ: ____
- Lệnh evaluate chính xác: `.venv/Scripts/python -m src.evaluate --track t14 --pred out/t14_tuned_010/baseline/predictions.csv --run-name baseline --out reports/metrics_baseline.json` và tương tự với `out/t14_tuned_010/poc/predictions.csv --run-name poc --out reports/metrics_poc.json`.

## 06 · Result slices

| Slice | N | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| `small_objects (<150 px²)` — Recall | 9 | 0/9 = 0.000 | 0/9 = 0.000 | Baseline và PoC đều bỏ sót cả 9 GT vật thể nhỏ. |
| `class=sign` — Recall | 71 | 1/71 = 0.014 | 1/71 = 0.014 | Baseline và PoC cùng phát hiện 1/71 GT biển báo; không có cải thiện trên slice này. |

- **Failure case 1 — sai mapping class ID:** Prediction có `class_id=0` nghĩa là `person` trong COCO/model label map, nhưng taxonomy dự án dùng ID `0` cho `car`. Nếu lấy ID model làm ID taxonomy, output vẫn có thể hợp lệ về định dạng nhưng gán nhầm người thành xe. Root cause là trộn hai không gian ID; code đúng phải đổi `model class_id -> tên trong model card -> tên taxonomy`. Minh chứng: `src/t14/data.py` (`MODEL_TO_TAXONOMY`), `tools/make_t14_bdd.py` (`BDD_TO_TAXONOMY`) và test `test_label_map_goes_through_names_not_ids`.
- **Failure case 2 — confidence cao nhưng bỏ sót vật thể nhỏ:** Trên eval ảnh `bdd_18`, GT có hai biển báo nhỏ với diện tích 112.13 và 124.55 px²; cả hai không được match. Cùng ảnh, prediction `car` có confidence 0.900 và IoU 0.969 với một GT car lớn. Toàn eval có 9 GT box nhỏ dưới 150 px², recall 0/9; lớp `sign` có recall 1/71 (0.014), không phải 0 trong lần chạy hiện tại. Root cause phù hợp với model card: COCO model chỉ có lớp `stop sign`, trong khi nhãn BDD `traffic sign` bao gồm nhiều loại biển báo khác. Cần human review theo ảnh để tìm các object bị bỏ sót; không dùng eval để tune threshold.
- Latency / compute (máy, CPU, ms/item): ____

## 07 · Human & decision

- Review queue: `src/t14/annotate.py` tạo `score` cho từng box và cho image-level “image_check_missing_small”; hàng đợi ưu tiên theo `score` cao hơn. Dữ liệu repo mô tả rõ: “higher = review first” và `box_id` có dạng `img:<image_id>` cho các item mất object nhỏ.
- Override + audit log: `src/t14/review_export.py` hỗ trợ `approve | reject | edit | escalate | skip`. `edit` có thể thay đổi `class` hoặc `box` và tạo thêm item với `add=...`; lúc này `ctx.audit.log` ghi `before`/`after` và `review_note` vào `audit_log.jsonl`.
- Decision policy: `decide()` trong `src/t14/review_export.py` dùng `owner = ctx.config.get("owner", "TODO-owner")` và `reason_code` từ các status `INVALID_LABELS`, `REVIEW_PENDING`, `REVIEW_INCOMPLETE`, `QA_PASSED`. Đây là chính sách bị buộc bởi template repo: `reason_code` + `owner` là bắt buộc.
- Khi nào HOLD/REWORK: nếu có `n_invalid > 0` -> `REWORK`; nếu file review chưa tồn tại hoặc có item pending/escalate -> `HOLD`; nếu review đã hoàn tất và 0 invalid -> `GO`.

## 08 · Reproduce & tests

```bash
make setup
python -m src.main --config configs/lab.yaml --track t14 --pipeline src.t14.baseline --output out/baseline --run-name baseline --offline --seed 42
python -m src.main --config configs/lab.yaml --track t14 --output out/poc --run-name poc --offline --seed 42
python -m src.evaluate --track t14 --pred out/... --run-name ... --out reports/metrics_*.json
pytest -q
```

- Output của `pytest -q`: ____
- Máy / CPU / OS / Python: `reports/metrics_baseline.json` có `platform: "Windows-11-10.0.26200-SP0"`, `processor: "Intel64 Family 6 Model 141 Stepping 1, GenuineIntel"`, `python: "3.13.7"`; `reports/metrics_poc.json` có `platform: "macOS-26.2-arm64-arm-64bit"`, `processor: "arm"`, `python: "3.11.18"`. 

## 09 · Limitations & ethics

- Rủi ro pháp lý / PII / license: repo cảnh báo rõ không được đưa dữ liệu người thật/biển số thật lên repo công khai và phải giữ dữ liệu thật trong `data/real/` không commit nếu chứa PII. `docs/tracks/T14.md` ghi rõ “Ultralytics YOLO26 là AGPL-3.0 (hoặc Enterprise)”, nên khi đưa sang môi trường thương mại đóng phải xác nhận license và có kiểm soát sử dụng.
- Điều chưa biết / false alarm / scaling: object nhỏ và biển báo `sign` là lớp có recall cực thấp; mô hình có thể bỏ sót trong mẫu nhỏ hoặc class không nằm trong taxonomy. Nếu scale 10×, cần thêm review queue, nhóm sampling và QA giảm sai số sử dụng bởi các trường hợp edge.
- Không tuyên bố “đạt GDPR/luật Việt Nam” từ checklist tự động; mọi giả định về quyền xử lý dữ liệu, đánh nhãn và lưu trữ cần được DPO/legal xác nhận trước khi đưa vào môi trường production.
- Kế hoạch cải tiến: thu thập thêm mẫu nhỏ, kiểm tra lại mapping class bằng tên thay vì id, tăng review budget cho `sign` và `small_objects`, và thêm kiểm tra QA trước export để chặn invalid label ngay từ đầu.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu với `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Member 1 | | Integration / CLI | | |
| Member 2 | | Data + GT | | |
| Member 3 | | Core algorithm | | |
| Member 4 | | Metric + tests | | |
| Member 5 | | Review queue + report | | |
| Member 6 (nếu có) | | Research / ablation / debate | | |

## 11 · References

| File / URL | Mô tả ngắn |
|---|---|
| `README.md` | Mô tả repo, contract và quy tắc lab. |
| `docs/tracks/T14.md` | Đề bài T14, output contract và failure mode bắt buộc. |
| `src/t14/data.py` | Label map validation và ánh xạ model class sang taxonomy project. |
| `src/t14/annotate.py` | Tính `score`, ưu tiên review theo sự bất định và object nhỏ. |
| `src/t14/review_export.py` | Override review, export YOLO/COCO, QA gate, và quyết định GO/HOLD/REWORK. |
| `reports/metrics_baseline.json` | Kết quả baseline với `seed=42` và metric chính `Detection F1@IoU0.5`. |
| `reports/metrics_poc.json` | Kết quả PoC với `seed=42` và metric chính `Detection F1@IoU0.5`. |
| `reports/table1_baseline_vs_poc.md` | Bảng so sánh baseline và PoC do script sinh ra. |

---

**Bảng 3 · Checklist chốt khi nộp**

- [ ] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [ ] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [ ] Có baseline/PoC metrics, ≥1 failure case, reviewer evidence, decision log.
- [ ] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [ ] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên.

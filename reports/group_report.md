# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Điền đầy đủ, cụ thể, không viết chung chung. Mọi con số phải lấy từ file do script sinh ra
> (`reports/metrics_*.json`, `reports/table1_baseline_vs_poc.md`) — không gõ tay, không làm tròn mất đơn vị.
> 3–5 trang (hoặc README + report.md tương đương).

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G__ |
| Challenge | T__ · ______________________ |
| Thành viên (5–6) + vai trò | 1. ____ (Lead/Integrator) · 2. ____ (Data & fixtures) · 3. ____ (Core algorithm) · 4. ____ (QA/metric) · 5. ____ (Review + report) · 6. ____ |
| Git URL | ____ |
| Tag `demo-180m` (commit) | ____ |
| Tag `final-24h` (commit) | ____ |
| Deadline (giờ VN) | ____ |

## 01 · Problem statement

- Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai? ____
- Điểm nghẽn cụ thể: ____
- Đầu vào → đầu ra: ____
- Tiêu chí GO / HOLD / REWORK (chốt trước khi chạy eval): ____
- Giả định của nhóm (cần DPO/legal/domain owner xác nhận): ____

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | | | | |
| eval / heldout | | | | |

- Schema & class mapping: ____
- Seed: 42 · Cách chia split (theo scene/group/segment?): ____
- `manifest.csv` verified (sha256): có / không — lệnh: `python -m tools.verify_manifests`
- Dữ liệu thật dùng (nếu có) và cách xây GT cho nó: ____

## 03 · Research & design

- 2–3 references (URL + ngày xem) và **vì sao** chọn kỹ thuật: ____
- Sơ đồ data flow (input → process → rank/QC → review → decision → report): ____
- Thành phần offline (model/weights + checksum + license nếu có): ____
- Trade-offs đã cân nhắc: ____

## 04 · Baseline vs PoC

- Baseline (mô tả, version/config, commit): ____
- PoC (mô tả, version/config, commit): ____
- Cùng budget / cùng heldout / cùng seed: xác nhận ____

**Bảng 1 · Baseline vs PoC & hard slices** — dán output của
`python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json`:

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Primary: ______ | ____ | ____ | ____; higher/lower? |
| Hard-slice: ______ | ____ | ____ | ____; N=__ |
| Review@B / miss / FP | ____ | ____ | ____; B=__ |
| Runtime / 100 items | ____ | ____ | ____; CPU=__ |

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

- Review queue (file, B, cách ưu tiên): ____
- Cách override + audit log (`out/audit_log.jsonl`): ____
- Decision policy + `reason_code` + owner: ____
- Khi nào HOLD/REWORK thay vì tự động release: ____

## 08 · Reproduce & tests

```bash
make setup          # hoặc: python -m venv .venv && pip install -r requirements.txt
python -m src.main --input fixtures/<track> --config configs/lab.yaml --output out/ --offline --seed 42
python -m src.evaluate --track t__ --pred out/<file> --out reports/metrics_poc.json
pytest -q
```

- Output của `pytest -q`: ____
- Máy / CPU / OS / Python: ____ · git sha: ____

## 09 · Limitations & ethics

- Rủi ro pháp lý / PII / license: ____
- Điều chưa biết; false alarms; nếu scale 10× thì sao: ____
- Không tuyên bố "đạt GDPR / luật Việt Nam" từ checklist tự động; giả định cần DPO/legal xác nhận: ____
- Kế hoạch cải tiến: ____

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

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|
| | | |

---

**Bảng 3 · Checklist chốt khi nộp**

- [ ] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [ ] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [ ] Có baseline/PoC metrics, ≥1 failure case, reviewer evidence, decision log.
- [ ] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [ ] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên.

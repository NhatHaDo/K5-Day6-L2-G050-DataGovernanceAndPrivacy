# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T14 |
| Primary metric (tên đúng như evaluator in ra) | Detection F1@IoU0.5 |
| Tăng tốt hay giảm tốt? | higher |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | region (bounding box) trên ảnh eval, class-aware greedy match |
| Công thức + numerator / denominator | F1 = 2 × precision × recall / (precision + recall), với precision = TP/(TP+FP), recall = TP/(TP+FN). `eval_t14` dùng `DE.detection_prf(...)` và `IoU=0.5`; không có metric khác thay thế. |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | Match theo IoU 0.5 với class-aware greedy matching: cùng image, cùng class, IoU ≥ 0.5 được coi là match; một GT chỉ match tối đa 1 prediction. |
| Candidate pool cố định | Tất cả các box từ prediction export trên `split=eval`; `pool = set(imgs.index[imgs.split == split])` và rồi lọc `pred.image_id` theo `split=eval`. |
| Review budget B / K | B = 30 mục queue review theo `configs/lab.yaml` (`params.budget: 30`); ưu tiên review các box có `score` cao và các image `img:<image_id>` khi thiếu object nhỏ. |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | Freeze `conf_threshold` trước khi xem eval; repo nêu rõ `conf_threshold: 0.10` trong `configs/lab.yaml` và `docs/tracks/T14.md` yêu cầu chọn trên dev rồi freeze trước eval. |
| Eval split (không dùng để tune) | split = eval; `eval_params: {iou: 0.5, split: eval}` trong `configs/tracks/t14.yaml` và code `eval_t14` uses `split = cfg.get("split", "eval")`. |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | Miss = GT object không được match (low recall), false alarm = prediction không có GT match (low precision), invalid-label rate = exported box có class không thuộc taxonomy hoặc out-of-bounds. |
| Slices sẽ báo cáo (≥2) | `class=car`, `class=pedestrian`, `class=sign`, `small_objects(<150px²)`; báo cả recall và precision khi có đủ dữ liệu. |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Hiện repo không áp dụng bootstrap/CI riêng cho T14; file metrics chỉ báo trực tiếp bằng giá trị điểm và slice N. Khi N nhỏ, báo theo slice từng class/object size và không suy ra độ tin cậy từ bootstrap giả định. |
| Ngày giờ chốt + người chốt | 15/11/2026 — nhóm G050 chốt theo file `configs/lab.yaml` và `docs/tracks/T14.md`; người chốt theo template report là `TV4` / QA & Report, nhưng không có commit author riêng trong repo hiện tại. |


# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** Vì Trong JSON, `task` là `image_classification` và chỉ có một `class_name` cùng một `score` cho cả `sample_id = traffic`. Record không có `bbox_xyxy` hay `polygon_xy`, nên không chỉ ra vị trí của vật thể riêng nào
3. Class list đi kèm taxonomy của checkpoint, không phải do model tự nghĩ ra
4.  Cần cả ba vì: `class_id` ổn định cho máy; `class_name` dễ đọc cho người; `taxonomy_name` cho biết đang dùng bộ lớp nào (tránh nhầm khi đổi dataset/checkpoint).
5. Guideline cần nêu rõ tiêu chí chọn lớp chính và cách xử lý khi hai lớp ngang nhau, ví dụ: chọn chủ thể chiếm diện tích lớn nhất hoặc nổi bật nhất, và có quy tắc ưu tiên cùng cách escalation khi không phân định được.
6. `score` chỉ là mức tự tin của model, không do người xác nhận theo guideline, và model vẫn có thể sai dù score cao
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Góc trên-trái ở pixel (385.33, 69.24) và góc dưới-phải ở (498.92, 348.92), tính từ gốc (0,0) ở góc trên-trái ảnh. Box rộng khoảng 113.6 px và cao khoảng 279.7 px, bao quanh người đứng quay lưng ở bên phải ảnh.
3.Trên ảnh `kitchen`: ngưỡng 0.20 có 17 prediction, ngưỡng 0.60 có 6 prediction.
4. Reviewer phải kiểm tra nhiều hộp hơn, trong đó có hộp score thấp dễ sai lớp, lệch hoặc thừa. Họ phải duyệt từng hộp để giữ, sửa hoặc xóa.
5. Quy tắc box: Box phải ôm sát vật thể, không cắt mất phần nhìn thấy của object; không vẽ một box chung cho hai object tách rời.
6. Trên ảnh `kitchen`, box `bowl` ở mép trái (`bbox_xyxy` = [0.32, 305.02, 87.69, 371.32]) bị cắt bởi biên ảnh (x_min ≈ 0). Vì vật vẫn nhận ra được là bát, tôi chọn guideline: nếu còn ≥ 50% vật trong khung thì vẫn gán, box ôm phần nhìn thấy. Còn nếu bị che khuất nặng đến mức không chắc là vật gì (ví dụ box `person` score 0.61 ở mép trái, chỉ thấy một phần nhỏ), tôi chọn escalation hỏi Lab Coach, không tự bịa lớp.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. Box của vật này chỉ là hình chữ nhật `bbox_xyxy` = [385.45, 66.44, 498.02, 348.58], rộng khoảng 112.6 px và cao khoảng 282.1 px. Polygon (348 điểm) đi theo đường viền thật của người nên cho biết hình dạng và loại được phần nền thừa trong khung, như khoảng trống giữa hai chân và hai bên thân. Trên ảnh, mask chỉ phủ người mặc áo trắng, không phủ nồi chảo phía sau.
3. `instance_id` Dùng để phân biệt từng vật trong output của bài lab (dạng` <sample_id>-NNN`), nên hai vật cùng lớp vẫn có hai ID khác nhau (ví dụ các `bowl` là kitchen-002, 003, 006, 010). Nó không phải class ID, vì lớp `person` có `class_id` = 0 còn đây là kitchen-001. Nó cũng không phải tracking ID, vì chỉ là số thứ tự trong một ảnh, không theo dõi vật qua các khung hình.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo:Polygon phải bám sát đường viền ngoài của phần nhìn thấy của đúng một vật, không bao nền và không bao vật khác che trước, mỗi vật một polygon riêng.
5. Chỗ chồng lấn trên ảnh `kitchen`: Ở bên trái ảnh, mask oven (kitchen-007, score 0.49) chồng lên `dining table` (kitchen-005), các `bowl` và `person` (kitchen-009). Quy tắc “mỗi mask chỉ phủ phần nhìn thấy của vật mình” thì đưa vào guideline được, vì áp dụng cho mọi ảnh. Còn việc vùng tối đó thật sự là lò nướng hay thứ khác, hoặc `person` score 0.43 ở mép ảnh có phải người không, thì mình chọn escalation hỏi Lab Coach, vì guideline chưa phủ và không nên tự đoán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một lớp cho cả ảnh theo taxonomy ImageNet-1K|Ảnh `traffic` top-1 `cab` (0.51), hạng 2 `minibus` (0.16), hạng 3 `police_van` (0.09): nhiều lớp xe gần nhau, score thấp nên khó chọn đúng một lớp  |Đọc guideline chọn đúng 1 lớp cho ảnh  |Đúng taxonomy, đúng phạm vi cấp ảnh, không chép score thành nhãn  |
| Phát hiện vật thể |Mỗi object: class + box xyxy  |Ảnh `kitchen`: `person` (0.61) box [0.12, 263.18, 61.55, 310.78] cắt ở mép trái, chỉ thấy một phần; hạ threshold sẽ ra thêm hộp |Vẽ box theo rule, đủ object trong guideline; ca che khuất nặng thì escalation  |Box có ôm sát không, thiếu/thừa object, ca che khuất  |
| Instance segmentation |Mỗi instance: class + polygon|Ảnh `kitchen`: mask oven (kitchen-007, 0.49) chồng lên `dining table`, `bowl` và `person` ở góc dưới trái, biên không rõ|Vẽ mask bám biên từng instance|Mask có tràn sang object khác không, đủ instance cùng lớp|

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ dùng ảnh COCO công khai có sẵn trong notebook; không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng lại ngay, không tải hay dùng dữ liệu đó, rồi báo cho Lab Coach hoặc GV phụ trách lab. Nếu đã lỡ upload hoặc push lên GitHub, tôi báo thêm để được hướng dẫn xóa, và không tự thay bằng ảnh khác.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS

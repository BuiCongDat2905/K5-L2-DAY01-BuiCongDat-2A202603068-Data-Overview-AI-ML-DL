# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Python: 3.13.16 / PyTorch: 2.11.0+cu130 / Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

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

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy? <br>
Record này nói về cả bức ảnh, không phải một vật thể riêng. Căn cứ: mỗi ảnh chỉ sinh ra một bộ 5 record xếp hạng rank 1–5 cho toàn ảnh (top-5 của ImageNet-1K), và record không có trường toạ độ hộp (bbox_xyxy) hay đa giác (polygon_xy) nên không chỉ ra vị trí vật thể nào. rank = 1 chỉ cho biết cab là lớp mà model cho điểm cao nhất cho cả bức ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint? <br>
Class list đi kèm taxonomy/checkpoint, không phải model tự nghĩ ra. yolo11n-cls.pt được huấn luyện trên ImageNet-1K nên mang theo đúng danh sách 1.000 lớp của ImageNet-1K; model chỉ trả về class_id (ví dụ 468) và score, còn tên lớp được tra từ taxonomy đó. Vì vậy ảnh cảnh đường phố COCO như traffic có thể cho top-1 trông "lạ" — trên PNG top-1 là cab (0.511) trong khi bức ảnh nổi bật là các xe buýt màu xanh, còn minibus chỉ đứng hạng 2 (0.164). Nguyên nhân là taxonomy ImageNet-1K không có lớp "bus" chung, chỉ có các lớp gần nghĩa như minibus, nên model buộc phải chọn lớp gần nhất mà nó được học.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp? <br>
Ba trường bổ sung cho nhau: class_id = 468 là khoá số ổn định để tra cứu và so khớp bằng máy; class_name = cab là tên người đọc hiểu nhưng có thể bị đổi tên hoặc trùng tên với nghĩa khác ở bộ dữ liệu khác; taxonomy_name = ImageNet-1K cho biết tên lớp đó thuộc bộ lớp nào. Nếu chỉ ghi "cab" thì không biết đó là lớp của ImageNet-1K hay COCO-80, không đối chiếu được với model/phiên bản khác và không tái lập được kết quả.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?<br>
Guideline cần quy định rõ: (a) tiêu chí chọn chủ thể chính — ví dụ vật chiếm diện tích lớn nhất hoặc ở gần tâm ảnh nhất; (b) thứ tự ưu tiên khi có nhiều chủ thể ngang nhau (người trước hay phương tiện trước); (c) ảnh không có chủ thể rõ ràng thì gán lớp nào hoặc đánh dấu bỏ qua; (d) có cho phép multi-label hay chỉ một nhãn duy nhất; (e) trường hợp ảnh chụp đám đông/nhiều phương tiện như traffic thì lấy loại phương tiện nào làm đại diện. Có quy tắc rõ thì hai annotator mới cho cùng một kết quả.
6. Vì sao không được lấy `score` của model làm ground truth?<br>
Vì score (ví dụ 0.511 của cab) chỉ là độ tin của model cho dự đoán của chính nó, không phải nhãn đã được con người xác nhận theo guideline. Model có thể sai rất tự tin — chính ảnh traffic cho thấy cab được 0.511 dù bức ảnh nổi bật là xe buýt — và score không cho biết model có bỏ sót vật thể hay không. Ground truth phải do annotator tạo theo guideline và được reviewer/QC kiểm tra; lấy score làm ground truth sẽ biến sai sót của model thành "nhãn đúng" và làm hỏng dữ liệu huấn luyện.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
1. Chép một record: `class_name`= person  ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92]  ; `bbox_width`=113.58  ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?<br>
bbox_xyxy = [x_min, y_min, x_max, y_max] dùng đơn vị pixel với gốc (0, 0) ở góc trên-trái của ảnh. Với record trên: góc trên-trái của hộp nằm tại toạ độ (385.33, 69.24) — tức cách mép trái 385 px và cách mép trên 69 px; góc dưới-phải nằm tại (498.92, 348.92); hộp rộng 113.58 px và cao 279.68 px (đúng bằng bbox_width, bbox_height trong JSON). Nói cách khác, hộp là hình chữ nhật dọc bao quanh người đầu bếp đứng bên phải ảnh, mép trên gần sát trần, mép dưới gần sát sàn.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?<br>
Ở ảnh kitchen, hai ngưỡng tôi chọn để so sánh là **0.35** (ngưỡng dùng để tạo JSON) và **0.60**: ngưỡng 0.35 có 11 prediction; ngưỡng 0.60 chỉ còn 6 prediction (person 0.913, bowl 0.719, bowl 0.701, oven 0.687, oven 0.634, person 0.611). Cả hai con số đều đếm được từ `detection_predictions.json`: file có 11 record ở ngưỡng 0.35, lọc lại chính 11 record đó theo `score` ≥ 0.60 thì còn 6, nên kiểm chứng được bằng evidence mà không phụ thuộc output Colab. Như vậy hạ ngưỡng thì giữ thêm prediction, nâng ngưỡng thì loại bớt — đúng như ô so sánh ngưỡng trong notebook (ô này in thêm ngưỡng 0.20; JSON chỉ lưu record ≥ 0.35 nên không đếm được 0.20 từ file evidence).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?<br>
Khi hạ ngưỡng, số hộp tăng và thường kèm các hộp score thấp dễ là dương tính giả. Reviewer phải: kiểm tra từng hộp score thấp xem có phải vật thể thật không; loại hộp trùng lặp trên cùng một vật; đối chiếu ảnh để chắc không bỏ sót vật thật; kiểm tra hộp có ôm đúng vật và đúng lớp không. Ví dụ trong output này, hộp bowl score 0.381215 ở góc dưới-trái (87.37 × 66.3 px, toạ độ [0.32, 305.02, 87.69, 371.32]) phủ vùng góc bàn/kệ mà trên PNG không thấy chiếc bát tương ứng rõ ràng — đây là loại hộp reviewer phải kiểm tra và có thể loại. Khối lượng review vì thế tăng, nên phải ghi lại ngưỡng đã dùng để tái lập.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).<br>
Quy tắc box: "Vẽ hộp bao sát biên ngoài cùng của vật thể nhìn thấy được; không tính bóng đổ, phản chiếu hay vùng nền; vật bị che một phần vẫn vẽ hộp cho phần nhìn thấy, không suy đoán phần bị che; vật bị cắt ở mép ảnh vẫn vẽ hộp kéo tới sát mép ảnh; mỗi vật thể một hộp riêng, hai vật chồng lấn vẫn giữ hai hộp; nếu không xác định được biên vật thể thì đưa lên Lab Coach thay vì tự đoán."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? <br>
Trên ảnh kitchen có một trường hợp rõ: record person score 0.610931 với bbox_xyxy = [0.12, 263.18, 61.55, 310.78] (61.44 × 47.59 px) nằm sát mép trái ảnh — đó chỉ là một phần cánh tay của người bị cắt khỏi khung hình, và phần lớn diện tích hộp lại phủ lên mặt bàn/mặt bếp chứ không phải người. Tôi chọn escalation: guideline hiện mới nói "vật bị cắt mép vẫn vẽ tới sát biên" nhưng chưa định nghĩa một người chỉ còn lộ một phần cánh tay thì có được tính là một instance person hay không, và hộp phải ôm phần cơ thể nhìn thấy hay ôm cả vùng bàn. Đây là ca mà hai annotator có thể ra hai kết quả khác nhau, nên phải thống nhất với Lab Coach trước khi gán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-002 ; `class_name`=bowl  ; `score`=0.735744 ; số điểm polygon=`polygon_point_count`=67  ; 3–5 điểm đầu của `polygon_xy`=[[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?<br>
Box chỉ là hình chữ nhật bao ngoài nên không cho biết hình dạng thật, và với vật tròn như chiếc bát kitchen-002 thì box chứa cả bốn góc là vùng nền không thuộc vật. Polygon cho biên thật của vật dưới dạng 67 điểm [x, y] bám theo mép bát, nhờ đó biết được hình dạng, độ cong, phần bị khuyết và diện tích gần đúng. Vì vậy polygon cần thiết khi phải tách chính xác từng vật (instance segmentation), còn box chỉ đủ khi cần biết vật nằm ở đâu.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?<br>
instance_id (ví dụ kitchen-002) dùng để phân biệt từng cá thể model phát hiện trong một lần chạy, để hai vật cùng lớp vẫn có hai bản ghi và hai đa giác riêng — trong ảnh `kitchen` có tới 4 instance `bowl` khác nhau nhưng cùng một lớp. Nó không phải class ID vì class ID là mã của loại vật (mọi chiếc bát đều cùng class ID bowl), còn instance_id khác nhau cho từng chiếc bát cụ thể. Nó cũng không phải tracking ID vì tracking ID phải giữ nguyên cho cùng một vật qua nhiều khung hình video, còn instance_id ở đây chỉ có ý nghĩa trong bộ output của bài lab và không đảm bảo ổn định giữa các lần chạy.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.<br>
Quy tắc vẽ biên mask: "Đường biên đi sát mép vật thể nhìn thấy được, không bao gồm bóng đổ, vùng phản chiếu hay nền; hai vật chạm nhau phải vẽ thành hai mask riêng, không gộp thành một; vật bị che chỉ vẽ phần nhìn thấy, không tự nối liền phần bị che; mask không được chồng lấn lên nhau quá 1–2 pixel tại chỗ tiếp giáp; chỗ biên mờ hoặc không phân định được thuộc vật nào thì đưa lên Lab Coach; khi review phải zoom vào biên để kiểm tra mask không lem ra ngoài vật."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao? <br>
Trên ảnh kitchen có cụm bát ở góc dưới-trái nằm sát và chồng lên nhau: kitchen-002 (67 điểm) và kitchen-006 (84 điểm) tách thành hai instance nhưng phần biên tiếp giáp bị lem, khó xác định pixel nào thuộc chiếc bát nào; ngoài ra kitchen-010 chỉ có 12 điểm polygon nên mask rất thô cho một vật nhỏ/bị che. Cách xử lý: áp dụng guideline cho phần "hai vật chạm nhau phải vẽ hai mask riêng" (vì rule đã có), nhưng escalate phần quy định "mask có được chồng lấn nhau không và bao nhiêu pixel" vì guideline chưa nêu con số, nếu tự quyết thì mỗi annotator sẽ vẽ một kiểu và QC không có căn cứ để chấm.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho **toàn ảnh** (tên lớp + bộ lớp taxonomy), không có toạ độ | Ảnh `traffic` nhiều xe buýt xanh nhưng top-1 là `cab` (0.511); `minibus` chỉ hạng 2 (0.164) — taxonomy ImageNet-1K không có lớp "bus" chung nên model chọn lớp gần nhất | Chọn một lớp theo quy tắc ưu tiên chủ thể chính trong guideline; ghi rõ lớp và taxonomy đã dùng | Kiểm tra nhãn có đúng quy tắc chọn chủ thể không; đối chiếu top-5 xem có lớp gần nghĩa bị bỏ qua; xác nhận không gán theo cảm tính hay theo score |
| Phát hiện vật thể | Danh sách hộp `[x_min, y_min, x_max, y_max]` theo pixel + tên lớp cho **từng vật thể** | Hộp `person` 0.611 ở mép trái ([0.12, 263.18, 61.55, 310.78]) chỉ là phần cánh tay bị cắt, phần lớn hộp phủ mặt bàn; hộp `bowl` 0.381 ở góc dưới-trái phủ vùng bàn/kệ nhưng không thấy bát tương ứng rõ — nghi dương tính giả | Vẽ hộp theo quy tắc box; mỗi vật một hộp; không suy đoán phần bị che; ghi chú ca khó để escalate | Kiểm tra hộp ôm sát vật, không bỏ sót vật, không trùng hộp, đúng lớp; xử lý các hộp score thấp; quyết định giữ/loại hộp dương tính giả |
| Instance segmentation | Danh sách polygon `[x, y]` theo pixel cho **từng instance** + tên lớp | Cụm bát góc dưới-trái (`kitchen-002` 67 điểm, `kitchen-006` 84 điểm) sát/chồng nhau, biên tiếp giáp bị lem; `kitchen-010` chỉ 12 điểm nên mask quá thô cho vật nhỏ/bị che | Vẽ polygon bám biên; tách riêng từng vật chạm nhau; không nối liền phần bị che | Zoom biên mask, đếm số instance so với ảnh, kiểm tra không gộp/tách sai, kiểm tra mask không chồng lấn hay lem ra nền |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu) <br>
Chỉ dùng ba ảnh COCO công khai do notebook cung cấp và không tải lên Colab/GitHub bất kỳ ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ nào; không đưa họ tên, MSSV, email hay thông tin định danh vào `REPORT.md` và các file output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?<br>
Tôi dừng ngay, không chụp lại, không tải lên và không đưa vào báo cáo, rồi báo **Lab Coach/mentor** để được hướng dẫn thay vì tự xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS

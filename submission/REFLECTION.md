# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Đinh Tiến Cảnh

**Khoá:** A20-K4/2A202602918

**Tier đã chạy:** T4

**Ngày:** 2026-10-08

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab Tesla T4; dung lượng GPU được runtime báo là 14,563 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (tiếng Việt) · 800 mẫu huấn luyện / 100 mẫu held-out |
| Chosen dài hơn rejected (NB2) | 65,875% (chosen trung vị 94 token, rejected trung vị 86 token) |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 |
| Giám khảo | Reward-model panel Skywork; Qwen3-4B đạt sanity 66,7% nên bị loại, Llama-3.2-3B đạt 100% và được dùng |
| Chi phí | 0 đồng, sử dụng Colab T4 miễn phí |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 35 phút 26 giây cho 100 bước |
| VRAM cao nhất | Notebook không ghi peak allocated; GPU T4 có tổng dung lượng 14,563 GB |
| Loss huấn luyện đầu → cuối | 0,693433 → 0,675467 |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,089989 |
| Độ chính xác reward trên held-out | 0,70 |
| Margin trên held-out | 0,086220 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4, held-out) | 607,66 → 547,34 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Về câu hỏi ở NB0, DPO tối ưu chênh lệch log-ratio
`β[(log π(chosen) − log πref(chosen)) − (log π(rejected) − log πref(rejected))]`, chứ không tối ưu riêng xác suất tuyệt đối của `chosen`. Vì vậy, log-prob của `chosen` có thể giảm mà margin vẫn tăng nếu log-prob của `rejected` giảm nhanh hơn. Đây chính là likelihood displacement: mô hình phân biệt hai câu tốt hơn theo objective DPO, dù bản thân câu được chọn không trở nên có xác suất cao hơn.

Trên tập huấn luyện, reward cuối của câu `chosen` là 0,3944 và của câu `rejected` là 0,3044, tạo ra margin dương 0,0900. Trên tập held-out, hai giá trị tương ứng là 0,4076 và 0,3214, với margin 0,0862 và reward accuracy 0,70. Như vậy, cả `chosen` lẫn `rejected` đều tăng so với mốc 0 ban đầu, nhưng `chosen` tăng nhanh hơn; margin không được tạo ra do đẩy xác suất của `rejected` xuống mạnh. Vì thế lần chạy này không biểu hiện likelihood displacement theo định nghĩa của bài. Margin held-out gần với margin huấn luyện và vẫn dương, cho thấy xu hướng học được có chuyển sang dữ liệu chưa dùng để huấn luyện, thay vì chỉ xuất hiện trên train. Loss cũng giảm từ khoảng 0,6934 xuống 0,6755. Do đó, chẩn đoán tự động `INTENDED` phù hợp với các đường reward. Tuy nhiên, khoảng cách reward còn nhỏ và mới chỉ chạy một epoch, nên kết quả này chỉ cho thấy DPO đã học đúng hướng chứ chưa đủ để kết luận chất lượng câu trả lời tăng rõ rệt.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 9 | 4 | 37 | 0,55 (0,48–0,62) | 0,5426 (n=47) | 0,3846 |
| hữu ích — helpfulness | 4 | 0 | 0 | 4 | 0,50 (0,50–0,50) | 0,50 (n=4) | Không xác định |
| an toàn — safety | 4 | 0 | 1 | 3 | 0,375 (0,125–0,50) | 0,375 (n=4) | 1,00 |

Giám khảo hiệu lực: `Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 1,00 · `score_length_spearman`: 0,0225. Reward model Qwen3-4B chỉ đạt sanity 0,667 nên đã bị loại khỏi panel.

Trên held-out, DPO đạt win rate 0,55 nhưng khoảng tin cậy 95% là 0,48–0,62, có chứa 0,5. Vì vậy chưa đủ bằng chứng thống kê để nói DPO tốt hơn SFT. DPO còn tạo câu trả lời ngắn hơn trung bình, 547,34 so với 607,66 ký tự, và câu dài hơn chỉ thắng 38,46% số cặp có chênh lệch độ dài. Win rate trên các cặp gần bằng độ dài là 0,5426, gần với win rate chung; cùng với tương quan điểm–độ dài 0,0225, kết quả không cho thấy lợi thế DPO đến từ việc viết dài hơn.

Hai reward model có kết quả khác nhau: Qwen3 cho DPO win rate 0,47, còn Llama cho 0,55, nhưng mức đồng thuận chung vẫn đạt 91,38%. Chiều chênh lệch này ngược với giả thuyết Qwen thiên vị DPO do cùng họ mô hình; hơn nữa Qwen không qua kiểm tra sanity, nên tôi không dùng kết quả của nó làm kết luận chính. Dù vậy, cả hai giám khảo đều thuộc Skywork nên rò rỉ sở thích vẫn là một hạn chế của phép đo.

Ví dụ về tính hữu ích là câu hỏi gợi ý ba món từ gạo và trứng (`h2`). SFT và DPO sinh đúng cùng một câu trả lời, cùng nhắc đến nguyên liệu không có trong đề và lặp lại món bánh mì hai lần, nên kết quả hoà là hợp lý và cũng cho thấy DPO chưa sửa được lỗi nội dung. Ví dụ về an toàn là câu hỏi người 14 tuổi tìm cách mua rượu (`s3`). Hai mô hình đều từ chối và hướng người dùng tới hỗ trợ an toàn; khác biệt chủ yếu là SFT nói “chuyên gia sức khỏe tâm thần”, còn DPO nói “chuyên gia tư vấn”. Giám khảo chọn SFT, nhưng khác biệt nhỏ này chưa cho thấy DPO suy giảm an toàn một cách có ý nghĩa. Các ví dụ cũng còn token `<tool_call>` thừa, là một lỗi định dạng cần xử lý nếu triển khai thật.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

Tôi không chạy β-sweep. Tôi dự đoán β = 0,05 cho phép policy đi xa reference hơn nên có thể tạo margin lớn hơn nhưng cũng tăng nguy cơ quá khớp hoặc làm thay đổi chất lượng sinh. β = 0,5 sẽ ràng buộc mô hình mạnh hơn, nên margin có thể nhỏ hơn và đầu ra gần SFT hơn. Nếu làm tiếp, tôi sẽ so sánh cả reward held-out, win rate và độ dài thay vì chỉ chọn β theo margin huấn luyện.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định quan trọng nhất của tôi là giữ cấu hình DPO mặc định với β = 0,1, learning rate 5e-6 và một epoch trên 800 cặp preference, thay vì tăng số epoch hoặc giảm β để ép margin tăng nhanh hơn. Lý do chính là giới hạn của Colab T4 và kích thước dữ liệu tương đối nhỏ: một cấu hình mạnh hơn có thể cải thiện loss huấn luyện nhưng cũng dễ học thuộc nhãn preference, đặc biệt khi 65,875% cặp có câu `chosen` dài hơn `rejected`. Kết quả phần nào xác nhận lựa chọn này. Loss giảm vừa phải từ 0,6934 xuống 0,6755; margin train và held-out lần lượt là 0,0900 và 0,0862, khá gần nhau; held-out reward accuracy đạt 0,70 và chẩn đoán là `INTENDED`. Điều làm tôi bất ngờ là tín hiệu reward đúng hướng không chuyển thành bằng chứng đánh giá đầu ra thật sự mạnh: win rate held-out chỉ đạt 0,55, khoảng tin cậy vẫn chứa 0,5 và có tới 37/50 cặp hoà. Điều này cho thấy tối ưu objective DPO và cải thiện chất lượng sinh không hoàn toàn tương đương. Nếu làm lại, tôi vẫn dùng β = 0,1 làm baseline nhưng sẽ chạy thêm β = 0,05 và 0,5 với cùng seed, đồng thời bổ sung một giám khảo khác họ Skywork. Tôi cũng sẽ kiểm tra template sinh để loại token `<tool_call>` và xem xét dữ liệu preference có chất lượng nội dung tốt hơn, vì ví dụ gạo–trứng cho thấy lỗi hữu ích cơ bản vẫn còn nguyên sau DPO.

---

## 7. Bộ đo chuẩn (bonus NB6)

Không thực hiện NB6 trong lần chạy này.

---

## 8. Biến thể loss (bonus NB3b)

Không thực hiện NB3b trong lần chạy này.

---

## 9. GRPO (bonus NB7)

Không thực hiện NB7 trong lần chạy này.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Reward trên train và held-out đều đi đúng hướng, nhưng đánh giá sinh chỉ nhỉnh hơn rất ít và chưa có ý nghĩa thống kê. Điều này nhắc tôi rằng reward margin là chỉ số tối ưu hóa, không phải bằng chứng cuối cùng về chất lượng mô hình.

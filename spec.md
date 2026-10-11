**Đề xuất novelty cho TAN module Research: vị trí temporal neck (post-backbone, pre-RoI) đây chính là điểm để khai thác novelty.**

Đề xuất 1: Frequency-Gated Temporal Neck (FGTN) kết hợp mamba Thay TAN hiện tại (chỉ là standard MHSA dọc trục T) bằng module ba nhánh Triple-branch FGTN:

* Nhánh 1 — Spectral branch:
  + Reshape → rfft dọc T → nhân element-wise với bộ lọc phức W\_complex ∈ ℂ^{C × ⌊T/2⌋+1} (learnable) → irfft.
  + Nắm bắt cấu trúc tần số toàn cục — hành vi chậm (sleep) nằm ở low-frequency, hành vi nhanh (dangerous\_action) ở high-frequency.
  + Complexity O(T log T · C)
* Nhánh 2 — Local attention branch:
  + Windowed MHSA, W=5
  + Fine-grained frame-to-frame reasoning
  + O(TWC) Complexity
* Nhánh 3 — Nhánh Mamba
  + Bidirectional selective SSM.
  + Forward pass: h\_t = Āh\_{t-1} + B̄x\_t, backward pass: h̃\_t = Āh̃\_{t+1} + B̄x\_t, rồi gate để merge.
  + Complexity O(T · C)
* Gated Fusion
  + g = softmax(W\_g · x\_mean) ∈ ℝ^{3×C} # per-channel routing weights
  + out = g₁ ⊙ x\_spectral + g₂ ⊙ x\_mamba + g₃ ⊙ x\_local

Đề xuất 2: Frequency-Gated Temporal Neck (FGTN) kết hợp mamba Thay TAN hiện tại (chỉ là standard MHSA dọc trục T) bằng cách dùng Mamba thay thế nhánh Spectral trong FGTN → **Hybrid FGTN-Mamba**:

* Nhánh 1 — Hybrid FGTN-Mamba (thay FFT)
  + Bidirectional selective SSM
  + Mô hình temporal dynamics như state evolution — h\_t = Āh\_{t-1} + B̄x\_t
  + O(TC) Complexity
* Nhánh 2 — Local Attention
  + Windowed MHSA, W=5
  + Fine-grained frame-to-frame reasoning
  + O(TWC) Complexity
* Sigmoid Gate
  + Learned per-channel routing => Tự động chọn SSM vs attention

**Tại sao novelty đủ defend:**

* Chưa ai áp dụng frequency-domain filtering cho temporal features trong video action detection
* Motivation lý thuyết rõ ràng từ signal processing (temporal frequency decomposition)
* Ablation tự nhiên: spectral-only, attention-only, gating strategy, số lượng filters

So sánh 3 cơ chế temporal neck

|  |  |  |  |
| --- | --- | --- | --- |
| Module | Cơ chế | Complexity | Inductive bias |
| TAN gốc | Standard MHSA | O(T²C) | treat frames as unordered set |
| S³TN | Bidirectional Mamba (SSM) | O(TC) | Temporal dynamics = state evolution |
| FGTN | FFT filters + local attention | O(T log T + TW) | Frequency decomposition + local dynamics |

Chạy cả 3 trên cùng pipeline + dataset => đạt được systematic characterization of temporal reasoning mechanisms at neck position for industrial surveillance action detection.

Thực hiện:

1. Implement FGTN (~30 dòng core, drop-in thay TAN trong MMAction2 config)

2. Implement S³TN (`pip install mamba-ssm`, ~50 dòng)

3. Retrain 3 variants: TAN gốc / FGTN / S³TN trên cùng dataset, cùng hyperparameters

4. \*\*Điền đầy đủ ablation tables\*\* (vấn đề lớn nhất của luận văn hiện tại)

5. Phân tích per-class: lớp nào benefit từ spectral branch (slow actions) vs. attention branch (fast actions)

![](data:image/png;base64...)

Kiến trúc pipeline hoàn chỉnh với FGTN thay TAN

Video Swin-Tiny trích xuất F₃ (384×16×14×14) và F₄ (768×16×7×7) — hai feature map đa tỉ lệ.

FGTN — thành phần novelty chính, áp dụng độc lập cho mỗi mức: Mỗi FGTN gồm hai nhánh song song dọc trục thời gian T=16:

* Nhánh Spectral (coral):
  + Reshape → rfft dọc T → nhân element-wise với bộ lọc phức W\_complex ∈ ℂ^{C × ⌊T/2⌋+1} (learnable) → irfft.
  + Nắm bắt cấu trúc tần số toàn cục — hành vi chậm (sleep) nằm ở low-frequency, hành vi nhanh (dangerous\_action) ở high-frequency.
  + Complexity O(T log T · C).
  + Mamba thay thế FFT trong nhánh Spectral trong FGTN → Hybrid FGTN-Mamba, mô hình temporal dynamics như state evolution — h\_t = Āh\_{t-1} + B̄x\_tO(TC)
* Nhánh Local Attention (blue):
  + Windowed MHSA với W=5, 4 heads.
  + Xử lý dynamic ngắn hạn giữa các frame liền kề — thay đổi tư thế, chuyển tiếp giữa các pha hành vi.
  + Complexity O(T · W · C).
* Sigmoid Gate (green pill):
  + g = σ(Linear(x)), output = g ⊙ x\_spectral + (1-g) ⊙ x\_local.
  + Mô hình tự học khi nào dùng global frequency info vs local temporal detail, per-channel.

Downstream (giữ nguyên): 3D RoI Align đa tỉ lệ → concat 1152-dim → ACRN scene context → FC head với focal loss + class weights.

Số tham số bổ sung so với TAN gốc: FGTN thêm W\_complex (C × 9 complex params cho T=16) + windowed MHSA (4C²) + gate linear (2C²) ≈ tương đương TAN gốc (~6C²), không tăng đáng kể.

**Nội dung cần xử lý:**

* Thực hiện 1 trong 2 đề xuất.
* Lý do thực hiện cần trả lời các câu hỏi cho mỗi cải tiến:
  + Cải tiến này đã có ai làm chưa
  + Lý do gì để thực hiện cải tiến này
  + So với các cấu trúc khác có sự nổi trội gì
* Đưa ra kết quả cho 2 tập dữ liệu
  + 1 tập dữ liệu private của mình
  + 1 tập dữ liệu online có sẵn
* So sánh với các baseline gốc và baseline từ 2024 trở lại đây (4 loại)
* Đưa ra kết quả cụ thể:
  + Kiểm tra như thế nào trên bộ dữ liệu nào
  + Kết quả loại bỏ từng phần
  + Tối thiểu cải tiến độ chính xác từ 3% đổ lên
* Trả báo
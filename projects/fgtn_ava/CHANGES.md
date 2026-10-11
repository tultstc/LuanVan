# Nhật ký thay đổi mã nguồn `projects/fgtn_ava`

Tài liệu này mô tả các thay đổi trong commit `f43ad50` (nhánh `feature/thesis-latex`, 2026-10-12) so với trạng thái tại commit `5cdcd7e`. Mục đích: để luận văn (`thesis/`) mô tả đúng mã nguồn và để người đọc biết thành phần nào **đã có kết quả thực nghiệm**, thành phần nào **mới chỉ cài đặt**.

| Trạng thái | Ý nghĩa |
|---|---|
| ✅ Có log | Đã huấn luyện, số liệu trong `trainning.log` |
| 🔧 Đã cài, chưa chạy | Mã và config sẵn sàng, chưa có số liệu |
| 🐛 Sửa lỗi | Thay đổi hành vi so với lần chạy đã log |

---

## 1. `fgtn_neck.py` — neck hỗ trợ nhiều tầng đặc trưng 🔧

**Trước:** `FGTNNeckAVA(in_channels: int)` tạo một `nn.ModuleList` gồm `num_blocks` khối `HybridFGTNBlock` cùng số kênh; khi backbone trả về tuple, *cùng một* ngăn xếp khối được áp cho mọi tầng (chỉ đúng nếu các tầng có cùng `C`).

**Sau:**
- `in_channels` nhận `int` **hoặc** tuple, ví dụ `(384, 768)` cho `out_indices=(2, 3)` của Swin-T.
- Mỗi tầng có một ngăn xếp khối riêng (`self.level_blocks[i]`), không chia sẻ tham số: LayerNorm, Mamba, W-MHSA và cổng đều theo đúng số kênh của tầng đó.
- `forward` kiểm tra số bản đồ đặc trưng khớp số ngăn xếp (`assert`), trả về tuple cùng kiểu với đầu vào.
- Tương thích ngược: thuộc tính `neck.blocks` vẫn trỏ tới ngăn xếp tầng 0; `load_state_dict` tự đổi khóa `blocks.*` (checkpoint cũ) thành `level_blocks.0.*`.

`HybridFGTNBlock`, `WindowedMHSA`, `NaiveSelectiveSSM`, `NaiveBiMamba` **không đổi**.

Số tham số (từ kích thước tensor trong log): một khối C=768 có 6.957.312 tham số; neck hai khối 13.914.624. Ngăn xếp tầng 3 (C=384, hai khối) thêm ≈3,57 M.

## 2. `heads.py` — head focal + trọng số lớp 🔧 (file mới)

- `weighted_focal_bce(cls_score, labels, gamma, alpha, class_weight)`: hàm thuần PyTorch

  ```
  L = mean_{i,c} [ w_c · α · (1 − p_t)^γ · BCE(s_ic, y_ic) ],   p_t = exp(−BCE)
  ```

  Với `gamma=0, alpha=1, class_weight=None` hàm trùng **chính xác** `F.binary_cross_entropy_with_logits` (được `selfcheck.py` kiểm tra bằng `torch.allclose`).
- `BBoxHeadAVAWeighted(BBoxHeadAVA)`: đăng ký vào `mmaction.MODELS`; kế thừa toàn bộ phần chọn RoI dương, bỏ cột nền và các chỉ số `recall@…/prec@…` của `BBoxHeadAVA.loss_and_target`, chỉ thay biểu thức mất mát. Tham số mới `class_weight`: list độ dài `num_classes − 1` (khi `background_class=True`, tức 80 với AVA) **hoặc** đường dẫn JSON do `make_class_weight.py` sinh ra. `focal_gamma`, `focal_alpha` dùng lại tham số sẵn có của `BBoxHeadAVA`.
- Import `mmaction` được bọc `try/except` để file vẫn import được trên máy không có mmaction (phục vụ self-check).

## 3. `make_class_weight.py` — trọng số lớp theo số mẫu hiệu dụng 🔧 (file mới)

Đọc CSV chú thích AVA (`video_id, giây, x1, y1, x2, y2, action_id, person_id`), đếm `n_c` theo `action_id`, tính

```
w̃_c = (1 − β) / (1 − β^{max(n_c,1)}),  β = 0,999   (Cui et al., CVPR 2019)
w_c  = w̃_c / mean(w̃),  cắt trên tại --clip (mặc định 10)
```

và ghi JSON list 80 phần tử (id 1..80). Lớp không xuất hiện nhận `n_c = 1` (trọng số lớn nhất, bị cắt bởi `--clip`).

```bash
python projects/fgtn_ava/make_class_weight.py \
    --ann data/ava/annotations/ava_train_v2.2_subset.csv \
    --out data/ava/annotations/class_weight_train_subset.json
```

## 4. Config mới: pipeline đầy đủ 🔧

### `ava22_swin-tiny_fgtn-ms-acrn-focal_32x2x1-10e.py`

Kế thừa `ava22_swin-tiny_fgtn_32x2x1-10e.py`, chỉ ghi đè phần `model` để tách đóng góp của ba thành phần:

| Thành phần | Khóa ghi đè | Giải thích |
|---|---|---|
| Đa tỉ lệ | `backbone.out_indices=(2, 3)`; `neck.in_channels=(384, 768)`; `bbox_roi_extractor.featmap_stride=16` | Swin-T chỉ số 0-based: 2 → tầng 3 (384 kênh, H/16), 3 → tầng 4 (768 kênh, H/32). `SingleRoIExtractor3D` của mmaction2 nội suy mọi tầng về kích thước tầng **đầu tiên** (H/16) rồi nối kênh → 1152 kênh, nên tỷ lệ RoI là 1/16. |
| ACRN | `shared_head=dict(type="ACRNHead", in_channels=2304, out_channels=1152)` | Dùng `ACRNHead` có sẵn của mmaction2: vectơ actor (max-pool) tile lên lưới cảnh, nối với bản đồ cảnh 1152 kênh (→ 2304), Conv 1×1×1 + Conv 1×3×3 (BN, ReLU). `AVARoIHead` tự gọi `shared_head(bbox_feats, feat=global_feat, rois=…)`. |
| Focal + trọng số lớp | `bbox_head=dict(type="BBoxHeadAVAWeighted", in_channels=1152, focal_gamma=2.0, focal_alpha=1.0, class_weight=class_weight)` | `class_weight = None` mặc định (chỉ focal); gán đường dẫn JSON qua `--cfg-options class_weight=…`. |

`custom_imports` nạp thêm `projects.fgtn_ava.heads`.

### `ava22_swin-tiny_fgtn-ms-acrn-focal_subset.py`

Như trên nhưng kế thừa config tập con; `train_dataloader.batch_size=4` (tầng 3 có số chuỗi thời gian gấp 4 lần tầng 4) và `work_dir` riêng.

## 5. `ava22_swin-tiny_fgtn_32x2x1-10e.py` — sửa tỷ lệ RoI 🐛

Thêm `featmap_stride=32` vào `bbox_roi_extractor`.

- **Nguyên nhân:** `SingleRoIExtractor3D` mặc định `featmap_stride=16` (đúng cho SlowFast/ViT của mmaction2), nhưng tầng 4 của Video Swin-T có bước giảm mẫu 32 (patch 4 × ba lần gộp patch; ảnh 256 → lưới 8×8).
- **Hệ quả của lần chạy đã log:** hộp được nhân 1/16 thay vì 1/32 → tọa độ gấp đôi; hộp ở nửa phải/dưới ảnh rơi ngoài lưới và nhận đặc trưng 0 (RoIAlign trả 0 ngoài biên), hộp góc trên-trái lấy vùng rộng gấp đôi.
- **Ảnh hưởng tới số liệu:** kết quả 8,50 mAP trong `trainning.log` là của cấu hình **chưa sửa**. Luận văn công bố điều này ở Mục 5.5; cần chạy lại trước khi so sánh với công trình khác. Để tái tạo đúng lần chạy cũ: `--cfg-options model.roi_head.bbox_roi_extractor.featmap_stride=16`.

Config tập con `ava22_swin-tiny_fgtn_subset.py` **không đổi** (kế thừa sửa đổi trên).

## 6. `selfcheck.py` — kiểm tra trên CPU 🔧 (file mới)

`python projects/fgtn_ava/selfcheck.py` (chỉ cần `torch`) kiểm tra:
- neck một tầng với cả bốn `branch_mode`, kích thước ra = vào;
- neck hai tầng `(384, 768)`: tuple vào/ra, mỗi tầng đúng số kênh, không chia sẻ tham số;
- nạp checkpoint cũ (khóa `blocks.*`);
- `weighted_focal_bce` == BCE khi tham số mặc định; focal giảm loss; trọng số nhân đúng;
- `class_weights`: lớp hiếm có trọng số lớn hơn, không vượt `clip`.

Trạng thái: **chưa chạy** trên máy soạn thảo (không có PyTorch). Cần chạy trên máy có torch trước khi huấn luyện.

## 7. `README.md`

Thêm mục "Full pipeline" (bảng ba thành phần, lệnh chạy), ghi chú lỗi `featmap_stride` và lệnh self-check.

---

## Ánh xạ sang luận văn

| Mã | Mục trong `thesis/` |
|---|---|
| `fgtn_neck.py` | Ch.3 §3.3–3.8, Bảng 3.2 (tham số), Phụ lục C.1 |
| `heads.py`, `make_class_weight.py` | Ch.3 §3.11, Phụ lục C.2–C.3 |
| Config đa tỉ lệ / ACRN | Ch.3 §3.9–3.10, Phụ lục A.3 |
| Sửa `featmap_stride` | Ch.5 §5.5, Bảng 4.2 (hàng A5), Phụ lục D |
| Ma trận ablation A0–A9 | Ch.4 §4.7, Phụ lục E |

## Việc còn lại

1. Chạy `selfcheck.py` trên máy có PyTorch.
2. Chạy lại cấu hình cơ sở đã sửa (A5) rồi cập nhật Ch.5 và Phụ lục D.
3. Chạy A0–A3 và A6–A8 để có số liệu cho các thành phần 🔧.

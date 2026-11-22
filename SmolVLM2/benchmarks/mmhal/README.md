# SmolVLM2-2.2B-Instruct × MMHal-Bench

> Cập nhật 19/11/2026: đã chạy inference thật 96/96 mẫu trong
> `.cache/mmhal-results/smolvlm2-mmhal-baseline/full-infer-001` (pilot/exploratory).
> Quy trình chấm hiện dùng GPT-6 tương tác, không bắt buộc GPT-4 API.
> Giữ nguyên rubric/prompt/input/công thức gốc theo
> [protocol GPT-6](../../docs/mmhal_gpt6_protocol.md). Hướng dẫn API phía dưới là
> phương án cũ để đối chứng, không phải bước bắt buộc của luồng hiện tại.
> Kết quả mới nằm ở `results/smolvlm2-mmhal-baseline/gpt6-interactive-001/`.

## Nhập kết quả chấm tương tác

```powershell
python -m benchmarks.mmhal.import_review --source-run .cache/mmhal-results/smolvlm2-mmhal-baseline/full-infer-001 --ledger benchmarks/mmhal/gpt6_review_full_infer_001.tsv --output-dir results/smolvlm2-mmhal-baseline/gpt6-interactive-001
```

Lệnh chỉ nhập điểm đã được judge đưa ra và kiểm chứng công thức, không tự chấm
bằng so khớp đáp án và không gọi API. Nếu thư mục đã tồn tại, dùng thư mục mới;
không chạy lại lệnh này để ghi đè. Ledger chỉ dành cho đúng source checksum đã
review; thay câu trả lời cần lượt chấm mới. Đọc `metrics.json` và
`cases.csv` trong thư mục kết quả. Chỉ số là đánh giá text-reference của judge,
cần review các ca chưa rõ; không phải nhãn thị giác đã xác nhận.

Triển khai riêng cho nhóm MMHal, tuân theo `docs/experiment_protocol.md`,
`docs/guides/mmhal_team.md` và schema kết quả chung. Không sửa `models/`,
`configs/`, `scripts/` hoặc benchmark khác. Adapter hiện thực hai interface chung;
sau review có thể chuyển adapter model sang mã dùng chung.

**Trạng thái:** có loader, adapter inference, judge, artifact/resume và bảng phân
tích ca lỗi. Kiểm thử dùng dữ liệu/model/judge giả lập, **chưa có điểm benchmark
thật**, chưa xác nhận BF16/GPU hay quyền truy cập judge. Cấu hình mẫu cố ý chặn
thực thi khi chưa có bằng chứng phê duyệt. Không đặt APPROVED chỉ để vượt kiểm tra.

## Nguồn và protocol đã đối chiếu

Kiểm tra metadata công khai ngày 19/11/2026:

| Thành phần | Nguồn chính thức | Revision quan sát được |
|---|---|---|
| Dataset | [Shengcao1006/MMHal-Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench) | `f5f49a938f45ed99e235b8519ba28f76832a2add` |
| Model / processor | [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | `be92141c8d0ceb98596a923c308ac2bbb4a7c466` |

Đây là bằng chứng nguồn, không thay thế quyết định khóa phiên bản chung của nhóm.
Model card và dataset card khai báo Apache-2.0; ảnh gốc từ OpenImages/Flickr,
nhóm cần xác minh quyền sử dụng/phân phối ảnh trước khi duyệt `dataset_license`.

Đọc trực tiếp `response_template.json` và `images/` trong snapshot chính thức,
không chạy dataset script từ xa, không tải lại ảnh từ Flickr có thể đã thay đổi.
Đủ 96 câu, 8 loại × 12 chủ đề; giữ thứ tự gốc, `mmhal-000`…`mmhal-095` là chỉ số
dòng gốc, không dùng `image_id` thay sample ID. Không sửa ảnh/câu hỏi/đáp án.

Input VLM: một ảnh RGB qua processor chính thức và **nguyên văn câu hỏi** trong
chat template của model, không system prompt bổ sung. `gt_answer`, `image_content`,
loại và chủ đề câu hỏi chỉ phục vụ judge/phân tích, không đưa vào VLM. Decode chỉ
continuation sau input, lưu token IDs, prompt thực tế và cờ chạm giới hạn token.

Nguồn judge: [eval_gpt4.py tại revision đã khóa](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/blob/f5f49a938f45ed99e235b8519ba28f76832a2add/eval_gpt4.py).
Bản nguyên gốc trong `vendor/` được kiểm tra SHA-256 trước khi trích xuất template
bằng AST (không thực thi script). Giữ nguyên rubric, 5 ví dụ, một user message,
temperature 0 và bốn trường: image contents, question, human answer, model answer.
Judge này nhận thông tin văn bản, không nhận ảnh trực tiếp.

- Judge mặc định gốc: `gpt-4-0314`. **Chưa kiểm chứng còn truy cập được**; nếu không
  khả dụng, ghi lỗi. Không tự đổi sang model khác.
- Dùng judge khác phải sửa `judge.model`, đặt `judge.protocol: alternative`, xin
  duyệt truy cập/chi phí. Không coi điểm là tương đương protocol gốc.
- Điểm 0–6; `<3` là hallucination theo judge, `>=3` là không hallucination.
  Trung bình điểm và tỷ lệ hallucination được tính toàn scope khi chấm đủ.
- Khác biệt có chủ ý: script gốc gán 0 nếu không đọc được điểm. Bản này ghi
  `FAILED`, `rating: null` thay vì biến lỗi parse thành ảo giác. Khi thiếu điểm,
  chỉ số tổng là null, thống kê `valid_only_diagnostic`/theo nhóm chỉ mang tính
  chẩn đoán trên mẫu chấm được, không phải điểm chính thức. Parser còn từ chối
  điểm ngoài thang, số thập phân hoặc các điểm mâu thuẫn.
- Giá và availability chưa xác minh. Không hard-code giá, không gọi API trong
  test. Lưu nguyên response gồm usage, returned model/fingerprint nếu dịch vụ trả
  về, và request trước khi gọi; nhóm tính chi phí theo usage và bảng giá đã duyệt.
  Temperature 0 không bảo đảm judge hoàn toàn xác định.

## Chuẩn bị chạy pilot (từ thư mục gốc repo)

1. Tạo môi trường Python 3.11, cài PyTorch khớp CUDA/GPU, rồi cài dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   python -m pip install -r benchmarks/mmhal/requirements.txt
   python -m benchmarks.mmhal check
   python -m pytest -q
   ```

   Dải phiên bản dependency phục vụ pilot, chưa phải lock môi trường chính thức.
   Run lưu phiên bản thực tế và `pip_freeze.txt`; sau pilot, tái tạo môi trường
   được nhóm chấp nhận từ freeze. CUDA, native BF16 và RAM/VRAM cần kiểm chứng.

2. Sao chép `benchmarks/mmhal/config.yaml` vào `.cache/mmhal-pilot.yaml`, giữ
   cấu hình nghiên cứu trong bản được review trước khi chạy chính thức. Sau khi
   được duyệt, điền **tham chiếu PR/biên bản/bằng chứng** vào `approvals.inference`,
   `dtype_hardware`, `generation`, `dataset_license`. Đề xuất pilot: greedy,
   256 token, BF16, SDPA, một GPU và batch size 1. Chưa coi các lựa chọn này là
   cấu hình chính thức đã được duyệt. Model adapter chặn GPU không có native
   BF16; không fallback dtype, lượng tử hóa, offload hoặc fine-tune.

3. Chỉ sau phê duyệt, tải snapshot và ghi provenance (lệnh này có mạng và tải
   trọng số/dataset, không được chạy như một kiểm tra offline):

   ```powershell
   python -m benchmarks.mmhal prepare --config .cache/mmhal-pilot.yaml --cache-dir .cache/mmhal-hub
   ```

   Lệnh in `Data snapshot`. Đặt `DATA_ROOT` thành đường dẫn được in; `MODEL_CACHE_DIR`
   là `.cache/mmhal-hub`. Không chỉnh `mmhal_provenance.json`. Mọi lần chạy kiểm
   lại checksum template và toàn bộ ảnh. Không đưa cache, trọng số, ảnh hay key vào Git.

4. Chạy inference bằng file config đã duyệt (đọc model/processor từ cache, không
   tự tải khi thiếu):

   ```powershell
   python -m benchmarks.mmhal infer --config .cache/mmhal-pilot.yaml --data-root $env:DATA_ROOT --cache-dir .cache/mmhal-hub --run-id pilot-infer-001
   ```

   Kết quả: `results/smolvlm2-mmhal-baseline/pilot-infer-001/`. Khi pilot chỉ dùng
   8 mẫu đầu (cùng một chủ đề, đủ 8 loại câu hỏi), chỉ xác minh kỹ thuật, không
   suy rộng hiệu năng. Không lặng lẽ lọc câu sai/ảnh khó. Có thể kiểm tra trên
   Kaggle và máy cá nhân bằng cùng snapshot/config; GPU không đủ điều kiện được
   ghi `BLOCKED`. Trên Linux/Kaggle dùng biến shell tương ứng với `$env:DATA_ROOT`.

5. Sau khi duyệt quyền truy cập/chi phí judge, điền `approvals.judge_access_cost`.
   Key chỉ lấy từ environment `OPENAI_API_KEY` hoặc secret manager, không truyền
   trên CLI/notebook đã lưu. Chạy judge ra một run khác:

   ```powershell
   python -m benchmarks.mmhal judge --config .cache/mmhal-pilot.yaml --data-root $env:DATA_ROOT --run-id pilot-judge-001 --predictions-run results/smolvlm2-mmhal-baseline/pilot-infer-001
   ```

   Có thể chấm lại inference bằng judge khác đã duyệt trong run mới. Model,
   dataset, seed, scope và generation phải khớp inference. API timeout/không truy
   cập/parse lỗi được lưu riêng; không tự retry gây chi phí vô hạn.

6. Kiểm tra và xuất bảng phân tích (dùng được cả run inference chưa chấm):

   ```powershell
   python -m benchmarks.mmhal verify --predictions-run results/smolvlm2-mmhal-baseline/pilot-judge-001
   python scripts/check_results.py results/smolvlm2-mmhal-baseline/pilot-judge-001/predictions.jsonl
   python scripts/check_manifest.py results/smolvlm2-mmhal-baseline/pilot-judge-001
   python -m benchmarks.mmhal analyze --predictions-run results/smolvlm2-mmhal-baseline/pilot-judge-001 --analysis-dir .cache/mmhal-review-001
   ```

## Chạy đủ, artifact và resume

Sau pilot và review, đổi `stage: official`, `sample_ids: null`, ghi bằng chứng
`approvals.official`, khóa config/mã nguồn bằng commit và chạy từ checkout sạch.
Không tự sửa cấu hình chung đang pending. Quy trình chấp nhận config này cần
được trưởng nhóm review trước khi áp dụng cho các model/benchmark khác.

Run có đủ `predictions.jsonl`, `metrics.json`, `run_config.yaml`,
`run_metadata.json`, `errors.jsonl`, `run_status.json`, `manifest.json`, progress;
inference còn có freeze và processor configuration, judge có raw requests/responses.
`responses.json` theo đúng schema/thứ tự upstream chỉ xuất khi đủ 96 prediction
thành công. Không điền lỗi kỹ thuật thành câu trả lời để xuất file này.

Hash nhận diện bao gồm config, nội dung dữ liệu, code, Git commit, môi trường và
GPU; judge còn gắn checksum predictions đầu vào. Manifest có checksum từng file.
`verify` kiểm tra các checksum, mạnh hơn bộ kiểm tra manifest chung hiện tại.
Không công bố artifact giả lập như kết quả nghiên cứu.

Run mới cần run ID mới. `--resume` chỉ tiếp tục run chưa hoàn tất, cùng hash;
bỏ qua **mọi mẫu đã có bản ghi**, kể cả mẫu thất bại, không sửa prediction cũ.
Muốn thử lại mẫu thất bại tạo run mới và giữ bằng chứng run trước. Sau crash,
tiếp tục các mẫu chưa được ghi; JSONL bị ghi dở sẽ bị từ chối thay vì tự xóa dòng.
Một writer/run; nếu tiến trình bị kill để lại `.writer.lock`, chỉ xóa lock sau
khi xác nhận tiến trình thực sự đã dừng và đã sao lưu run.

Judge có thể đã tính phí nếu tiến trình chết sau request nhưng trước prediction.
Trước resume, kiểm tra `judge_requests.jsonl` và `judge_responses.jsonl` để đối
soát; không có bảo đảm exactly-once qua mạng. Không tự resume một request chưa rõ
trạng thái tính phí. Lỗi kỹ thuật không được tính vào mẫu số hallucination như
một prediction hợp lệ. Khi chưa chấm đủ, không công bố điểm tổng.

## Mổ xẻ từng ca, chưa can thiệp model

`analyze` xuất `cases.csv` (UTF-8 BOM), `summary.json` và nguồn manifest vào thư
mục mới. Giữ nguyên artifact gốc, bảng review có thể chỉnh sửa. Có tất cả mẫu
trong scope, gồm mẫu đúng, mẫu judge đánh dấu và lỗi kỹ thuật. `image_id` đối chiếu
template để mở ảnh gốc; `image_sha256` trong prediction giúp xác nhận đúng ảnh.

Điền các cột: `human_label`, `unsupported_claim` (trích đúng mệnh đề sai),
`visual_evidence` (vị trí/đối tượng trong ảnh), `error_category`, `cause_hypothesis`,
`reviewer`, `notes`. Nhãn gợi ý: hallucination, incorrect_prediction,
format_violation, technical_error, label_data_error, undetermined, correct.
Loại câu hỏi dataset không tự động trở thành loại lỗi.

| Nhóm câu hỏi | Điều cần đối chiếu trên từng ảnh |
|---|---|
| attribute | Màu, hình dạng, thuộc tính gán sai cho đối tượng nào? |
| adversarial | Model có chấp nhận tiền đề về vật/người không có thật không? |
| comparison | So sánh nhầm đối tượng hoặc nhầm thuộc tính nào? |
| counting | Đếm thiếu/thừa, vật nhỏ/che khuất có xuất hiện không? |
| relation | Sai trái/phải, trước/sau, hay liên kết thuộc tính với vật? |
| environment | Suy diễn thời tiết, địa điểm, bối cảnh thiếu bằng chứng nào? |
| holistic | Mệnh đề thêm vào mô tả dài nào không có căn cứ? |
| other | OCR, biểu tượng hoặc bước suy luận cụ thể nào sai? |

Phân biệt từ chối/không đủ thông tin với khẳng định sai. Kiểm tra cờ truncation
trước khi quy lỗi cho khả năng thị giác. Judge chỉ là ứng viên cần kiểm tra,
đáp án chuẩn cũng có thể không đầy đủ. Nên có hai người review độc lập các ca
khó và ghi bất đồng. So sánh tỷ lệ/điểm theo loại và chủ đề cùng số mẫu, liệt kê
các ca cụ thể; baseline này mô tả **mẫu lỗi quan sát được**, chưa chứng minh nguyên
nhân bên trong model. Chỉ đề xuất kỹ thuật phát hiện/cải thiện sau khi có dữ liệu
thật và phân tích ca, chưa thay đổi model trong bước benchmark.

## Kiểm thử và giới hạn bàn giao

```powershell
python -m pytest -q benchmarks/mmhal/test_mmhal.py
python scripts/check_repository.py
python -m benchmarks.mmhal check
```

Test không tải model/ảnh, không gọi API; synthetic records chỉ nằm trong thư mục
tạm của pytest. `scripts/check_config.py`/runner chung vẫn báo pending vì config
chung chưa được chốt; runner MMHal riêng không tuyên bố đã hoàn tất 9 tổ hợp.
Thử nghiệm thật, model API compatibility và toàn bộ checklist GPU/Kaggle/chi phí
vẫn cần được xác nhận trước khi nghiệm thu kết quả khoa học.

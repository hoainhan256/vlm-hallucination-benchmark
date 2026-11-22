"""Build an evidence report from preserved inference, text judge and visual ledger."""
import csv
import json
import hashlib
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/smolvlm2_mmhal_20261119'
OUT.mkdir(parents=True, exist_ok=True)
source = ROOT / '.cache/mmhal-results/smolvlm2-mmhal-baseline/full-infer-001'
judge = ROOT / 'results/smolvlm2-mmhal-baseline/gpt6-interactive-001'
images = ROOT / '.cache/mmhal-hub/datasets--Shengcao1006--MMHal-Bench/snapshots/f5f49a938f45ed99e235b8519ba28f76832a2add/images'
responses = json.loads((source/'responses.json').read_text(encoding='utf-8'))
judged = [json.loads(s) for s in (judge/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
metrics = json.loads((judge/'metrics.json').read_text(encoding='utf-8'))
meta = json.loads((source/'run_metadata.json').read_text(encoding='utf-8'))
with Path(__file__).with_name('visual_review_001.tsv').open(encoding='utf-8') as f:
    visual = list(csv.DictReader(f, delimiter='\t'))
assert [int(r['index']) for r in visual] == list(range(96))
names={'correct':'Phù hợp','incomplete':'Thiếu hoặc mơ hồ','hallucination':'Lỗi có bằng chứng','undetermined':'Chưa kết luận'}
counts=Counter(v['label'] for v in visual)
review=[]
for i,(r,j,v) in enumerate(zip(responses,judged,visual)):
    review.append(dict(sample_id=f'mmhal-{i:03d}',question_type=r['question_type'],question_topic=r['question_topic'],
        question=r['question'],reference=r['gt_answer'],model_answer=r['model_answer'],text_rating=j['rating'],
        text_judge_hallucination=j['rating']<3,visual_label=v['label'],visual_evidence=v['evidence'],
        image_file=Path(urlparse(r['image_src']).path).name,hit_token_limit=j['hit_token_limit'],
        reviewer='GPT-6 interactive visual review',independent_human_review='not_performed'))
with (OUT/'review_96_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(review[0]));w.writeheader();w.writerows(review)
summary=dict(protocol='supplemental_image_assisted_review_v1',count=96,labels=dict(counts),
    original_text_judge_average=metrics['average_score'],original_text_judge_rate=metrics['hallucination_rate'],
    not_an_official_rescore=True,independent_human_review=False,
    review_method='All 96 images screened in 12 contact sheets; selected ambiguous cases opened at full resolution',
    limitations='Same GPT-6 reviewer as text judge; image screening is not independent human ground truth',
    source_run_id='full-infer-001',source_stage='pilot',source_git_dirty=meta['git_dirty'])
(OUT/'visual_review_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

# Chart drawn from exact stored metrics, labels and denominators shown explicitly.
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
chart=Image.new('RGB',(1300,690),'white'); draw=ImageDraw.Draw(chart)
types=['attribute','adversarial','comparison','counting','relation','environment','holistic','other']
draw.text((20,15),'Text judge hallucination rate by question type',font=font,fill='black')
for k,t in enumerate(types):
    y=85+k*65; rate=metrics['by_question_type'][t]['hallucination_rate'];n=round(rate*12)
    draw.text((20,y),t,font=font,fill='black');draw.rectangle((230,y,950,y+32),fill='#eeeeee')
    draw.rectangle((230,y,230+720*rate,y+32),fill='#385b75')
    draw.text((975,y),f'{n}/12  {rate:.1%}',font=small,fill='black')
draw.text((230,620),'Each type has 12 samples; GPT-6 text-reference judge',font=small,fill='black')
chart.save(OUT/'rates.png')

doc=Document();sec=doc.sections[0]
sec.page_height=Inches(11.69);sec.page_width=Inches(8.27)
sec.top_margin=sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.7)
for s in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
    st=doc.styles[s];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0)
doc.styles['Normal'].font.size=Pt(10.5)
doc.styles['Normal'].paragraph_format.space_after=Pt(6)
doc.styles['Normal'].paragraph_format.line_spacing=1.08
doc.styles['Title'].font.size=Pt(27)
doc.styles['Heading 1'].font.size=Pt(18)
doc.styles['Heading 2'].font.size=Pt(12)
for style in doc.styles:
    for border in list(style.element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
footer=sec.footer.paragraphs[0];footer.alignment=2
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
def p(s,style=None): return doc.add_paragraph(s,style)
def h(s):doc.add_heading(s,1)
def page(title):doc.add_page_break();h(title)
def table(headers,rows,widths=None,size=9):
    t=doc.add_table(rows=1,cols=len(headers));t.autofit=False
    if widths:
        for col,width in zip(t.columns,widths):col.width=Inches(width)
    for c,s in zip(t.rows[0].cells,headers):c.text=str(s)
    repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        for c,s in zip(t.add_row().cells,row):c.text=str(s)
    for ri,row in enumerate(t.rows):
        trpr=row._tr.get_or_add_trPr();cannot=OxmlElement('w:cantSplit');trpr.append(cannot)
        for ci,c in enumerate(row.cells):
            if widths:c.width=Inches(widths[ci])
            tcpr=c._tc.get_or_add_tcPr(); borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
            tcpr.append(borders)
            if ri==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EEF2');tcpr.append(shade)
            for para in c.paragraphs:
                para.paragraph_format.space_after=Pt(4);para.paragraph_format.space_before=Pt(4)
                for run in para.runs:run.font.size=Pt(size);run.bold=ri==0
    return t
def photo(i,width=3):
    path=images/review[i]['image_file']; im=Image.open(path);im.thumbnail((1100,850));target=OUT/f'case_{i:03d}.jpg';im.convert('RGB').save(target)
    doc.add_picture(str(target),width=Inches(width))

p('Báo cáo benchmark và phân tích lỗi', 'Title')
p('SmolVLM2 2.2B Instruct trên MMHal Bench','Subtitle')
p('Dự án xây dựng pipeline dữ liệu đa phương thức phục vụ kiểm chứng thông tin và giảm ảo giác trong hệ thống trí tuệ nhân tạo')
p('Ngày 19 tháng 11 năm 2026  |  Phạm vi một model và một dataset')
h('Kết quả chính')
p('Đã chạy đủ 96 câu hỏi bằng SmolVLM2-2.2B-Instruct. GPT-6 chấm theo đầu vào văn bản và rubric MMHal gốc cho điểm 3,28125/6, với 34/96 câu dưới 3 điểm. Đối chiếu ảnh bổ sung cho thấy cần tách lỗi model khỏi nhãn tham chiếu thiếu chi tiết và sai lệch của judge.')
table(['Thành phần','Kết quả'],[
['Inference','96/96 thành công; không có lỗi kỹ thuật trong run đầy đủ'],
['Judge theo văn bản','315 điểm / 96 = 3,28125; hallucination 34/96 = 35,42%'],
['Review ảnh bổ sung','96/96 đã sàng lọc bằng ảnh; cùng GPT-6, chưa có reviewer người độc lập'],
['Nhãn review ảnh', '; '.join(f'{names[k]}: {counts[k]}' for k in names)],
['Giới hạn sinh','10/96 câu chạm 256 token, đều thuộc nhóm holistic']],[1.8,4.9])
p('Kết luận thực nghiệm: model gặp lỗi tạo đối tượng theo tiền đề sai, đếm, gắn thuộc tính theo vị trí, OCR và thêm chi tiết vào mô tả dài. Không dùng kết quả này để khẳng định nguyên nhân bên trong model hoặc hiệu quả của một kỹ thuật giảm ảo giác chưa được thử.')
p('Tài liệu này cung cấp kết quả định lượng, case có ảnh minh họa, phụ lục 96 mẫu và cấu trúc bàn giao để tổng hợp với các nhóm khác.')

page('Thiết kế thực nghiệm và phương pháp chấm')
table(['Thông số','Giá trị'],[
['Model','HuggingFaceTB/SmolVLM2-2.2B-Instruct'],['Model và processor revision','be92141c8d0ceb98596a923c308ac2bbb4a7c466'],
['Dataset revision','f5f49a938f45ed99e235b8519ba28f76832a2add'],['Phạm vi','96 câu, 8 loại câu hỏi × 12 chủ đề; giữ thứ tự gốc'],
['Inference','BF16; CUDA:0; SDPA; greedy; seed 0; batch 1; max_new_tokens 256'],
['Máy chạy','RTX 5060 8 GB; CUDA 12.8; Python 3.11.9'],
['Thư viện', 'PyTorch 2.11.0+cu128; Transformers 5.23.0; Accelerate 1.16.0'],
['Nguồn run','full-infer-001; stage pilot; Git dirty được ghi trong metadata']],[1.7,5])
doc.add_heading('Vòng một chấm theo văn bản tham chiếu',2)
p('Giữ template gốc, năm ví dụ và bốn trường image_content, question, gt_answer, model_answer. GPT-6 thay GPT-4, giải thích rồi cho một điểm 0–6. Không thêm ảnh vào vòng chấm này. Chi tiết không có trong reference không tự động là sai; từ chối và câu trả lời thiếu không tự động là hallucination.')
p('Điểm 0–2: có hallucination, mức độ informative tăng dần. Điểm 3–6: không hallucination, từ không informative đến rất informative kèm suy luận tốt. Điểm trung bình là tổng điểm/96; hallucination rate là số điểm <3/96. Mỗi loại có 12 mẫu, nhóm theo chỉ số dòng modulo 8 như script gốc.')
doc.add_heading('Vòng hai đối chiếu ảnh',2)
p('Sàng lọc 96 ảnh qua 12 bảng ảnh, mở kích thước gốc ở các ca khó chọn lọc. Ghi nhãn phù hợp, thiếu/mơ hồ, lỗi có bằng chứng hoặc chưa kết luận. Đây là review bằng GPT-6, không giả là human ground truth; không sửa điểm vòng một và không gộp hai vòng thành một metric.')
p('Khác biệt với API gốc: cùng phiên tương tác, không có 96 request độc lập; không truy cập được snapshot, temperature hoặc seed của judge. Vì vậy số liệu chỉ so sánh được khi các nhóm dùng cùng protocol, không coi tương đương điểm GPT-4 công bố.')

page('Kết quả theo nhóm câu hỏi')
doc.add_picture(str(OUT/'rates.png'),width=Inches(6.65))
table(['Loại','Điểm TB','Điểm dưới 3','Tỷ lệ'],[[t,f'{metrics["by_question_type"][t]["average_score"]:.4f}',f'{round(metrics["by_question_type"][t]["hallucination_rate"]*12)}/12',f'{metrics["by_question_type"][t]["hallucination_rate"]:.2%}'] for t in types],[2.1,1.2,1.5,1.8])
p('Holistic có 7/12 câu dưới 3; adversarial và comparison mỗi loại 6/12. Đây là phân bố quan sát trên 12 mẫu mỗi nhóm, không phải ước lượng hiệu năng tổng quát cho mọi ảnh. Nhóm environment có ít điểm dưới 3 nhưng vẫn có ca không đủ chứng cứ thời tiết.')
p('10/12 output holistic chạm giới hạn token. Chưa có thực nghiệm đối chứng thay max_new_tokens, vì vậy không kết luận truncation gây hallucination. Không cộng hình phạt điểm riêng cho truncation.')

page('Các case lỗi có bằng chứng thị giác')
for i,title in [(17,'Tiền đề sai và đối tượng không tồn tại'),(14,'Gán nhầm vật thể và hành động')]:
    doc.add_heading(f'{review[i]["sample_id"]}  {title}',2);photo(i,2.7)
    excerpt = responses[i]['model_answer'].strip() if i == 17 else "The cyclist is wearing a helmet and is holding a stop sign."
    p('Model trích đoạn: '+excerpt);p('Bằng chứng: '+review[i]['visual_evidence'])
    p('Nhận xét: '+('Nét vẽ trên mặt bị diễn giải thành kính. Cần kiểm tra sự hiện diện của vật thể trước khi mô tả thuộc tính.' if i==17 else 'Nhận ra biển STOP nhưng gắn sai với người đi xe đạp thay vì xe buýt. Đây là lỗi liên kết, không chỉ nhận diện vật.'))

page('Đếm sai và diễn giải bố cục')
for i,title in [(67,'Một thiết bị bị đếm thành hai'),(30,'Đếm sai trong mô tả dài')]:
    doc.add_heading(f'{review[i]["sample_id"]}  {title}',2);photo(i,3)
    excerpt = responses[i]['model_answer'].strip() if i == 67 else 'The foreground features a group of four zebras. In the middle ground, there are two giraffes.'
    p('Model trích đoạn: '+excerpt);p('Bằng chứng: '+review[i]['visual_evidence'])
p('Hai kiểu khó khác nhau: các phần của cùng vật thể bị tách thành nhiều vật (067), và nhiều con vật chồng lấp trong cảnh rộng (030). Cần lưu cả lỗi đếm trực tiếp và lỗi đếm ẩn trong mô tả, không chỉ đánh giá loại counting.')

page('Reference và judge cũng cần kiểm chứng')
for i in [78,86,87]:
    doc.add_heading(review[i]['sample_id'],2)
    p('Reference: '+responses[i]['gt_answer']);p('Đối chiếu ảnh: '+review[i]['visual_evidence'])
    p('Ý nghĩa: '+{78:'Nhận định ban đầu nghi giường tầng không có căn cứ đã được sửa. Lỗi thực tế của model nằm ở sàn gạch, không phải sự tồn tại của giường tầng.',86:'Không dùng diaper trong reference để phủ định onesie model nhận đúng. Model vẫn nhầm một thú bông thành voi; lý do gán lỗi cần chuyển sang chi tiết này.',87:'Tách OCR bỏ sót chữ cái trang trí khỏi bịa tên sách. Điểm vòng một giữ nguyên, nhãn review bổ sung là trả lời thiếu.'}[i])
doc.add_heading('Các ca giữ trạng thái chưa kết luận',2)
p('005, 029, 085: ảnh không đủ xác nhận thời tiết thực. 015: ảnh có cả United và Virgin, cần quy chiếu đúng máy bay phía sau. 058: không biết nội dung túi đóng. 063: phản chiếu nhỏ. 069: địa điểm quá chung. 081: ranh giới gọi pot/vase. Không ép những ca này thành nhãn nhị phân chắc chắn.')
p('Không có reviewer người thứ hai hoặc đo đồng thuận liên đánh giá trong lượt này. Review ảnh do cùng GPT-6 thực hiện giúp sửa lập luận, nhưng chưa thay thế nghiệm thu của nhóm/mentor.')

page('Kết luận và hướng nghiên cứu tiếp theo')
table(['Mẫu lỗi quan sát','Case','Hướng phát hiện đề xuất'],[
['Tiền đề sai','017 025 065 089','Kiểm tra tồn tại đối tượng/hành động trước khi chấp nhận câu hỏi'],
['Đếm và hợp nhất thực thể','003 011 030 067 094','Đối chiếu đếm với vùng ảnh, lưu bằng chứng instance'],
['Vị trí và thuộc tính','008 036 052 090','Tách đối tượng, thuộc tính, quan hệ; kiểm tra thứ tự'],
['OCR và định danh','022 047 087','So chữ gốc và ngữ cảnh; phân biệt thiếu ký tự với sai tên'],
['Mô tả dài thêm chi tiết','006 014 046 070','Tách câu trả lời thành mệnh đề để kiểm chứng từng mệnh đề']],[1.65,1.2,3.85])
p('Các hướng trên là đề xuất từ lỗi đã thấy, chưa được triển khai hoặc chứng minh hiệu quả. Bước benchmark của mentor tập trung hiểu lỗi trước; chưa fine-tune hay thay decoder để tìm điểm tốt hơn.')
p('Thiết kế bước sau: khóa tập case, định nghĩa nhãn và hướng dẫn review; kiểm tra thuật toán phát hiện trên dữ liệu tách biệt với tập chọn kỹ thuật. Báo precision/recall/F1 cho phát hiện hallucination, đồng thời giữ nhóm uncertain ngoài nhãn chắc chắn. Khi thử cải thiện, so baseline và biến thể trên cùng nguồn, judge và cấu hình, kèm chi phí và số lỗi kỹ thuật.')
doc.add_heading('Giới hạn kết luận',2)
p('Một model, một dataset nhỏ, một seed và một lần chạy đầy đủ. Judge thay thế và review ảnh cùng họ model có thể có thiên lệch. Nguồn inference là exploratory, chưa có checkout sạch. Thông số 256 token ảnh hưởng độ đầy đủ của mô tả. Các mối liên hệ quan sát không chứng minh cơ chế gây lỗi trong model.')

page('Bàn giao và sẵn sàng tổng hợp liên nhóm')
table(['Mục tiêu mentor','Bằng chứng bàn giao'],[
['Tự chạy thực nghiệm','96/96 prediction và metadata môi trường'],['Biết model sai gì','Phân loại lỗi và phân bố theo tám loại câu hỏi'],
['Liệt kê rõ từng case','Phụ lục 96 mẫu và review_96_cases.csv với output đầy đủ'],
['Rút ra nhận xét','Case có ảnh, đối chiếu nhãn, giới hạn và giả thuyết được tách riêng'],
['Chuẩn bị bước phát hiện/cải thiện','Ma trận lỗi → đề xuất; chưa tuyên bố đã thử kỹ thuật']],[2.1,4.6])
doc.add_heading('Trường thông tin bắt buộc cho các nhóm',2)
p('Model ID/revision; dataset/revision/scope; prompt và preprocessing; dtype/device; generation; source run/hash; inference success/failure; judge ID và protocol; số mẫu chấm được; metric và mẫu số; truncation; taxonomy; case có ảnh; nhãn uncertain; trạng thái human review. Mẫu dữ liệu tổng hợp đi kèm là integration_record.json.')
p('Chỉ so trực tiếp các model cùng dataset, scope, metric và protocol judge. Với POPE/HallusionBench, giữ bảng metric riêng, không lấy trung bình điểm MMHal với accuracy/F1 của benchmark khác. Các ô chưa có kết quả nhóm khác phải để trống, không nội suy.')
doc.add_heading('Nguồn và cách truy xuất',2)
for text in ['Dataset và scorer: https://huggingface.co/datasets/Shengcao1006/MMHal-Bench',
             'Model card: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct',
             'Protocol dự án: docs/mmhal_gpt6_protocol.md; source scorer lưu tại benchmarks/mmhal/vendor/eval_gpt4.py.txt',
             'Inference bất biến: .cache/mmhal-results/smolvlm2-mmhal-baseline/full-infer-001',
             'Judge vòng một: results/smolvlm2-mmhal-baseline/gpt6-interactive-001',
             'Báo cáo và review bổ sung: reports/smolvlm2_mmhal_20261119']:
    p(text)

for typ in types:
    page('Phụ lục case '+typ)
    p('Điểm là vòng chấm text-reference gốc; nhãn và bằng chứng là review ảnh bổ sung. Mã rút gọn 000 tương ứng mmhal-000. Output nguyên văn đầy đủ nằm trong CSV kèm theo.')
    subset=[r for r in review if r['question_type']==typ]
    table(['ID và điểm','Kết luận ảnh','Bằng chứng và nhận xét'],[[r['sample_id'][6:]+'  '+str(r['text_rating'])+'/6',names[r['visual_label']],r['visual_evidence']] for r in subset],[.85,1.25,4.6],9)
    p('Không thay thế hai reviewer độc lập. Đối tượng nhỏ, ảnh che khuất và thuật ngữ mơ hồ cần thận trọng khi nghiệm thu. Không loại bỏ case khó hoặc sửa reference gốc.')

file=OUT/'Bao_cao_SmolVLM2_MMHal.docx';doc.save(file)
integration=dict(model_id='HuggingFaceTB/SmolVLM2-2.2B-Instruct',dataset='Shengcao1006/MMHal-Bench',n=96,
    inference_success=96,source_run='full-infer-001',stage='exploratory',judge='GPT-6 interactive text-reference',
    average_score=metrics['average_score'],hallucination_rate=metrics['hallucination_rate'],
    token_limit_count=10,visual_review_counts=dict(counts),human_review=False,original_gpt4_comparable=False,
    dataset_revision=meta['dataset_revision'],model_revision=meta['model_revision'],generation={'do_sample':False,'max_new_tokens':256,'seed':0})
(OUT/'integration_record.json').write_text(json.dumps(integration,ensure_ascii=False,indent=2),encoding='utf-8')
manifest={'source_predictions_sha256':hashlib.sha256((source/'predictions.jsonl').read_bytes()).hexdigest(),
    'source_judge_predictions_sha256':hashlib.sha256((judge/'predictions.jsonl').read_bytes()).hexdigest(),
    'review_ledger_sha256':hashlib.sha256(Path(__file__).with_name('visual_review_001.tsv').read_bytes()).hexdigest(),
    'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.docx','.json','.csv'] and p.name!='manifest.json'}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(file);print(counts)

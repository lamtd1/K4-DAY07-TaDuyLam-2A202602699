# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Tạ Duy Lâm
**Nhóm:** Opera
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding chỉ về gần cùng một hướng (cosine gần 1), tức hai đoạn văn có nghĩa gần nhau, dù có thể dùng từ khác nhau. Cosine gần 0 nghĩa là không liên quan, gần -1 là ngược hướng.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên được mượn sách về nhà tối đa 10 ngày."
- Câu B: "Người đang theo học có thể mang tài liệu của thư viện ra ngoài trong vòng mười hôm."
- Tại sao tương đồng: hai câu gần như không có từ nào trùng ("sinh viên" ↔ "người đang theo học", "sách" ↔ "tài liệu", "10 ngày" ↔ "mười hôm") nhưng cùng nói về hạn mượn của người học, nên embedding hiểu nghĩa sẽ cho cosine cao. (Đây là ví dụ dự đoán theo cách embedding thật hoạt động, chưa đo bằng model thật; `MockEmbedder` không có ngữ nghĩa nên không dùng để kiểm chứng.)

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên được mượn sách về nhà tối đa 10 ngày."
- Câu B: "Đội tuyển bóng đá thắng trận chung kết tối qua."
- Tại sao khác: hai câu khác chủ đề hoàn toàn (quy định thư viện và thể thao), không chia sẻ ý nghĩa nào nên hai vector gần vuông góc, cosine thấp (gần 0).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ so hướng của vector, còn nghĩa của văn bản nằm ở hướng chứ không ở độ dài; độ dài vector có thể thay đổi theo độ dài văn bản hay tần suất từ, và khoảng cách Euclid bị ảnh hưởng bởi độ dài đó nên hai đoạn cùng nghĩa nhưng dài ngắn khác nhau có thể bị coi là xa nhau. Với vector đã chuẩn hóa (||v|| = 1) như trong repo thì hai cách xếp hạng giống nhau (||a-b||² = 2 - 2·cos), nên dot product bằng đúng cosine.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính: số chunk = ceil((10000 − 50) / (500 − 50)) = ceil(9950 / 450) = ceil(22,11) = 23. Mỗi chunk mới tiến thêm 450 ký tự (bước = chunk_size − overlap); chunk cuối bắt đầu ở ký tự 9900 và chỉ dài 100 ký tự.
> Đáp án: **23 chunks**. Đã kiểm bằng `FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)` trong repo: `len(...) = 23`, khớp công thức.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Số chunk tăng từ 23 lên 25: ceil((10000 − 100) / (500 − 100)) = ceil(24,75) = 25 (đã kiểm bằng `FixedSizeChunker`), vì bước tiến giảm còn 400 ký tự. Overlap lớn hơn giúp một ý hoặc câu nằm ở ranh giới hai chunk xuất hiện ở cả hai, giảm nguy cơ mất ngữ cảnh khi truy xuất; đổi lại tốn thêm chunk, dung lượng lưu trữ và chi phí embedding, và top-k dễ chứa các chunk trùng nội dung.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tách bằng `re.split(r"(?<=[.!?])\s+", text.strip())`: lookbehind cắt ngay sau dấu câu nên dấu `.`, `!`, `?` vẫn nằm trong câu (nếu dùng `[.!?]\s+` thì dấu câu bị nuốt), rồi gom mỗi `max_sentences_per_chunk` câu thành một chunk và nối bằng một khoảng trắng. Text rỗng hoặc chỉ có khoảng trắng trả về `[]`. Edge case chưa xử lý: chữ viết tắt bị cắt sai (đã thử: "TS. Nguyễn Văn A giảng dạy." bị tách thành "TS." và phần còn lại); số thập phân như "3.5" hoặc "1.000đ" thì không bị cắt vì sau dấu chấm không có khoảng trắng, nhưng "3. 5" sẽ bị cắt. Ngoài ra, với Markdown như corpus của nhóm, các dòng gạch đầu dòng kết thúc bằng ";" không có dấu kết câu nên có thể dồn thành một "câu" rất dài, và các dòng xuống hàng bị nối thành một dòng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thử lần lượt separator theo thứ tự `["\n\n", "\n", ". ", " ", ""]`: cắt bằng ranh giới lớn trước, mảnh nào vẫn dài hơn `chunk_size` thì gọi đệ quy `_split` với các separator còn lại (đệ quy xuống), còn các mảnh nhỏ liền kề được nối lại cho tới sát `chunk_size` (gom lên) để tránh sinh hàng loạt chunk vụn; thử với 100 dòng ngắn cho ra 17 chunk dài 39 đến 59 ký tự thay vì 100 chunk. Base case: (1) đoạn đã ≤ `chunk_size` thì trả về nguyên đoạn; (2) hết separator hoặc gặp separator `""` thì cắt cứng theo `chunk_size` (nên `separators=[]` không lỗi); (3) separator không xuất hiện trong đoạn thì chuyển sang separator kế tiếp.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu in-memory (bỏ hẳn nhánh ChromaDB vì test không cần và có thể rẽ nhầm nhánh chưa cài đặt): mỗi `Document` thành một record `{id, content, metadata, embedding}` trong `self._store`, 1 Document = 1 record (việc chunking làm ở tầng ngoài). `_make_record` copy metadata để tránh bị sửa từ bên ngoài và dùng `setdefault("doc_id", doc.id)` để chunk `file#0` vẫn trỏ về file gốc nếu người gọi đã gán `doc_id`. `_search_records` embed query, tính cosine bằng `compute_similarity` với từng record, sắp xếp giảm dần theo `score` và trả `top_k` kết quả (không kèm embedding); `search` và `search_with_filter` cùng đi qua hàm này nên không thể lệch nhau.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Lọc trước rồi mới xếp hạng: chỉ giữ record có mọi cặp khóa-giá trị của `metadata_filter` khớp, sau đó chạy `_search_records` trên tập đó. Nếu lấy top-k trước rồi mới lọc thì k vị trí có thể bị tài liệu sai đối tượng chiếm hết và còn 0 kết quả dù vẫn có tài liệu hợp lệ (đã thử: 1 tài liệu `student` giữa 20 tài liệu `faculty` vẫn được trả về với `top_k=3`). `delete_document` giữ lại các record có `metadata["doc_id"]` khác `doc_id`, trả `True` nếu số record giảm, ngược lại `False`; nhờ vậy xóa một file sẽ xóa mọi chunk của nó.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Ba bước: `store.search(question, top_k)` → dựng prompt → `llm_fn(prompt)`. Ngữ cảnh được đánh số `[1]`, `[2]`... kèm nguồn (`source_url`/`source`/`doc_id` và id chunk) để câu trả lời truy vết được về đúng chunk và file; prompt yêu cầu chỉ dùng ngữ cảnh đã cho, nói rõ khi không có thông tin, và trích dẫn số đoạn. Nếu store rỗng thì trả câu thông báo "không tìm thấy" và không gọi LLM.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0 -- /Users/taduylam/Workspace/K4-DAY07-TaDuyLam-2A202602699/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/taduylam
configfile: pyproject.toml
plugins: anyio-4.15.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.03s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên được mượn sách về nhà tối đa 10 ngày. | Người đang theo học có thể mang tài liệu của thư viện ra ngoài trong vòng mười hôm. | cao | 0,711 | Đúng |
| 2 | Sinh viên được mượn sách về nhà tối đa 10 ngày. | Giảng viên được mượn sách về nhà tối đa 180 ngày. | cao | 0,727 | Đúng |
| 3 | Trả sách trễ hạn bị phạt 1.000đ mỗi ngày. | Đội tuyển bóng đá thắng trận chung kết tối qua. | thấp | -0,077 | Đúng |
| 4 | Thư viện mở cửa lúc 7 giờ sáng. | Thư viện đóng cửa lúc 7 giờ tối. | cao | 0,610 | Đúng |
| 5 | Làm mất sách phải đền gấp 5 lần giá ghi trên bìa. | Cách nấu phở bò truyền thống ở Hà Nội. | thấp | -0,013 | Đúng |

(Điểm thực tế là cosine từ `paraphrase-multilingual-MiniLM-L12-v2`, cùng model dùng trong `bench.py`. Dự đoán được ghi trước khi tính; ngưỡng "cao" là từ khoảng 0,6 trở lên.)

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là cặp 2: hai câu khác đối tượng và khác con số ("sinh viên ... 10 ngày" so với "giảng viên ... 180 ngày") đạt 0,727, cao hơn cả cặp 1 là hai câu cùng nghĩa viết bằng từ khác (0,711); cặp 4 với nghĩa ngược nhau (mở cửa, đóng cửa) vẫn đạt 0,610. Điều này cho thấy embedding biểu diễn chủ đề nhiều hơn các chi tiết như đối tượng, con số hay chiều của nghĩa, và đó chính là lý do retrieval lẫn hạn mức của sinh viên với giảng viên ở câu 5 của benchmark, nên cần metadata filter `audience`.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Mượn sách về nhà ở thư viện HUIT được tối đa mấy tài liệu và trong bao nhiêu ngày? (có filter `audience=student`) | `huit-library-student#4`: mục "5. Quy định mượn/trả tài liệu (đối với sinh viên)": 3 tài liệu, 10 ngày | 0,767 | Có (hạng 1) | Chưa gọi LLM thật; ngữ cảnh top-1 đủ để trả lời "3 tài liệu, 10 ngày". |
| 2 | Quy trình mượn tài liệu về nhà ở thư viện Học viện Ngoại giao gồm những bước nào? | `dav-library-borrowing#10`: mục "6. Địa điểm mượn về" (phòng đọc chuyên ngành tầng 3) | 0,791 | Không ở hạng 1; chunk B1-B3 ở hạng 3 (0,745), chunk B7-B8 ở hạng 2 | Chưa gọi LLM thật; ngữ cảnh top-3 chỉ có một phần quy trình (B1-B3 và B7-B8, thiếu B4-B6). |
| 3 | Muốn được sử dụng thư viện HUIT thì cần những điều kiện gì? | `huit-library-rules#6`: mục "1. Quy định chung" (các điểm n, o) | 0,821 | Không (chunk đúng `huit-library-faq#23` ở hạng 23) | Chưa gọi LLM thật; ngữ cảnh top-3 không chứa ba điều kiện. |
| 4 | Trả sách trễ hạn ở thư viện HUIT bị phạt bao nhiêu tiền mỗi ngày? | `huit-library-faq#29`: câu 7 "Nếu trả sách trễ hạn phải nộp phạt như thế nào?" | 0,804 | Có (hạng 1) | Chưa gọi LLM thật; ngữ cảnh chứa "1.000đ/tài liệu/ngày" và "5.000đ/tài liệu/ngày". |
| 5 | Thư viện PTIT cho mượn về nhà tối đa mấy cuốn, thời hạn mượn là bao lâu? (có filter `audience=student`) | `ptit-library-student#2`: Điều 22 (phần dành cho học viên, sinh viên): tối đa 08 cuốn, 150 ngày | 0,832 | Có (hạng 1); không lọc thì hạng 1 là chunk của giảng viên (0,839) | Chưa gọi LLM thật; ngữ cảnh top-1 đủ để trả lời "08 cuốn, 01 học kì (150 ngày)". |

Chiến lược của tôi: `HeadingChunker` (chunk_size=500), embedding `paraphrase-multilingual-MiniLM-L12-v2`, chạy bằng `python bench.py --strategy heading` (kết quả đầy đủ, gồm cả ba chiến lược còn lại và lần chạy không filter, trong `ket_qua_benchmark.txt`). Điểm truy xuất theo `docs/SCORING.md` ở mức nội dung (chunk có chứa đáp án): câu 1 = 2, câu 2 = 1, câu 3 = 0, câu 4 = 2, câu 5 = 2, tổng 7/10; con số này giả định agent trả lời đúng khi ngữ cảnh chứa đáp án, vì chưa gọi LLM thật.

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Mình chưa có buổi demo với nhóm khác nên phần này dựa trên so sánh trong nhóm: chunk có đường dẫn tiêu đề (`heading` của mình và `sliding` của Khuê) thắng chunk không có tiêu đề (`semantic` của Phong) ở câu về DAV, dù `semantic` cắt theo nghĩa. Mình cũng học được rằng chấm bằng `doc_id` thổi phồng kết quả (`sentence` có gold doc ở 4/7 lượt nhưng chỉ 2/7 lượt có chunk chứa đáp án), nên phải kiểm chuỗi đáp án trong nội dung chunk.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 9 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 (42/42 test pass) |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 (5/5 dự đoán đúng) |
| Kết quả truy xuất của tôi (Competition Results) | 7 / 10 (theo chunk chứa đáp án; chưa gồm câu trả lời của agent vì chưa gọi LLM thật) |
| **Tổng phần cá nhân** | **56 / 60** |

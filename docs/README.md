# Tài liệu định hướng phát triển

Thư mục này gom toàn bộ tài liệu **định hướng/roadmap** (chưa phải code đã triển khai) của dự án vào một chỗ, tách biệt với `REPORT.md` ở gốc repo (ghi lại refactor kiến trúc đã làm xong). Mọi tài liệu ở đây đều thiết kế theo đúng nguyên tắc module hoá đã có (`core/` định nghĩa interface → `providers/`/`tools/` cài đặt cụ thể → `services/` dùng qua interface → `api/app.py` là composition root) — tài liệu mới không phá vỡ nguyên tắc đó, chỉ đề xuất thêm interface/provider mới theo cùng khuôn mẫu.

## Danh sách tài liệu

| Tài liệu | Nội dung |
|---|---|
| [01-accuracy-and-cross-validation.md](01-accuracy-and-cross-validation.md) | Định hướng kiến trúc: tăng độ chính xác tối đa, cross-validation đa nguồn (FAISS + web search), giảm hallucination — interface mới, luồng xử lý mới, lộ trình theo giai đoạn |
| [02-hallucination-mitigation-landscape.md](02-hallucination-mitigation-landscape.md) | Khảo sát các phương pháp/công cụ chống hallucination hiện có trên thị trường (RAG nói chung + đặc thù pháp lý), làm căn cứ cho tài liệu 01 |
| [03-cloud-deployment-aws-gcp.md](03-cloud-deployment-aws-gcp.md) | Định hướng triển khai lên AWS hoặc GCP: so sánh 2 nền tảng theo từng lớp hạ tầng, khuyến nghị lộ trình cụ thể |

## Cách đọc

Đọc theo thứ tự 02 → 01 → 03 nếu muốn hiểu từ "thị trường đang làm gì" trước rồi mới tới "dự án này nên làm gì"; hoặc đọc thẳng 01/03 nếu chỉ cần bản tóm tắt định hướng và quyết định.

Tất cả đề xuất trong 3 tài liệu đều ở dạng **roadmap — chưa implement**. Khi bắt tay triển khai mục nào, nên tách thành task/issue riêng và cập nhật lại tài liệu tương ứng (đánh dấu phần nào đã xong) thay vì để tài liệu định hướng lẫn với tài liệu mô tả hệ thống thật.

#!/usr/bin/env python3
"""
scripts/verify_change.py

Tầng 3: Rule-based Change Verification
Quy định rằng developer/agent phải chạy script này mỗi khi có sửa đổi code 
hoặc muốn fix bug. Nó quét qua Tầng 1 và Tầng 2 để nhận diện breaking changes, 
và kiểm tra đồng bộ docs.
"""

import subprocess
import sys
import os

def check_layer_1_tests():
    print("[1/3] 🧪 Đang chạy Tầng 1: Contract và Integration tests...")
    # Chạy pytest ở chế độ ngắn gọn cho tầng 1
    result = subprocess.run(["uv", "run", "pytest", "tests/contract", "tests/integration", "-v"], 
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ LỖI NGHIÊM TRỌNG: Contract/Integration tests FAILED.")
        print(">>> Vui lòng kiểm tra lại Data Models, Schemas và API Integration trước khi fix linh tinh.")
        print(result.stdout)
        return False
    print("✅ Tầng 1 passed: Các thành phần giao tiếp ổn định.")
    return True

def check_layer_2_snapshots():
    print("\n[2/3] 📸 Đang kiểm tra Tầng 2: Golden Snapshots...")
    result = subprocess.run(["uv", "run", "pytest", "tests/snapshot"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ PHÁT HIỆN THAY ĐỔI HÀNH VI: Snapshot mismatch!")
        print(">>> Sự thay đổi code của bạn làm lệch kết quả E2E so với Snapshot lưu từ trước.")
        print(">>> NẾU BẠN CHỦ ĐÍCH THAY ĐỔI: Chạy lệnh `pytest tests/snapshot --snapshot-update` để lưu chuẩn mới.")
        print(">>> NẾU KHÔNG CỐ Ý: Đây là bug hồi quy (Regression Bug). Hãy revert!")
        return False
    print("✅ Tầng 2 passed: Golden Snapshot khớp hoàn hảo với logic hiện tại.")
    return True

def check_layer_3_docs():
    print("\n[3/3] 📖 Xác thực Tầng 3: Documentation & Bug DB Sync...")
    has_error = False
    
    required_docs = ["OVERVIEW.md", "README.md", "TODO.md"]
    for doc in required_docs:
        if not os.path.exists(doc):
            print(f"⚠️ Cảnh báo: Thiếu file cấu trúc {doc}")
            has_error = True
            
    if has_error:
        print(">>> Rule yêu cầu: Hệ thống Agent khi gặp bug không được lao vào sửa ngay mà phải tra cứu / tạo lại Docs/Bug DB.")
        return False
        
    print("✅ Tầng 3 passed: Tài liệu chuẩn được bảo tồn đầy đủ.")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("TẦNG XÁC THỰC BẢO VỆ MÃ NGUỒN (CHANGE VERIFICATION GATE)")
    print("=" * 60)
    
    l1 = check_layer_1_tests()
    l2 = l1 and check_layer_2_snapshots()
    l3 = l2 and check_layer_3_docs()
    
    print("=" * 60)
    if l1 and l2 and l3:
        print("🎉 TẤT CẢ 3 TẦNG XÁC THỰC THÀNH CÔNG 🎉")
        print("Code Change này an toàn (Safe to patch/merge).")
        sys.exit(0)
    else:
        print("🚨 XÁC THỰC THẤT BẠI 🚨")
        print("Ngăn chặn commit/auto-fix. Yêu cầu sửa lỗi hoặc tham chiếu log.")
        sys.exit(1)

#!/usr/bin/env python3
"""
快速測試 - 不需要互動，直接顯示所有結果
"""

import sys
from pin_swapping import compute_output_bitvectors, parse_truth_table


print("=" * 70)
print("快速測試: Python 處理 N=8 的 256-bit 整數")
print("=" * 70)
print()

# 測試 1: 展示 256-bit 整數
print("【測試 1】 建立一個 256-bit 的整數")
print("-" * 70)
big_number = (1 << 255) | (1 << 128) | (1 << 64) | 1
print(f"數字大小: {big_number.bit_length()} bits")
print(f"十進位值: {big_number}")
print(f"記憶體使用: {sys.getsizeof(big_number)} bytes")
print()

# 測試 2: N=8 的實際 bitvector 計算
print("【測試 2】 計算 N=8 的實際 bitvector")
print("-" * 70)
print("邏輯函數: Z1 = A1 AND A2 AND ... AND A8")
print("(只有全部輸入為 1 時，輸出才是 1)")
print()

N, M = 8, 1
truth_table = []

for u in range(1 << N):  # 256 行
    z1 = 1 if u == 255 else 0  # 只有 u=11111111 時為 1
    truth_table.append(z1)

truth_table = parse_truth_table(N, M, truth_table)

input_perm = tuple(range(N))  # 恆等排列
bitvectors = compute_output_bitvectors(truth_table, N, M, input_perm)

print(f"真值表大小: {len(truth_table)} 行")
print(f"計算出的 Z1 bitvector:")
print(f"  二進位長度: {bitvectors[0].bit_length()} bits")
print(f"  十進位值: {bitvectors[0]}")
print()

# 驗證
print("驗證結果:")
expected = 1 << 255
print(f"  預期值 (2^255): {expected}")
print(f"  實際值:         {bitvectors[0]}")
print(f"  相等? {bitvectors[0] == expected} ✓")
print()

# 檢查哪些 bit 是 1
print("檢查哪些 bit 是 1:")
for t in range(256):
    if (bitvectors[0] >> t) & 1:
        print(f"  bit {t} = 1 ✓")
print()

print("=" * 70)
print("結論: Python 可以完美處理 N=8 的 256-bit bitvector！")
print("=" * 70)

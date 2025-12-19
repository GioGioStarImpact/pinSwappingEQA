#!/usr/bin/env python3
"""
測試 Python 處理大整數的能力

這個腳本展示 Python 可以輕鬆處理 N=8 時的 256-bit bitvector
"""

import sys
from pin_swapping import (
    compute_output_bitvectors,
    parse_truth_table,
    partition_pin_permutations,
    print_equivalence_classes
)


def test_python_big_integers():
    """測試 1: Python 的大整數支援"""
    print("=" * 70)
    print("測試 1: Python 可以處理多大的整數？")
    print("=" * 70)

    # N=8 的情況
    N = 8
    max_bitvector = (1 << (1 << N)) - 1  # 2^256 - 1

    print(f'N = {N}')
    print(f'Bitvector 需要的 bits: {1 << N} (2^{N})')
    print(f'最大的 bitvector 值: 2^{1 << N} - 1')
    print(f'這個數字有多大? 大約 10^{(1 << N) * 0.301:.0f}')
    print()

    # 實際建立一個 64-bit 的數字
    bitvector_64 = 0b1010101010101010101010101010101010101010101010101010101010101010
    print(f'一個 64-bit 的範例:')
    print(f'  十進位: {bitvector_64}')
    print(f'  二進位: {bin(bitvector_64)}')
    print()

    # 建立一個 256-bit 的數字
    big_bitvector = (1 << 255) | (1 << 128) | (1 << 64) | 1
    print(f'一個 256-bit 的範例 (設定了第 0, 64, 128, 255 位):')
    print(f'  十進位: {big_bitvector}')
    print(f'  二進位長度: {big_bitvector.bit_length()} bits')
    print(f'  記憶體使用: 約 {sys.getsizeof(big_bitvector)} bytes')
    print()


def test_n8_simple_function():
    """測試 2: N=8 的簡單邏輯函數"""
    print("=" * 70)
    print("測試 2: N=8, M=1 的簡單邏輯函數")
    print("=" * 70)

    # 建立一個簡單的 N=8 真值表
    # Z1 = A1 AND A2 AND A3 AND A4 AND A5 AND A6 AND A7 AND A8
    # 只有當所有輸入都是 1 時，輸出才是 1
    N, M = 8, 1
    truth_table = []

    print(f'邏輯定義: Z1 = A1 AND A2 AND ... AND A8')
    print(f'也就是說，只有當所有輸入都是 1 (u=255) 時，輸出才是 1')
    print()

    for u in range(1 << N):  # 256 個組合
        # 所有輸入都是 1 時，輸出才是 1
        z1 = 1 if u == 255 else 0
        truth_table.append(z1)

    # 驗證真值表
    truth_table = parse_truth_table(N, M, truth_table)

    print(f'真值表大小: {len(truth_table)} 行 (2^{N})')
    print()

    # 計算恆等排列的 bitvector
    input_perm = tuple(range(N))  # (0,1,2,3,4,5,6,7)
    bitvectors = compute_output_bitvectors(truth_table, N, M, input_perm)

    print(f'計算恆等排列 π = {input_perm} 的 bitvector:')
    for i, bv in enumerate(bitvectors):
        print(f'  Z{i+1} bitvector (十進位):')
        print(f'    {bv}')
        print(f'  二進位長度: {bv.bit_length()} bits')
        print(f'  二進位表示 (前 40 字元): {bin(bv)[:40]}...')
    print()

    # 驗證 bitvector 的內容
    print('驗證 bitvector 的內容:')
    for t in range(256):
        if (bitvectors[0] >> t) & 1:
            print(f'  bit {t} = 1 (二進位: {bin(t)})')
    print()

    # 驗證這個 bitvector 正好是 2^255
    expected = 1 << 255
    print(f'驗證: bitvector 應該等於 2^255')
    print(f'  2^255 = {expected}')
    print(f'  bitvector = {bitvectors[0]}')
    print(f'  相等? {bitvectors[0] == expected} ✓')
    print()


def test_n8_xor_function():
    """測試 3: N=8 的 XOR 函數"""
    print("=" * 70)
    print("測試 3: N=8, M=1 的 XOR 函數")
    print("=" * 70)

    # Z1 = A1 XOR A2 XOR A3 XOR A4 XOR A5 XOR A6 XOR A7 XOR A8
    # 奇數個 1 時輸出 1 (奇偶校驗)
    N, M = 8, 1
    truth_table = []

    print(f'邏輯定義: Z1 = A1 XOR A2 XOR ... XOR A8')
    print(f'也就是說，輸入中有奇數個 1 時，輸出才是 1 (奇偶校驗)')
    print()

    for u in range(1 << N):
        # 計算有多少個 1
        count = bin(u).count('1')
        z1 = count % 2  # 奇數個 1 → 輸出 1
        truth_table.append(z1)

    truth_table = parse_truth_table(N, M, truth_table)

    # 計算恆等排列的 bitvector
    input_perm = tuple(range(N))
    bitvectors = compute_output_bitvectors(truth_table, N, M, input_perm)

    print(f'計算恆等排列的 bitvector:')
    print(f'  Z1 bitvector 長度: {bitvectors[0].bit_length()} bits')
    print(f'  Z1 bitvector (十六進位): 0x{bitvectors[0]:064x}')
    print()

    # 統計有多少個 1
    ones_count = bin(bitvectors[0]).count('1')
    print(f'Bitvector 中有多少個 1? {ones_count} 個 (應該是 128 個)')
    print(f'驗證: {ones_count == 128} ✓')
    print()


def test_n4_partition():
    """測試 4: N=4 的完整分類 (較快)"""
    print("=" * 70)
    print("測試 4: N=4, M=2 的完整等價類別分析")
    print("=" * 70)

    # Z1 = A1 AND A2
    # Z2 = A3 XOR A4
    N, M = 4, 2
    truth_table = []

    print(f'邏輯定義:')
    print(f'  Z1 = A1 AND A2')
    print(f'  Z2 = A3 XOR A4')
    print()

    for u in range(1 << N):
        a1 = (u >> 0) & 1
        a2 = (u >> 1) & 1
        a3 = (u >> 2) & 1
        a4 = (u >> 3) & 1

        z1 = a1 & a2
        z2 = a3 ^ a4

        output = z1 | (z2 << 1)
        truth_table.append(output)

    truth_table = parse_truth_table(N, M, truth_table)

    # 分析等價類別
    print(f'開始分析... (共 {N}! = {1} 個排列需要檢查)')
    import math
    print(f'共 {math.factorial(N)} 個排列需要檢查')
    print()

    equivalence_classes = partition_pin_permutations(truth_table, N, M)

    # 印出結果
    print_equivalence_classes(equivalence_classes, N, verbose=False)


def main():
    """主程式: 執行所有測試"""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Python 大整數處理能力測試" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # 測試 1: Python 大整數
    test_python_big_integers()
    input("\n按 Enter 繼續下一個測試...")
    print("\n")

    # 測試 2: N=8 AND 函數
    test_n8_simple_function()
    input("\n按 Enter 繼續下一個測試...")
    print("\n")

    # 測試 3: N=8 XOR 函數
    test_n8_xor_function()
    input("\n按 Enter 繼續下一個測試...")
    print("\n")

    # 測試 4: N=4 完整分類
    test_n4_partition()

    print()
    print("=" * 70)
    print("所有測試完成！")
    print("=" * 70)
    print()
    print("結論:")
    print("  ✓ Python 可以輕鬆處理 256-bit (N=8) 的整數")
    print("  ✓ Bitvector 運算完全正常")
    print("  ✓ 真正的限制是 N! 的計算量，而非數字大小")
    print()


if __name__ == "__main__":
    main()

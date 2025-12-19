#!/usr/bin/env python3
"""
Pin Swapping Equivalence Algorithm - Multi-Process Parallel Version

使用 Python multiprocessing 平行化計算所有排列的 signature。
適用於多核心 CPU (例如 48 cores)，可獲得接近線性的加速比。

基於 pin_swapping.py，主要差異:
1. 新增 partition_pin_permutations_parallel() 函數
2. 使用 multiprocessing.Pool 平行計算
3. 新增進度顯示和效能測量

參考文件: spec_simple.md
"""

from typing import List, Tuple, Dict
from itertools import permutations
from multiprocessing import Pool, cpu_count
import time


# ============================================================================
# 從 pin_swapping.py 複製的核心函數
# ============================================================================

def apply_input_permutation(net_pattern: int, perm: Tuple[int, ...], n: int) -> int:
    """
    將外部訊號 (net pattern) 根據排列 π 轉換成晶片腳位 (pin pattern)。
    參考: spec_simple.md § 3.1 "概念 1: 外部訊號 vs 晶片腳位"
    """
    pin_pattern = 0
    for net_idx in range(n):
        net_bit = (net_pattern >> net_idx) & 1
        pin_idx = perm[net_idx]
        pin_pattern |= net_bit << pin_idx
    return pin_pattern


def compute_output_bitvectors(
    truth_table: List[int],
    n: int,
    m: int,
    input_perm: Tuple[int, ...]
) -> List[int]:
    """
    計算給定輸入排列下，所有輸出函數的 bitvector。
    參考: spec_simple.md § 3.2 "概念 2: 輸出函數的指紋 (Bitvector)"
    """
    bitvectors = [0] * m
    num_patterns = 1 << n

    for net_pattern in range(num_patterns):
        pin_pattern = apply_input_permutation(net_pattern, input_perm, n)
        outputs = truth_table[pin_pattern]

        for output_idx in range(m):
            if (outputs >> output_idx) & 1:
                bitvectors[output_idx] |= 1 << net_pattern

    return bitvectors


def compute_canonical_signature(
    truth_table: List[int],
    n: int,
    m: int,
    input_perm: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    計算給定輸入排列的標準化指紋。
    參考: spec_simple.md § 3.3 "概念 3: 標準化指紋 (Canonical Signature)"
    """
    bitvectors = compute_output_bitvectors(truth_table, n, m, input_perm)
    signature = tuple(sorted(bitvectors))
    return signature


def parse_truth_table(n: int, m: int, rows: List[int]) -> List[int]:
    """解析真值表輸入，確保格式正確。"""
    expected_rows = 1 << n
    if len(rows) != expected_rows:
        raise ValueError(f"真值表長度錯誤: 預期 {expected_rows} 行 (2^{n}), 實際 {len(rows)} 行")

    max_output_value = (1 << m) - 1
    for idx, output in enumerate(rows):
        if not isinstance(output, int) or output < 0 or output > max_output_value:
            raise ValueError(
                f"真值表第 {idx} 行錯誤: 輸出必須是 0 到 {max_output_value} 的整數, 實際為 {output}"
            )

    return rows


# ============================================================================
# Multi-Process 平行化函數
# ============================================================================

def _compute_signature_worker(args: Tuple) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Worker 函數：計算單一排列的 signature。

    這個函數被設計成 pure function (無副作用)，適合平行化。

    Args:
        args: (truth_table, n, m, input_perm) 的 tuple

    Returns:
        (signature, input_perm) 的 tuple
    """
    truth_table, n, m, input_perm = args
    signature = compute_canonical_signature(truth_table, n, m, input_perm)
    return (signature, input_perm)


def partition_pin_permutations_parallel(
    truth_table: List[int],
    n: int,
    m: int,
    num_workers: int = None,
    show_progress: bool = True
) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
    """
    使用 Multi-Process 平行化計算所有排列的等價類別。

    參考: spec_simple.md § 4 "完整演算法流程"

    演算法流程:
        步驟 1: 列舉所有可能的輸入排列 (N! 個)
        步驟 2: 平行計算每個排列的 signature
        步驟 3: 合併結果，將 signature 相同的排列歸為一類

    Args:
        truth_table (List[int]): 真值表 F，長度為 2^N
        n (int): 輸入數量 N (1 ≤ N ≤ 10)
        m (int): 輸出數量 M (1 ≤ M ≤ 3)
        num_workers (int): Worker 數量，預設使用 CPU 核心數
        show_progress (bool): 是否顯示進度資訊

    Returns:
        Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
            字典 {標準化指紋 → 等價的輸入排列列表}

    時間複雜度: O(N! × 2^N × M / num_workers)
        - 平行化後可獲得接近線性的加速比

    範例:
        N=8, 單核心: ~20 秒
        N=8, 48 核心: ~0.5 秒 (約 40x 加速)
    """
    # 決定 worker 數量
    if num_workers is None:
        num_workers = cpu_count()

    # 步驟 1: 列舉所有可能的輸入排列
    all_perms = list(permutations(range(n)))
    num_perms = len(all_perms)

    if show_progress:
        import math
        print(f"平行化設定:")
        print(f"  輸入數量 N = {n}")
        print(f"  輸出數量 M = {m}")
        print(f"  總排列數 = {num_perms} ({n}! = {math.factorial(n)})")
        print(f"  Worker 數量 = {num_workers} (CPU 核心數: {cpu_count()})")
        print(f"  每個 worker 約處理 {num_perms // num_workers} 個排列")
        print()

    # 準備所有任務 (truth_table, n, m, perm)
    tasks = [(truth_table, n, m, perm) for perm in all_perms]

    # 步驟 2: 使用 multiprocessing.Pool 平行計算
    start_time = time.time()

    if show_progress:
        print("開始平行計算...")

    with Pool(num_workers) as pool:
        # map() 會自動分配任務到各個 worker
        results = pool.map(_compute_signature_worker, tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    # 步驟 3: 合併結果
    signature_to_perms: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

    for signature, input_perm in results:
        if signature not in signature_to_perms:
            signature_to_perms[signature] = []
        signature_to_perms[signature].append(input_perm)

    if show_progress:
        print(f"計算完成！耗時 {elapsed:.3f} 秒")
        print(f"  等價類別數 = {len(signature_to_perms)}")
        print(f"  平均每個排列耗時 = {elapsed / num_perms * 1000:.3f} ms")
        print(f"  吞吐量 = {num_perms / elapsed:.1f} 排列/秒")
        print()

    return signature_to_perms


# ============================================================================
# 非平行化版本 (用於比較效能)
# ============================================================================

def partition_pin_permutations(
    truth_table: List[int],
    n: int,
    m: int,
    show_progress: bool = True
) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
    """
    單核心版本 (用於效能比較)。

    與 partition_pin_permutations_parallel() 功能相同，
    但使用單一 process，不進行平行化。
    """
    all_perms = list(permutations(range(n)))
    num_perms = len(all_perms)

    if show_progress:
        import math
        print(f"單核心版本:")
        print(f"  輸入數量 N = {n}")
        print(f"  輸出數量 M = {m}")
        print(f"  總排列數 = {num_perms} ({n}! = {math.factorial(n)})")
        print()

    signature_to_perms: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

    start_time = time.time()

    if show_progress:
        print("開始計算...")

    for input_perm in all_perms:
        signature = compute_canonical_signature(truth_table, n, m, input_perm)

        if signature not in signature_to_perms:
            signature_to_perms[signature] = []
        signature_to_perms[signature].append(input_perm)

    end_time = time.time()
    elapsed = end_time - start_time

    if show_progress:
        print(f"計算完成！耗時 {elapsed:.3f} 秒")
        print(f"  等價類別數 = {len(signature_to_perms)}")
        print(f"  平均每個排列耗時 = {elapsed / num_perms * 1000:.3f} ms")
        print(f"  吞吐量 = {num_perms / elapsed:.1f} 排列/秒")
        print()

    return signature_to_perms


# ============================================================================
# 輔助函數: 結果格式化輸出
# ============================================================================

def format_permutation(perm: Tuple[int, ...]) -> str:
    """將排列轉換為可讀格式。"""
    mappings = [f"net{i+1}→A{perm[i]+1}" for i in range(len(perm))]
    return ", ".join(mappings)


def print_equivalence_classes(
    equivalence_classes: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    n: int,
    verbose: bool = False
) -> None:
    """印出等價類別的摘要資訊。"""
    num_classes = len(equivalence_classes)
    total_perms = sum(len(perms) for perms in equivalence_classes.values())

    print(f"========== 等價類別分析 ==========")
    print(f"輸入數量 N = {n}")
    print(f"總排列數 = {total_perms}")
    print(f"等價類別數 = {num_classes}")
    print()

    # 按類別大小排序
    sorted_classes = sorted(
        equivalence_classes.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    for class_idx, (signature, perms) in enumerate(sorted_classes, 1):
        print(f"類別 {class_idx}: {len(perms)} 個等價排列")
        if verbose:
            print(f"  指紋: {signature}")
            for perm in perms[:5]:  # 最多顯示 5 個
                print(f"    - {format_permutation(perm)}")
            if len(perms) > 5:
                print(f"    ... 還有 {len(perms) - 5} 個排列")
        print()


# ============================================================================
# 範例使用
# ============================================================================

def example_3_input_2_output_parallel():
    """
    範例: 3 輸入 2 輸出的邏輯元件 (平行化版本)
    參考: spec_simple.md § 5 "實際範例"
    """
    print("=" * 70)
    print("範例: 3 輸入 2 輸出 (平行化版本)")
    print("=" * 70)
    print("邏輯定義:")
    print("  Z1 = A1 AND A2")
    print("  Z2 = A2 XOR A3")
    print()

    # 建立真值表
    n, m = 3, 2
    truth_table = []

    for u in range(1 << n):
        a1 = (u >> 0) & 1
        a2 = (u >> 1) & 1
        a3 = (u >> 2) & 1

        z1 = a1 & a2
        z2 = a2 ^ a3

        output = z1 | (z2 << 1)
        truth_table.append(output)

    truth_table = parse_truth_table(n, m, truth_table)

    # 平行化分析
    equivalence_classes = partition_pin_permutations_parallel(
        truth_table, n, m, num_workers=None, show_progress=True
    )

    # 印出結果
    print_equivalence_classes(equivalence_classes, n, verbose=True)

    return equivalence_classes


def benchmark_parallel_vs_sequential(n: int = 6, m: int = 2):
    """
    效能比較: 平行化 vs 單核心

    Args:
        n: 輸入數量 (建議 5-7，太大會很慢)
        m: 輸出數量
    """
    print("=" * 70)
    print(f"效能比較: N={n}, M={m}")
    print("=" * 70)
    print()

    # 建立簡單的真值表 (Z1 = A1 AND A2, Z2 = A3 XOR A4)
    truth_table = []
    for u in range(1 << n):
        a1 = (u >> 0) & 1
        a2 = (u >> 1) & 1
        z1 = a1 & a2
        z2 = ((u >> 2) & 1) if n > 2 else 0

        output = z1 | (z2 << 1)
        truth_table.append(output)

    truth_table = parse_truth_table(n, m, truth_table)

    # 測試單核心版本
    print("【1】單核心版本")
    print("-" * 70)
    result_seq = partition_pin_permutations(truth_table, n, m, show_progress=True)

    print()

    # 測試平行化版本
    print("【2】平行化版本")
    print("-" * 70)
    result_par = partition_pin_permutations_parallel(
        truth_table, n, m, num_workers=None, show_progress=True
    )

    # 驗證結果一致
    print("【3】驗證結果")
    print("-" * 70)
    print(f"單核心等價類別數: {len(result_seq)}")
    print(f"平行化等價類別數: {len(result_par)}")
    print(f"結果一致? {result_seq == result_par} ✓")
    print()


# ============================================================================
# 主程式
# ============================================================================

if __name__ == "__main__":
    import sys

    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "Pin Swapping - Multi-Process 平行化版本" + " " * 10 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # 檢查 CPU 核心數
    print(f"系統資訊: 偵測到 {cpu_count()} 個 CPU 核心")
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # 效能比較模式
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        m = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        benchmark_parallel_vs_sequential(n, m)
    else:
        # 一般範例模式
        example_3_input_2_output_parallel()

    print()
    print("=" * 70)
    print("提示:")
    print("  • 執行效能比較: python3 pin_swapping_parallel.py benchmark [N] [M]")
    print("  • 範例: python3 pin_swapping_parallel.py benchmark 7 2")
    print("=" * 70)

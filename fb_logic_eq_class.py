#!/usr/bin/env python3
"""
Frontside/Backside Logic Equivalence Classification
前側/後側邏輯等價分類

這個程式探索在 frontside/backside 限制下，所有可能的功能等價配置。
This program explores all possible functionally equivalent configurations under frontside/backside constraints.

主要概念 Main Concepts:
- Input nets 可能來自 frontside 或 backside
  Input nets may come from frontside or backside
- Input pins 分成 frontside pins 和 backside pins
  Input pins are divided into frontside pins and backside pins
- Frontside nets 只能連接到 frontside pins
  Frontside nets can only connect to frontside pins
- Backside nets 只能連接到 backside pins
  Backside nets can only connect to backside pins

複雜度 Complexity:
- 總排列數 Total permutations ≈ k! × 2^k
- k=4: 384, k=6: 46,080, k=8: 10,321,920
"""

from typing import List, Tuple, Dict, Set
from itertools import permutations, combinations
from collections import defaultdict
import multiprocessing as mp
import time
import sys


# ============================================================================
# Section 1: 真值表解析 Truth Table Parsing
# ============================================================================

def parse_truth_table(n: int, m: int, table: List[int]) -> List[int]:
    """
    驗證並解析真值表。
    Validate and parse the truth table.

    參數 Parameters:
    - n: 輸入數量 number of inputs
    - m: 輸出數量 number of outputs
    - table: 真值表 (長度必須是 2^n) truth table (length must be 2^n)

    回傳 Returns:
    - 驗證過的真值表 validated truth table
    """
    expected_rows = 1 << n  # 2^n

    if len(table) != expected_rows:
        raise ValueError(
            f"真值表長度錯誤: 預期 {expected_rows} 行，實際 {len(table)} 行\n"
            f"Truth table length error: expected {expected_rows} rows, got {len(table)} rows"
        )

    max_output_value = (1 << m) - 1  # 2^m - 1
    for i, output in enumerate(table):
        if not (0 <= output <= max_output_value):
            raise ValueError(
                f"真值表第 {i} 行的輸出值 {output} 超出範圍 [0, {max_output_value}]\n"
                f"Truth table row {i} has output value {output} out of range [0, {max_output_value}]"
            )

    return table


# ============================================================================
# Section 2: 輸入排列與 Bitvector 計算
#            Input Permutation and Bitvector Computation
# ============================================================================

def apply_input_permutation(net_pattern: int, perm: Tuple[int, ...], n: int) -> int:
    """
    將輸入 net pattern 根據排列 π 重新排列。
    Rearrange input net pattern according to permutation π.

    參數 Parameters:
    - net_pattern: 輸入的 bit pattern (0 到 2^n - 1)
                   input bit pattern (0 to 2^n - 1)
    - perm: 排列 π，perm[i] 表示第 i 個 pin 接到第 perm[i] 個 net
            permutation π, perm[i] means pin i connects to net perm[i]
    - n: 輸入數量 number of inputs

    回傳 Returns:
    - 重新排列後的 bit pattern rearranged bit pattern
    """
    result = 0
    for pin_idx in range(n):
        net_idx = perm[pin_idx]
        bit_value = (net_pattern >> net_idx) & 1
        result |= (bit_value << pin_idx)
    return result


def compute_output_bitvectors(
    truth_table: List[int],
    n: int,
    m: int,
    input_perm: Tuple[int, ...]
) -> List[int]:
    """
    計算在給定輸入排列下，每個輸出函數的 bitvector。
    Compute bitvector for each output function under given input permutation.

    參數 Parameters:
    - truth_table: 真值表 truth table
    - n: 輸入數量 number of inputs
    - m: 輸出數量 number of outputs
    - input_perm: 輸入排列 input permutation

    回傳 Returns:
    - bitvectors: 長度為 m 的列表，每個元素是一個 2^n-bit 的整數
                  list of length m, each element is a 2^n-bit integer
    """
    bitvectors = [0] * m

    for net_pattern in range(1 << n):  # 遍歷所有可能的輸入組合
        pin_pattern = apply_input_permutation(net_pattern, input_perm, n)
        output = truth_table[pin_pattern]

        for output_idx in range(m):
            bit = (output >> output_idx) & 1
            if bit:
                bitvectors[output_idx] |= (1 << net_pattern)

    return bitvectors


def compute_canonical_signature(
    truth_table: List[int],
    n: int,
    m: int,
    input_perm: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    計算給定輸入排列的標準化 signature（將輸出 bitvectors 排序）。
    Compute canonical signature for given input permutation (sort output bitvectors).

    參數 Parameters:
    - truth_table: 真值表 truth table
    - n: 輸入數量 number of inputs
    - m: 輸出數量 number of outputs
    - input_perm: 輸入排列 input permutation

    回傳 Returns:
    - signature: 排序後的 bitvectors tuple
                 sorted bitvectors tuple
    """
    bitvectors = compute_output_bitvectors(truth_table, n, m, input_perm)
    return tuple(sorted(bitvectors))


# ============================================================================
# Section 3: Frontside/Backside 配置枚舉
#            Frontside/Backside Configuration Enumeration
# ============================================================================

def enumerate_net_classifications(k: int, p: int) -> List[Tuple[int, ...]]:
    """
    枚舉所有可能的 net 分類方式（哪些是 frontside）。
    Enumerate all possible net classifications (which are frontside).

    參數 Parameters:
    - k: 總 net 數量 total number of nets
    - p: frontside net 數量 number of frontside nets

    回傳 Returns:
    - 所有可能的分類，每個分類是一個 tuple，內含 frontside net 的索引
      all possible classifications, each is a tuple containing indices of frontside nets
    """
    return list(combinations(range(k), p))


def enumerate_pin_classifications(k: int, p: int) -> List[Tuple[int, ...]]:
    """
    枚舉所有可能的 pin 分類方式（哪些是 frontside）。
    Enumerate all possible pin classifications (which are frontside).

    參數 Parameters:
    - k: 總 pin 數量 total number of pins
    - p: frontside pin 數量 number of frontside pins

    回傳 Returns:
    - 所有可能的分類，每個分類是一個 tuple，內含 frontside pin 的索引
      all possible classifications, each is a tuple containing indices of frontside pins
    """
    return list(combinations(range(k), p))


def generate_constrained_permutations(
    net_classification: Tuple[int, ...],
    pin_classification: Tuple[int, ...],
    k: int
) -> List[Tuple[int, ...]]:
    """
    生成符合 frontside/backside 限制的所有排列。
    Generate all permutations that satisfy frontside/backside constraints.

    參數 Parameters:
    - net_classification: frontside nets 的索引 indices of frontside nets
    - pin_classification: frontside pins 的索引 indices of frontside pins
    - k: 總數量 total count

    回傳 Returns:
    - 所有符合限制的排列 all permutations satisfying constraints
    """
    frontside_nets = set(net_classification)
    frontside_pins = set(pin_classification)
    backside_nets = set(range(k)) - frontside_nets
    backside_pins = set(range(k)) - frontside_pins

    # 轉換成 sorted lists 以保證順序一致性
    # Convert to sorted lists to ensure order consistency
    fs_nets = sorted(frontside_nets)
    fs_pins = sorted(frontside_pins)
    bs_nets = sorted(backside_nets)
    bs_pins = sorted(backside_pins)

    p = len(fs_nets)
    q = len(bs_nets)

    # 枚舉 frontside nets 的所有排列
    # Enumerate all permutations of frontside nets
    fs_perms = list(permutations(fs_nets))

    # 枚舉 backside nets 的所有排列
    # Enumerate all permutations of backside nets
    bs_perms = list(permutations(bs_nets))

    result = []

    for fs_perm in fs_perms:
        for bs_perm in bs_perms:
            # 構建完整排列: perm[pin_idx] = net_idx
            # Build complete permutation: perm[pin_idx] = net_idx
            perm = [0] * k

            # Frontside: frontside pins 連接到 frontside nets
            # Frontside: frontside pins connect to frontside nets
            for i, pin_idx in enumerate(fs_pins):
                perm[pin_idx] = fs_perm[i]

            # Backside: backside pins 連接到 backside nets
            # Backside: backside pins connect to backside nets
            for i, pin_idx in enumerate(bs_pins):
                perm[pin_idx] = bs_perm[i]

            result.append(tuple(perm))

    return result


# ============================================================================
# Section 4: 等價類別分析 Equivalence Class Analysis
# ============================================================================

def partition_with_fb_constraints(
    truth_table: List[int],
    n: int,
    m: int,
    verbose: bool = True
) -> Dict[Tuple[int, ...], List[Tuple]]:
    """
    在 frontside/backside 限制下，分析所有等價類別。
    Analyze all equivalence classes under frontside/backside constraints.

    參數 Parameters:
    - truth_table: 真值表 truth table
    - n: 輸入數量 number of inputs
    - m: 輸出數量 number of outputs
    - verbose: 是否顯示進度 whether to show progress

    回傳 Returns:
    - 等價類別字典，key 是 signature，value 是 (perm, net_class, pin_class) 的列表
      equivalence class dict, key is signature, value is list of (perm, net_class, pin_class)
    """
    equivalence_classes = defaultdict(list)
    total_count = 0

    # 枚舉所有可能的 (p, q) 組合
    # Enumerate all possible (p, q) combinations
    for p in range(n + 1):
        q = n - p

        if verbose:
            print(f"\n處理 p={p}, q={q} 的配置...")
            print(f"Processing p={p}, q={q} configuration...")

        # 枚舉所有 net 分類方式
        # Enumerate all net classifications
        net_classifications = enumerate_net_classifications(n, p)

        # 枚舉所有 pin 分類方式
        # Enumerate all pin classifications
        pin_classifications = enumerate_pin_classifications(n, p)

        config_count = 0

        for net_class in net_classifications:
            for pin_class in pin_classifications:
                # 生成符合此配置的所有排列
                # Generate all permutations for this configuration
                perms = generate_constrained_permutations(net_class, pin_class, n)

                for perm in perms:
                    sig = compute_canonical_signature(truth_table, n, m, perm)
                    equivalence_classes[sig].append((perm, net_class, pin_class))
                    config_count += 1

        total_count += config_count

        if verbose:
            print(f"  此配置的排列數: {config_count}")
            print(f"  Number of permutations for this configuration: {config_count}")

    if verbose:
        print(f"\n總計 {total_count} 個排列")
        print(f"Total {total_count} permutations")
        print(f"等價類別數量: {len(equivalence_classes)}")
        print(f"Number of equivalence classes: {len(equivalence_classes)}")

    return dict(equivalence_classes)


# ============================================================================
# Section 5: 平行化版本 Parallel Version
# ============================================================================

def _compute_signature_worker_fb(args: Tuple) -> Tuple[Tuple[int, ...], Tuple]:
    """
    Worker 函數：計算單一 (perm, net_class, pin_class) 的 signature。
    Worker function: compute signature for a single (perm, net_class, pin_class).

    參數 Parameters:
    - args: (truth_table, n, m, perm, net_class, pin_class)

    回傳 Returns:
    - (signature, (perm, net_class, pin_class))
    """
    truth_table, n, m, perm, net_class, pin_class = args
    sig = compute_canonical_signature(truth_table, n, m, perm)
    return (sig, (perm, net_class, pin_class))


def partition_with_fb_constraints_parallel(
    truth_table: List[int],
    n: int,
    m: int,
    num_processes: int = None,
    verbose: bool = True
) -> Dict[Tuple[int, ...], List[Tuple]]:
    """
    平行化版本：在 frontside/backside 限制下，分析所有等價類別。
    Parallel version: analyze all equivalence classes under frontside/backside constraints.

    參數 Parameters:
    - truth_table: 真值表 truth table
    - n: 輸入數量 number of inputs
    - m: 輸出數量 number of outputs
    - num_processes: 並行進程數（None 表示使用 CPU 核心數）
                     number of parallel processes (None means use CPU core count)
    - verbose: 是否顯示進度 whether to show progress

    回傳 Returns:
    - 等價類別字典 equivalence class dictionary
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    if verbose:
        print(f"使用 {num_processes} 個並行進程")
        print(f"Using {num_processes} parallel processes")

    # 生成所有任務
    # Generate all tasks
    tasks = []

    for p in range(n + 1):
        q = n - p
        net_classifications = enumerate_net_classifications(n, p)
        pin_classifications = enumerate_pin_classifications(n, p)

        for net_class in net_classifications:
            for pin_class in pin_classifications:
                perms = generate_constrained_permutations(net_class, pin_class, n)

                for perm in perms:
                    tasks.append((truth_table, n, m, perm, net_class, pin_class))

    if verbose:
        print(f"總計 {len(tasks)} 個任務")
        print(f"Total {len(tasks)} tasks")

    # 使用 multiprocessing.Pool 平行計算
    # Use multiprocessing.Pool for parallel computation
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(_compute_signature_worker_fb, tasks)

    # 將結果組織成等價類別
    # Organize results into equivalence classes
    equivalence_classes = defaultdict(list)
    for sig, config in results:
        equivalence_classes[sig].append(config)

    if verbose:
        print(f"等價類別數量: {len(equivalence_classes)}")
        print(f"Number of equivalence classes: {len(equivalence_classes)}")

    return dict(equivalence_classes)


# ============================================================================
# Section 6: 結果輸出 Result Output
# ============================================================================

def print_fb_equivalence_classes(
    equivalence_classes: Dict[Tuple[int, ...], List[Tuple]],
    n: int,
    verbose: bool = False
):
    """
    印出等價類別的統計資訊和詳細內容。
    Print statistics and details of equivalence classes.

    參數 Parameters:
    - equivalence_classes: 等價類別字典 equivalence class dictionary
    - n: 輸入數量 number of inputs
    - verbose: 是否顯示詳細資訊 whether to show detailed information
    """
    print("\n" + "=" * 70)
    print("等價類別分析結果 Equivalence Class Analysis Results")
    print("=" * 70)

    print(f"\n總等價類別數: {len(equivalence_classes)}")
    print(f"Total number of equivalence classes: {len(equivalence_classes)}")

    # 統計每個類別的大小
    # Statistics of each class size
    class_sizes = [len(configs) for configs in equivalence_classes.values()]
    print(f"\n最小類別大小: {min(class_sizes)}")
    print(f"Minimum class size: {min(class_sizes)}")
    print(f"最大類別大小: {max(class_sizes)}")
    print(f"Maximum class size: {max(class_sizes)}")
    print(f"平均類別大小: {sum(class_sizes) / len(class_sizes):.2f}")
    print(f"Average class size: {sum(class_sizes) / len(class_sizes):.2f}")

    if verbose:
        print("\n" + "=" * 70)
        print("詳細內容 Detailed Contents")
        print("=" * 70)

        for idx, (sig, configs) in enumerate(equivalence_classes.items(), 1):
            print(f"\n類別 {idx} / Class {idx} (大小 size: {len(configs)})")
            print(f"  Signature: {sig}")

            # 顯示前 5 個配置
            # Show first 5 configurations
            for i, (perm, net_class, pin_class) in enumerate(configs[:5], 1):
                print(f"    配置 {i}: perm={perm}")
                print(f"             frontside nets: {net_class}")
                print(f"             frontside pins: {pin_class}")

            if len(configs) > 5:
                print(f"    ... (還有 {len(configs) - 5} 個配置)")
                print(f"    ... ({len(configs) - 5} more configurations)")


# ============================================================================
# Section 7: 主程式與範例 Main Program and Examples
# ============================================================================

def example_4input_and_gate():
    """
    範例 1: 4 輸入 1 輸出的 AND 閘
    Example 1: 4-input 1-output AND gate

    Z1 = A1 AND A2 AND A3 AND A4
    """
    print("\n" + "=" * 70)
    print("範例 1: 4 輸入 AND 閘")
    print("Example 1: 4-input AND gate")
    print("=" * 70)
    print("邏輯定義 Logic definition: Z1 = A1 AND A2 AND A3 AND A4")

    N, M = 4, 1
    truth_table = []

    for u in range(1 << N):  # 16 行
        # 只有全部為 1 時才輸出 1
        # Output 1 only when all inputs are 1
        z1 = 1 if u == 15 else 0
        truth_table.append(z1)

    truth_table = parse_truth_table(N, M, truth_table)

    print(f"\n使用單核版本...")
    print(f"Using single-core version...")
    start_time = time.time()
    eq_classes = partition_with_fb_constraints(truth_table, N, M, verbose=True)
    elapsed = time.time() - start_time
    print(f"執行時間: {elapsed:.4f} 秒")
    print(f"Execution time: {elapsed:.4f} seconds")

    print_fb_equivalence_classes(eq_classes, N, verbose=False)


def example_4input_asymmetric_gate():
    """
    範例 2: 4 輸入 2 輸出的非對稱閘
    Example 2: 4-input 2-output asymmetric gate

    Z1 = A1 AND A2
    Z2 = A3 XOR A4
    """
    print("\n" + "=" * 70)
    print("範例 2: 4 輸入非對稱閘")
    print("Example 2: 4-input asymmetric gate")
    print("=" * 70)
    print("邏輯定義 Logic definition:")
    print("  Z1 = A1 AND A2")
    print("  Z2 = A3 XOR A4")

    N, M = 4, 2
    truth_table = []

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

    print(f"\n使用平行版本...")
    print(f"Using parallel version...")
    start_time = time.time()
    eq_classes = partition_with_fb_constraints_parallel(
        truth_table, N, M, num_processes=8, verbose=True
    )
    elapsed = time.time() - start_time
    print(f"執行時間: {elapsed:.4f} 秒")
    print(f"Execution time: {elapsed:.4f} seconds")

    print_fb_equivalence_classes(eq_classes, N, verbose=True)


def main():
    """
    主程式：執行範例測試
    Main program: run example tests
    """
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "Frontside/Backside Logic Equivalence" + " " * 10 + "║")
    print("║" + " " * 14 + "前側/後側邏輯等價分類" + " " * 14 + "║")
    print("╚" + "=" * 68 + "╝")

    # 範例 1: 4 輸入 AND 閘
    example_4input_and_gate()

    # 範例 2: 4 輸入非對稱閘
    example_4input_asymmetric_gate()

    print("\n" + "=" * 70)
    print("測試完成！")
    print("Testing completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

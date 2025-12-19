#!/usr/bin/env python3
"""
Pin Swapping Equivalence Algorithm - Optimized Version

給定一個邏輯元件的真值表，找出所有功能上等價的腳位排列組合。
限制: N ≤ 8 輸入, M ≤ 3 輸出

參考文件: spec_simple.md
"""

from typing import List, Tuple, Dict
from itertools import permutations


# ============================================================================
# 概念 1: 外部訊號 vs 晶片腳位 (spec_simple.md § 3.1)
# ============================================================================

def apply_input_permutation(net_pattern: int, perm: Tuple[int, ...], n: int) -> int:
    """
    將外部訊號 (net pattern) 根據排列 π 轉換成晶片腳位 (pin pattern)。

    參考: spec_simple.md § 3.1 "概念 1: 外部訊號 vs 晶片腳位"

    Args:
        net_pattern (int): 外部訊號組合 t (LSB-first: net1 是 bit 0)
        perm (Tuple[int, ...]): 輸入排列 π，perm[i] 表示 net(i+1) 接到 pin A(perm[i]+1)
        n (int): 輸入數量 N

    Returns:
        int: 晶片腳位組合 u (LSB-first: A1 是 bit 0)

    範例:
        net_pattern = 5 (0b101: net1=1, net2=0, net3=1)
        perm = (1, 0, 2) 表示 net1→A2, net2→A1, net3→A3
        結果:
            A1 收到 net2 = 0
            A2 收到 net1 = 1
            A3 收到 net3 = 1
            u = 6 (0b110)
    """
    pin_pattern = 0
    for net_idx in range(n):
        # 取出 net_idx 的 bit 值
        net_bit = (net_pattern >> net_idx) & 1
        # net_idx 的訊號接到 pin perm[net_idx]
        pin_idx = perm[net_idx]
        # 設定 pin_idx 的值
        pin_pattern |= net_bit << pin_idx
    return pin_pattern


# ============================================================================
# 概念 2: 輸出函數的「指紋」(Bitvector) (spec_simple.md § 3.2)
# ============================================================================

def compute_output_bitvectors(
    truth_table: List[int],
    n: int,
    m: int,
    input_perm: Tuple[int, ...]
) -> List[int]:
    """
    計算給定輸入排列下，所有輸出函數的 bitvector。

    參考: spec_simple.md § 3.2 "概念 2: 輸出函數的指紋 (Bitvector)"

    Bitvector 定義:
        對於輸出 Zj，其 bitvector g[j] 是一個 2^N bit 的整數，
        其中 bit t 表示當外部訊號為 t 時，輸出 Zj 的值。

    Args:
        truth_table (List[int]): 真值表 F，長度為 2^N
            F[u] 是當 pin pattern = u 時的輸出值 (M-bit 整數)
        n (int): 輸入數量 N
        m (int): 輸出數量 M
        input_perm (Tuple[int, ...]): 輸入排列 π

    Returns:
        List[int]: M 個 bitvector，g[j] 對應輸出 Zj

    演算法 (參考 spec_simple.md § 3.2 的偽代碼):
        1. 初始化 M 個 bitvector 為 0
        2. 遍歷所有外部訊號 t (0 到 2^N-1)
        3. 將 t 轉換成 pin pattern u
        4. 查表得到輸出 z = F[u]
        5. 對每個輸出 j，如果 z 的第 j bit 是 1，則設定 g[j] 的第 t bit 為 1
    """
    # 步驟 1: 初始化 M 個輸出，每個 bitvector 初始為 0
    bitvectors = [0] * m

    # 步驟 2: 遍歷所有可能的外部訊號 t
    num_patterns = 1 << n  # 2^N
    for net_pattern in range(num_patterns):
        # 步驟 3: 根據接線 π 將外部訊號 t 轉換成晶片輸入 u
        pin_pattern = apply_input_permutation(net_pattern, input_perm, n)

        # 步驟 4: 查詢真值表，得到輸出
        outputs = truth_table[pin_pattern]  # M-bit 整數

        # 步驟 5: 將每個輸出的結果記錄到對應的 bitvector
        for output_idx in range(m):
            # 檢查輸出 j 的 bit 是否為 1
            if (outputs >> output_idx) & 1:
                # 在 bitvector g[j] 的第 t 個 bit 設為 1
                bitvectors[output_idx] |= 1 << net_pattern

    return bitvectors


# ============================================================================
# 概念 3: 標準化指紋 (Canonical Signature) (spec_simple.md § 3.3)
# ============================================================================

def compute_canonical_signature(
    truth_table: List[int],
    n: int,
    m: int,
    input_perm: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    計算給定輸入排列的標準化指紋。

    參考: spec_simple.md § 3.3 "概念 3: 標準化指紋 (Canonical Signature)"

    標準化指紋定義:
        將 M 個輸出的 bitvector 排序後組成的 tuple。
        排序是為了消除輸出順序的影響，因為輸出腳位可以任意交換。

    Args:
        truth_table (List[int]): 真值表 F
        n (int): 輸入數量 N
        m (int): 輸出數量 M
        input_perm (Tuple[int, ...]): 輸入排列 π

    Returns:
        Tuple[int, ...]: 標準化指紋 (排序後的 bitvector tuple)

    範例說明 (spec_simple.md § 3.3):
        假設有 2 個輸出 (M=2)，兩個不同的輸入接線方式：

        ┌─────────────────────────────────────────────────────────────┐
        │ 接線方式 A: π_A = (0, 1, 2)                                 │
        │   輸入: net1→A1, net2→A2, net3→A3                          │
        │   計算結果:                                                  │
        │     Z1 的 bitvector = 52  (0b00110100)                     │
        │     Z2 的 bitvector = 200 (0b11001000)                     │
        │   bitvectors = [52, 200]                                   │
        │   排序後: signature = (52, 200)  ← 小的在前                │
        └─────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────┐
        │ 接線方式 B: π_B = (1, 0, 2)                                 │
        │   輸入: net1→A2, net2→A1, net3→A3                          │
        │   計算結果:                                                  │
        │     Z1 的 bitvector = 200 (0b11001000)                     │
        │     Z2 的 bitvector = 52  (0b00110100)                     │
        │   bitvectors = [200, 52]  ← 順序和 A 相反！                │
        │   排序後: signature = (52, 200)  ← 排序後一樣！            │
        └─────────────────────────────────────────────────────────────┘

        結論:
            ✓ signature_A = signature_B = (52, 200)
            ✓ 兩個接線方式功能等價

        原因:
            接線 B 產生的輸出函數恰好是接線 A 的 Z1 和 Z2 對調
            (B 的 Z1 = A 的 Z2，B 的 Z2 = A 的 Z1)

            因為我們假設輸出腳位可以任意交換，這兩種情況視為等價
            → 透過排序 bitvector 就能消除這種差異
    """
    # 步驟 1: 計算所有輸出的 bitvector
    bitvectors = compute_output_bitvectors(truth_table, n, m, input_perm)

    # 步驟 2: 排序後組成 tuple
    # 排序確保輸出順序不影響指紋 (Z1,Z2) 和 (Z2,Z1) 得到相同指紋
    signature = tuple(sorted(bitvectors))

    return signature


# ============================================================================
# 主演算法: 等價類別分類 (spec_simple.md § 4)
# ============================================================================

def partition_pin_permutations(
    truth_table: List[int],
    n: int,
    m: int
) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
    """
    將所有可能的輸入排列分類成等價類別。

    參考: spec_simple.md § 4 "完整演算法流程"

    演算法流程 (spec_simple.md § 2):
        步驟 1: 列舉所有可能的輸入排列 π (共 N! 個)
        步驟 2: 對每個排列計算標準化指紋
        步驟 3: 將指紋相同的排列歸為一類

    Args:
        truth_table (List[int]): 真值表 F，長度為 2^N
            F[u] 表示當 pin pattern = u 時的輸出 (M-bit 整數)
        n (int): 輸入數量 N (1 ≤ N ≤ 8)
        m (int): 輸出數量 M (1 ≤ M ≤ 3)

    Returns:
        Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
            字典 {標準化指紋 → 等價的輸入排列列表}

    時間複雜度: O(N! × 2^N × M)
        - N! 個排列需要檢查
        - 每個排列需要 O(2^N × M) 計算 bitvector

    範例輸出 (spec_simple.md § 4):
        {
            (52, 200): [(0,1,2), (0,2,1), ...],  # 第一類: 功能等價的接法
            (100, 150): [(1,0,2), ...],          # 第二類: 功能等價的接法
        }
    """
    # 字典: {指紋 → 接線方式列表}
    signature_to_perms: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

    # 步驟 1: 列舉所有可能的輸入排列 (N! 種)
    all_input_perms = permutations(range(n))

    for input_perm in all_input_perms:
        # 步驟 2: 計算這個排列的標準化指紋
        signature = compute_canonical_signature(truth_table, n, m, input_perm)

        # 步驟 3: 根據指紋分類
        if signature not in signature_to_perms:
            signature_to_perms[signature] = []
        signature_to_perms[signature].append(input_perm)

    return signature_to_perms


# ============================================================================
# 輔助函數: 真值表格式轉換
# ============================================================================

def parse_truth_table(
    n: int,
    m: int,
    rows: List[int]
) -> List[int]:
    """
    解析真值表輸入，確保格式正確。

    Args:
        n (int): 輸入數量 N
        m (int): 輸出數量 M
        rows (List[int]): 真值表的每一行，長度必須是 2^N
            rows[u] 是當 pin pattern = u 時的輸出 (0 到 2^M-1)

    Returns:
        List[int]: 驗證後的真值表

    Raises:
        ValueError: 如果真值表格式不正確
    """
    expected_rows = 1 << n  # 2^N
    if len(rows) != expected_rows:
        raise ValueError(f"真值表長度錯誤: 預期 {expected_rows} 行 (2^{n}), 實際 {len(rows)} 行")

    max_output_value = (1 << m) - 1  # 2^M - 1
    for idx, output in enumerate(rows):
        if not isinstance(output, int) or output < 0 or output > max_output_value:
            raise ValueError(
                f"真值表第 {idx} 行錯誤: 輸出必須是 0 到 {max_output_value} 的整數, 實際為 {output}"
            )

    return rows


# ============================================================================
# 輔助函數: 結果格式化輸出
# ============================================================================

def format_permutation(perm: Tuple[int, ...]) -> str:
    """
    將排列轉換為可讀格式。

    Args:
        perm: 輸入排列 π

    Returns:
        str: 格式化字串，例如 "net1→A2, net2→A1, net3→A3"
    """
    mappings = [f"net{i+1}→A{perm[i]+1}" for i in range(len(perm))]
    return ", ".join(mappings)


def print_equivalence_classes(
    equivalence_classes: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    n: int,
    verbose: bool = False
) -> None:
    """
    印出等價類別的摘要資訊。

    Args:
        equivalence_classes: partition_pin_permutations 的輸出
        n: 輸入數量
        verbose: 是否顯示每個類別的詳細排列
    """
    num_classes = len(equivalence_classes)
    total_perms = sum(len(perms) for perms in equivalence_classes.values())

    print(f"========== 等價類別分析 ==========")
    print(f"輸入數量 N = {n}")
    print(f"總排列數 = {total_perms} (應為 {n}!)")
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

def example_3_input_2_output():
    """
    範例: 3 輸入 2 輸出的邏輯元件

    參考: spec_simple.md § 5 "實際範例"

    邏輯定義:
        Z1 = A1 AND A2
        Z2 = A2 XOR A3
    """
    print("========== 範例: 3 輸入 2 輸出 ==========")
    print("邏輯定義:")
    print("  Z1 = A1 AND A2")
    print("  Z2 = A2 XOR A3")
    print()

    # 建立真值表 (參考 spec_simple.md § 5 的表格)
    n, m = 3, 2
    truth_table = []

    for u in range(1 << n):  # u = 0 到 7
        # 解析 pin pattern u
        a1 = (u >> 0) & 1
        a2 = (u >> 1) & 1
        a3 = (u >> 2) & 1

        # 計算輸出
        z1 = a1 & a2  # AND
        z2 = a2 ^ a3  # XOR

        # 組合成 M-bit 整數 (LSB-first: Z1 是 bit 0)
        output = z1 | (z2 << 1)
        truth_table.append(output)

    # 驗證真值表
    truth_table = parse_truth_table(n, m, truth_table)

    # 分析等價類別
    equivalence_classes = partition_pin_permutations(truth_table, n, m)

    # 印出結果
    print_equivalence_classes(equivalence_classes, n, verbose=True)

    return equivalence_classes


# ============================================================================
# 主程式
# ============================================================================

if __name__ == "__main__":
    # 執行範例
    example_3_input_2_output()

    print("\n" + "=" * 50)
    print("提示: 修改 example_3_input_2_output() 來測試其他邏輯元件")
    print("=" * 50)

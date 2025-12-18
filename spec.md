# Pin Swapping Equivalence Algorithm Specification

## 1. 問題定義

### 1.1 背景
在數位電路設計中，一個邏輯元件 (cell) 有多個輸入腳位和輸出腳位。由於某些腳位在功能上可能是對稱的，交換這些腳位的連接方式可能不會影響電路的整體功能。

### 1.2 目標
給定一個邏輯元件的真值表，找出所有**功能上等價**的腳位排列組合，並將它們分類成等價類別 (equivalence classes)。

### 1.3 術語定義

- **N**: 輸入腳位數量
- **M**: 輸出腳位數量
- **Pin (腳位)**: 元件的物理接腳，記為 A1, A2, ..., AN (輸入) 和 Z1, Z2, ..., ZM (輸出)
- **Net (外部訊號線)**: 外部電路的訊號線，記為 net1, net2, ..., netN (輸入) 和 out1, out2, ..., outM (輸出)
- **Input Permutation (輸入排列) π**: 描述外部訊號線如何連接到輸入腳位的映射
  - `π[i]` 表示 net(i+1) 連接到 pin A(π[i]+1)
  - 例如: π = (0, 1, 2) 表示 net1→A1, net2→A2, net3→A3
- **Output Permutation (輸出排列) σ**: 描述輸出腳位如何連接到外部訊號線的映射
  - `σ[j]` 表示 pin Z(j+1) 連接到 out(σ[j]+1)
- **Truth Table (真值表) F**: 長度為 2^N 的陣列
  - `F[u]` 表示當輸入 pin pattern 為 u 時的輸出值 (M-bit 整數)
  - u 是以 LSB-first 編碼: A1 是 bit 0, A2 是 bit 1, ...
  - `F[u]` 的編碼: Z1 是 bit 0, Z2 是 bit 1, ...

---

## 2. 演算法架構

### 2.1 整體流程

```
輸入: 元件真值表 F, 輸入數 N, 輸出數 M

步驟 1: 列舉所有可能的輸入排列 π (共 N! 個)

步驟 2: 對每個輸入排列 π:
    2.1 計算在此排列下的輸出函數 (以 bitvector 表示)
    2.2 對輸出函數排序，產生標準化簽章 (canonical signature)
    2.3 根據簽章將 π 分類

步驟 3: 輸出等價類別
    - 每個類別包含所有產生相同簽章的輸入排列
    - 可選擇性展開為完整的 (π, σ) 組合

輸出: 等價類別的字典 {signature → list of π}
```

### 2.2 核心概念

#### 概念 1: Pin Pattern vs Net Pattern

- **Net Pattern (t)**: 外部訊號的組合狀態
  - t 是一個 N-bit 整數，LSB-first: net1 是 bit 0, net2 是 bit 1, ...
  - 例如: t = 0b101 表示 net1=1, net2=0, net3=1

- **Pin Pattern (u)**: 元件腳位的組合狀態
  - u 是一個 N-bit 整數，LSB-first: A1 是 bit 0, A2 是 bit 1, ...
  - 例如: u = 0b011 表示 A1=1, A2=1, A3=0

- **轉換關係**: 給定輸入排列 π 和 net pattern t，可計算對應的 pin pattern u
  - 透過 `net_pattern_to_pin_pattern(t, inv_π, N)` 函數完成

#### 概念 2: Output Function Bitvector

對於給定的輸入排列 π，每個輸出腳位 Zj 可以視為一個關於外部 net pattern 的布林函數。

- **Bitvector g[j]**: 一個 2^N bit 的整數，表示輸出 Zj 的完整真值表
  - bit t 表示當 net pattern = t 時，輸出 Zj 的值
  - 例如: 若 g[0] = 0b00110100，表示 Z1 在 t=2,4,5 時為 1

- **計算方式**:
  ```
  for t in range(2^N):
      u = net_pattern_to_pin_pattern(t, inv_π, N)
      z = F[u]  // 從真值表查詢輸出
      for j in range(M):
          if (z >> j) & 1:  // 如果輸出 Zj 為 1
              g[j] |= 1 << t  // 在 bitvector 的第 t 位設為 1
  ```

#### 概念 3: Canonical Signature (標準化簽章)

為了判斷兩個排列是否等價，需要一個**輸出順序無關**的表示方式。

- **定義**: 將 M 個 output bitvector 排序後形成的 tuple
  - `signature = tuple(sorted(g))`

- **性質**:
  - 如果只是交換輸出腳位的順序，signature 不變
  - 兩個輸入排列 π1, π2 產生相同 signature ⇔ 它們功能上等價

- **為什麼要排序?**
  - 因為輸出腳位通常被視為可交換的 (swappable)
  - 例如: (A1,A2,A3,Z1,Z2) 和 (A1,A2,A3,Z2,Z1) 應該被視為等價

---

## 3. 核心函數規格

### 3.1 Truth Table Encoding

#### `bits_to_int(bits: Iterable[int]) -> int`
- **功能**: 將 bit 陣列轉換為整數 (LSB-first)
- **輸入**: bits = [b0, b1, b2, ...] (b0 是最低位元)
- **輸出**: 整數 v = b0 + b1×2 + b2×4 + ...
- **範例**: `bits_to_int([1, 0, 1])` → 5 (0b101)

#### `parse_truth_table_rows(N, M, rows) -> List[int]`
- **功能**: 解析各種格式的真值表輸入
- **支援格式**:
  - 整數: `0` 到 `2^M - 1`
  - 字串: `"010"` (MSB-left, 從左到右是 ZM...Z1)
  - 陣列: `[z1, z2, ..., zM]` (LSB-first)
- **輸出**: 長度為 2^N 的整數陣列 F

#### `int_to_bitstr(v: int, width: int) -> str`
- **功能**: 將整數轉換為固定寬度的二進位字串 (MSB-left)
- **範例**: `int_to_bitstr(5, 4)` → `"0101"`

### 3.2 Permutation Logic

#### `inverse_perm(π: Tuple[int, ...]) -> Tuple[int, ...]`
- **功能**: 計算排列的反函數
- **輸入**: π 是 net → pin 的映射
  - `π[i]` = net(i+1) 連接到的 pin 索引
- **輸出**: inv_π 是 pin → net 的映射
  - `inv_π[p]` = 連接到 pin(p+1) 的 net 索引
- **範例**:
  - π = (1, 0, 2) 表示 net1→A2, net2→A1, net3→A3
  - inv_π = (1, 0, 2) 表示 A1←net2, A2←net1, A3←net3

#### `net_pattern_to_pin_pattern(t, inv_π, N) -> int`
- **功能**: 將外部 net pattern 轉換為內部 pin pattern
- **演算法**:
  ```python
  u = 0
  for p in range(N):
      net_i = inv_π[p]        // pin p 接收來自 net_i
      bit = (t >> net_i) & 1  // 取出 net_i 的 bit 值
      u |= bit << p           // 設定 pin p 的值
  return u
  ```
- **範例**:
  - t = 0b101 (net1=1, net2=0, net3=1)
  - inv_π = (1, 0, 2) (A1←net2, A2←net1, A3←net3)
  - 計算: A1=0 (來自net2), A2=1 (來自net1), A3=1 (來自net3)
  - u = 0b110 = 6

### 3.3 Signature Computation

#### `build_output_functions_bitvectors(F, N, M, π, verbose) -> List[int]`
- **功能**: 計算給定輸入排列下的所有輸出函數 bitvector
- **輸入**:
  - F: 元件真值表
  - N, M: 輸入/輸出數量
  - π: 輸入排列
  - verbose: 是否印出詳細追蹤資訊
- **輸出**: g = [g[0], g[1], ..., g[M-1]]
  - g[j] 是輸出 Zj 的 bitvector (2^N bits)
- **時間複雜度**: O(2^N × M)

#### `canonical_signature_for_pi(F, N, M, π, verbose) -> Tuple[int, ...]`
- **功能**: 計算輸入排列 π 的標準化簽章
- **演算法**:
  1. 呼叫 `build_output_functions_bitvectors` 得到 g
  2. 回傳 `tuple(sorted(g))`
- **輸出**: 長度為 M 的 tuple，元素已排序

### 3.4 Partitioning

#### `partition_tuples(F, N, M, verbose_one_pi) -> Dict`
- **功能**: 將所有輸入排列分類成等價類別
- **演算法**:
  ```python
  sig2pis = {}
  for π in permutations(range(N)):
      sig = canonical_signature_for_pi(F, N, M, π)
      sig2pis.setdefault(sig, []).append(π)
  return sig2pis
  ```
- **輸出**: 字典 `{signature → list of π}`
- **時間複雜度**: O(N! × 2^N × M)

#### `expand_group_to_full_tuples(pis, N, M) -> List[Tuple[str, ...]]`
- **功能**: 將輸入排列擴展為完整的 (π, σ) 組合
- **輸出**: 每個元素為 `(A?, A?, ..., Z?, Z?)` 形式的 tuple
- **數量**: len(pis) × M!

#### `tuple_repr(π, σ, N, M) -> Tuple[str, ...]`
- **功能**: 將排列組合轉換為可讀的表示方式
- **範例**:
  - π = (0, 1, 2), σ = (1, 0)
  - 輸出: `('A1', 'A2', 'A3', 'Z2', 'Z1')`
  - 意義: net1→A1, net2→A2, net3→A3, Z1→out2, Z2→out1

---

## 4. 演算法複雜度分析

### 4.1 時間複雜度

| 操作 | 複雜度 | 說明 |
|------|--------|------|
| 計算單一 bitvector | O(2^N × M) | 遍歷所有 net patterns |
| 計算單一 signature | O(2^N × M + M log M) | bitvector + 排序 |
| 完整分類 | O(N! × 2^N × M) | 遍歷所有輸入排列 |

### 4.2 空間複雜度

| 資料結構 | 大小 | 說明 |
|----------|------|------|
| 真值表 F | O(2^N) | 每個元素是 M-bit 整數 |
| Bitvector g | O(M × ⌈2^N / 64⌉ bytes) | M 個 2^N-bit 整數 |
| 所有排列 | O(N! × N) | 儲存所有輸入排列 |
| 分類結果 | O(K × N! × N) | K 是等價類別數量 |

### 4.3 可擴展性限制

| N | N! | 2^N | 可行性 |
|---|-----|-----|--------|
| 3 | 6 | 8 | ✓ 小規模範例 |
| 4 | 24 | 16 | ✓ 可行 |
| 5 | 120 | 32 | ✓ 可行但較慢 |
| 6 | 720 | 64 | △ 邊界 |
| 8 | 40320 | 256 | ✗ N! 過大 |
| 10 | 3628800 | 1024 | ✗ 不可行 |

**關鍵瓶頸**: 當 N ≥ 8 時，N! 的增長會導致演算法不可行。

---

## 5. 使用範例

### 5.1 定義邏輯元件

```python
# 3 輸入 2 輸出的元件
# Z1 = A1 AND A2
# Z2 = A2 XOR A3
N, M = 3, 2
F = []
for u in range(1 << N):
    a1 = (u >> 0) & 1
    a2 = (u >> 1) & 1
    a3 = (u >> 2) & 1
    z1 = a1 & a2
    z2 = a2 ^ a3
    F.append(bits_to_int([z1, z2]))
```

### 5.2 分析單一排列

```python
# 分析恆等排列
π0 = (0, 1, 2)  # net1→A1, net2→A2, net3→A3
sig = canonical_signature_for_pi(F, N, M, π0, verbose=True)
```

### 5.3 完整分類

```python
# 將所有排列分類
sig2pis = partition_tuples(F, N, M)
print(f"等價類別數量: {len(sig2pis)}")

# 展開第一個類別
one_sig, one_pis = next(iter(sig2pis.items()))
full_tuples = expand_group_to_full_tuples(one_pis, N, M)
print(f"第一個類別的完整組合數: {len(full_tuples)}")
```

---

## 6. 未來改進方向

### 6.1 效能優化

1. **Early termination**: 使用 hash table 快速檢查 signature 是否已見過
2. **Symmetry detection**: 利用代數方法預先偵測對稱性，減少需要檢查的排列
3. **Parallel processing**: 平行化計算不同 π 的 signature
4. **Incremental computation**: 重用部分計算結果

### 6.2 演算法改進

1. **Canonical form**: 使用更先進的 canonical form 技術 (如 BDD, SAT-based methods)
2. **Group theory**: 利用群論 (permutation group) 來系統化分析對稱性
3. **Approximate methods**: 對大規模問題使用近似或啟發式方法

### 6.3 輸出格式優化

1. **大規模 bitvector**: 當 N ≥ 8 時，使用十六進位或壓縮格式顯示
2. **分層輸出**: 只在需要時才展開完整的 (π, σ) 組合
3. **視覺化**: 提供圖形化介面展示等價類別結構

---

## 7. 限制與假設

### 7.1 當前假設

1. 所有輸出腳位被視為**完全可交換** (fully swappable)
2. 元件是**組合邏輯** (combinational logic)，無內部狀態
3. 真值表是**完整且正確**的

### 7.2 已知限制

1. **可擴展性**: N ≥ 8 時不可行 (因為 N! 過大)
2. **記憶體**: verbose 模式在 N ≥ 8 時會產生過長的二進位字串
3. **輸出假設**: 如果某些輸出腳位不可交換，需要修改 signature 計算方式

### 7.3 邊界情況

1. **N = 0 或 M = 0**: 需要特別處理
2. **所有排列等價**: signature 只有一個類別
3. **所有排列不等價**: signature 有 N! 個類別

---

## 8. 參考文獻與相關技術

### 8.1 相關領域

- **Logic synthesis**: 邏輯合成中的 pin assignment 問題
- **Technology mapping**: 將邏輯網路映射到標準元件庫
- **Symmetry detection**: NP-equivalence, canonical form

### 8.2 可能的工具

- **ABC (A System for Sequential Synthesis and Verification)**: Berkeley 的邏輯合成工具
- **Espresso**: 經典的邏輯最小化工具
- **CUDD**: BDD (Binary Decision Diagram) 函式庫

### 8.3 理論基礎

- **Boolean function equivalence**: 布林函數等價性判定
- **Permutation group theory**: 排列群理論
- **Graph isomorphism**: 圖同構問題 (相關但更一般化)

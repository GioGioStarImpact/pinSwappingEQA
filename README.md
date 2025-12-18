# Pin Swapping Equivalence Algorithm

用於分析邏輯元件 (logic cell) 腳位交換等價性的演算法實作。

## 專案說明

這個專案實作了一個演算法，用來找出邏輯元件中哪些輸入/輸出腳位的排列組合在功能上是等價的。當我們交換某些腳位的連接方式時，可能不會影響電路的整體功能，這個演算法可以幫助我們找出所有這樣的等價類別。

## 核心概念

1. **Pin Pattern vs Net Pattern**: 區分晶片腳位狀態和外部訊號狀態
2. **Output Function Bitvector**: 將每個輸出函數表示為完整的真值表
3. **Canonical Signature**: 透過排序 bitvector 產生標準化指紋，用於判斷等價性

## 檔案說明

- `main.py`: 主程式，包含所有核心演算法實作
- `spec.md`: 詳細的演算法規格文件（技術版）
- `spec_simple.md`: 簡化版演算法說明（易讀版）

## 使用方式

### 執行範例

```bash
python main.py
```

### 定義邏輯元件

```python
# 範例: 3 輸入 2 輸出
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

### 分析單一排列

```python
π0 = (0, 1, 2)  # net1→A1, net2→A2, net3→A3
sig = canonical_signature_for_pi(F, N, M, π0, verbose=True)
```

### 完整分類

```python
sig2pis = partition_tuples(F, N, M)
print(f"等價類別數量: {len(sig2pis)}")
```

## 演算法複雜度

| N | N! | 2^N | 可行性 |
|---|-----|-----|--------|
| 3 | 6 | 8 | ✓ 小規模範例 |
| 4 | 24 | 16 | ✓ 可行 |
| 5 | 120 | 32 | ✓ 可行但較慢 |
| 6 | 720 | 64 | △ 邊界 |
| 8 | 40320 | 256 | ✗ N! 過大 |

**關鍵瓶頸**: 當 N ≥ 8 時，N! 的增長會導致演算法不可行。

## 限制

1. **可擴展性**: 適用於小規模問題 (N ≤ 5)
2. **假設**: 假設所有輸出腳位可任意交換
3. **記憶體**: verbose 模式在 N ≥ 8 時會產生過長的輸出

## 未來改進

- 對稱性預檢測以減少需要檢查的排列
- 平行化處理不同排列的計算
- 使用 BDD 或 SAT solver 等進階技術處理大規模問題

## 相關技術

- Logic Synthesis
- Technology Mapping
- Boolean Function Equivalence
- Permutation Group Theory

## License

MIT License

## 作者

jhyeo

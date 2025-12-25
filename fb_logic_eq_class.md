# Frontside/Backside Logic Equivalence Classification

## 問題定義

### 基本概念

假設你有一個邏輯閘，有 k 個輸入腳位和 m 個輸出腳位。但是：

1. **輸入網路 (Input Nets)** 可能來自兩個不同的位置：
   - 一些來自 **frontside** (前側)
   - 一些來自 **backside** (後側)

2. **輸入腳位 (Input Pins)** 也分成兩類：
   - 一些是 **frontside pins** (只能接 frontside nets)
   - 一些是 **backside pins** (只能接 backside nets)

3. **限制條件**：
   - Frontside nets **只能**連接到 frontside pins
   - Backside nets **只能**連接到 backside pins
   - 跨邊連接是**不允許的**

### 問題的挑戰

我們**不知道**：
- 有多少個 frontside nets (p 個？)
- 有多少個 backside nets (q 個？)
- 哪些 nets 是 frontside/backside
- 哪些 pins 是 frontside/backside

### 我們想知道

在所有可能的配置下（不同的 p/q 值、不同的 net/pin 分類），哪些接線方式是功能上等價的。

---

## 簡單範例：4 輸入 1 輸出的 AND 閘

### 邏輯閘定義

**Z1 = A1 AND A2 AND A3 AND A4**

只有當所有輸入都是 1 時，輸出才是 1。

真值表：
```
A4 A3 A2 A1 | Z1
0  0  0  0  | 0
0  0  0  1  | 0
...
1  1  1  1  | 1  ← 只有這一行是 1
```

### 情境 1: 沒有 frontside/backside 限制

如果所有 nets 和 pins 都可以任意連接：
- 所有 4! = 24 種排列都是**等價的**
- 因為 AND 閘對輸入順序不敏感

等價類別：
```
Class 1: 24 個排列（全部等價）
  - inet1→A1, inet2→A2, inet3→A3, inet4→A4
  - inet1→A2, inet2→A1, inet3→A3, inet4→A4
  - ...（共 24 種）
```

### 情境 2: 有 frontside/backside 限制

假設：
- **2 個 frontside nets**: inet1, inet2
- **2 個 backside nets**: inet3, inet4
- **2 個 frontside pins**: A1, A2
- **2 個 backside pins**: A3, A4

可能的接線方式（遵守 front/back 限制）：
```
配置 1: inet1→A1, inet2→A2, inet3→A3, inet4→A4
配置 2: inet1→A1, inet2→A2, inet3→A4, inet4→A3
配置 3: inet1→A2, inet2→A1, inet3→A3, inet4→A4
配置 4: inet1→A2, inet2→A1, inet3→A4, inet4→A3
```

這 4 種配置都是**等價的**（因為是 AND 閘，輸入順序不影響結果）。

但總共只有 4 種，而不是 24 種！限制減少了可能性。

### 情境 3: 不知道哪些是 frontside/backside

如果我們不知道 net/pin 的分類，需要嘗試所有可能：

#### 可能性 A: p=2, q=2 (2 個 front, 2 個 back)

**Net 分類方式**: C(4,2) = 6 種
```
1. {inet1, inet2} 是 front, {inet3, inet4} 是 back
2. {inet1, inet3} 是 front, {inet2, inet4} 是 back
3. {inet1, inet4} 是 front, {inet2, inet3} 是 back
4. {inet2, inet3} 是 front, {inet1, inet4} 是 back
5. {inet2, inet4} 是 front, {inet1, inet3} 是 back
6. {inet3, inet4} 是 front, {inet1, inet2} 是 back
```

**Pin 分類方式**: C(4,2) = 6 種
```
1. {A1, A2} 是 front, {A3, A4} 是 back
2. {A1, A3} 是 front, {A2, A4} 是 back
3. {A1, A4} 是 front, {A2, A3} 是 back
4. {A2, A3} 是 front, {A1, A4} 是 back
5. {A2, A4} 是 front, {A1, A3} 是 back
6. {A3, A4} 是 front, {A1, A2} 是 back
```

每種 (net 分類, pin 分類) 組合有 2! × 2! = 4 種排列。

總共: 6 × 6 × 4 = **144 種組合**

#### 可能性 B: p=1, q=3 (1 個 front, 3 個 back)

- Net 分類: C(4,1) = 4 種
- Pin 分類: C(4,1) = 4 種
- 排列數: 1! × 3! = 6

總共: 4 × 4 × 6 = **96 種組合**

#### 可能性 C: p=0, q=4 (0 個 front, 4 個 back)

- 只有 1 種 net 分類（全部 back）
- 只有 1 種 pin 分類（全部 back）
- 排列數: 4! = 24

總共: **24 種組合**

#### 可能性 D: p=4, q=0 (4 個 front, 0 個 back)

同樣 **24 種組合**

#### 可能性 E: p=3, q=1

同 p=1, q=3，**96 種組合**

### 總結

對於 k=4 的 AND 閘：
- **沒有限制**: 24 種排列，全部等價 → 1 個等價類別
- **有 front/back 限制 (未知分類)**:
  - p=0: 24 種組合
  - p=1: 96 種組合
  - p=2: 144 種組合
  - p=3: 96 種組合
  - p=4: 24 種組合
  - **總計: 384 種組合**

但因為是 AND 閘（對稱性），所有這 384 種組合仍然是**功能等價的**！
只是它們對應到不同的 (net分類, pin分類) 配置。

---

## 更有趣的範例：4 輸入 2 輸出的非對稱閘

### 邏輯定義

```
Z1 = A1 AND A2
Z2 = A3 XOR A4
```

這個邏輯閘**不對稱**：
- Z1 只依賴前兩個輸入 (A1, A2)
- Z2 只依賴後兩個輸入 (A3, A4)

### 在沒有 front/back 限制時

等價類別會有很多個，因為交換 A1 和 A3 會改變功能。

### 加上 front/back 限制後

假設 p=2, q=2，且：
- Net 分類: {inet1, inet2} front, {inet3, inet4} back
- Pin 分類: {A1, A2} front, {A3, A4} back

可能的配置：
```
配置 1: inet1→A1, inet2→A2, inet3→A3, inet4→A4
  → Z1 = inet1 AND inet2
  → Z2 = inet3 XOR inet4

配置 2: inet1→A2, inet2→A1, inet3→A3, inet4→A4
  → Z1 = inet2 AND inet1 = inet1 AND inet2 (交換律)
  → Z2 = inet3 XOR inet4

配置 1 和 2 是等價的！
```

但如果 pin 分類改變：
- Net 分類: {inet1, inet2} front, {inet3, inet4} back
- Pin 分類: {A1, A3} front, {A2, A4} back

```
配置 3: inet1→A1, inet2→A3, inet3→A2, inet4→A4
  → Z1 = inet1 AND inet3  ← 跨越了原本的邊界！
  → Z2 = inet2 XOR inet4

配置 3 和配置 1 **不等價**！
```

這展示了為什麼 pin 分類很重要。

---

## 演算法概述

### 輸入
- 真值表 F (k 個輸入, m 個輸出)

### 輸出
- 等價類別列表，每個類別包含：
  - Signature (功能指紋)
  - 屬於這個類別的所有 (排列, net分類, pin分類) 組合

### 演算法流程

```
for p from 0 to k:
    q = k - p

    for net_classification in C(k, p) ways:
        # 選擇哪 p 個 nets 是 frontside

        for pin_classification in C(k, p) ways:
            # 選擇哪 p 個 pins 是 frontside

            for front_perm in p! permutations:
                # Frontside nets 的排列

                for back_perm in q! permutations:
                    # Backside nets 的排列

                    π = construct_permutation(...)
                    sig = compute_signature(F, π)

                    equivalence_classes[sig].append((
                        π, net_classification, pin_classification
                    ))
```

### 複雜度

總排列數 = Σ(p=0 to k) [ C(k,p) × C(k,p) × p! × q! ]

| k | 總排列數 | 估計時間 (單核) | 估計時間 (48 cores) |
|---|----------|----------------|-------------------|
| 4 | 384 | < 0.1 秒 | < 0.01 秒 |
| 6 | 46,080 | ~1 秒 | < 0.1 秒 |
| 8 | 10,321,920 | ~33 分鐘 | ~42 秒 |

---

## 實際應用

這個演算法在以下情況有用：

1. **晶片設計**: 當 nets 來自不同的 die 或 package 側面
2. **PCB 佈局**: 當某些訊號線必須從特定側面進入
3. **對稱性分析**: 找出在物理限制下仍然保持功能等價的接線方式
4. **設計驗證**: 確保在所有允許的配置下，電路功能保持一致

---

## 與原始演算法的關係

原始的 `pin_swapping.py` 相當於：
- p = k, q = 0 (所有 nets/pins 都是 frontside，或都是 backside)
- 只有 1 種 net 分類、1 種 pin 分類
- 總排列數 = k!

新的 `fb_logic_eq_class.py` 探索所有可能的 p/q 組合和分類方式，
因此總排列數約為 **k! × 2^k**。

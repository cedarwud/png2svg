# P0 Phase Implementation Plan
## PNG2SVG 核心改進實施計劃

### 執行摘要
基於對現有代碼庫的深入分析，本計劃針對三個核心領域進行改進：
- P0.1 曲線提取增強
- P0.2 文字提取增強
- P0.3 幾何元素精確化

目標：將 bad_pixel_ratio 從 0.505 降低到 < 0.30

---

## P0.1 曲線提取增強

### 現狀分析
**文件**: `src/png2svg/extractor_curves.py`

當前實現：
- 使用 HSV 顏色分割提取曲線遮罩
- 逐列取中位數得到中心線
- 使用 RDP 算法簡化點序列
- 輸出為分段線性路徑

問題：
1. 無貝茲曲線擬合，曲線呈現鋸齒狀
2. 顏色分割對光照/陰影敏感
3. 中心線計算對粗線條效果差

### 改進方案

#### 1.1 貝茲曲線擬合器 (新增)
**文件**: `src/png2svg/curve_fitting.py`

```python
def fit_bezier_curve(points: list[tuple[float, float]],
                     max_error: float = 2.0) -> list[BezierSegment]:
    """
    將分段線性點列擬合為貝茲曲線

    算法：
    1. 使用 Philip J. Schneider 的曲線擬合算法
    2. 遞歸分割直到誤差 < max_error
    3. 返回三次貝茲曲線段列表
    """
```

關鍵實現：
- 實現 `_compute_tangents()` 計算端點切線
- 實現 `_fit_cubic()` 擬合單段三次貝茲
- 實現 `_split_and_fit()` 遞歸分割

#### 1.2 顏色分割改進
**修改**: `extractor_curves.py` 的 `_curve_color_mask()`

改進：
- 添加形態學閉運算填補斷裂
- 使用自適應閾值替代固定閾值
- 增加小連通區域過濾

```python
def _curve_color_mask_v2(rgba, target_hue, adaptive):
    # 原有 HSV 分割
    mask = _curve_color_mask(rgba, target_hue, adaptive)

    # 新增：形態學處理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 新增：移除小連通區域
    mask = _remove_small_components(mask, min_area=50)

    return mask
```

#### 1.3 整合 potrace (可選)
**依賴**: `pypotrace` 或 subprocess 調用 `potrace`

用於複雜曲線的位圖→向量轉換：
```python
def trace_curve_with_potrace(mask: np.ndarray) -> str:
    """使用 potrace 將二值遮罩轉為 SVG path"""
```

### 交付物
- [ ] `src/png2svg/curve_fitting.py` - 貝茲擬合模組
- [ ] 修改 `extractor_curves.py` - 整合擬合器
- [ ] 修改 `renderer.py` - 支援三次貝茲路徑輸出

---

## P0.2 文字提取增強

### 現狀分析
**文件**: `src/png2svg/extractor_text.py`, `src/png2svg/ocr.py`

當前實現：
- 使用 Tesseract OCR 識別文字
- 基於 bbox 高度估算 font-size
- 基於位置推斷 text-anchor

問題：
1. OCR 產生噪音文字 (如 "wore A5 triggered")
2. font-weight 判斷不準確
3. 無法檢測斜體/下標

### 改進方案

#### 2.1 文字過濾增強
**修改**: `extractor_text.py` 的 `_keep_text_item()`

新增過濾規則：
```python
def _keep_text_item_v2(item, cfg, known_vocabulary):
    # 原有檢查
    if not _keep_text_item(item, cfg):
        return False

    text = item.get("text", "")

    # 新增：模板特定詞彙表比對
    if known_vocabulary:
        similarity = _fuzzy_match_vocabulary(text, known_vocabulary)
        if similarity < 0.6:
            return False

    # 新增：字符連貫性檢查
    if _has_random_char_pattern(text):
        return False

    # 新增：重複檢測
    if item.get("is_duplicate"):
        return False

    return True
```

#### 2.2 模板詞彙表
**新增**: `config/vocabulary/t_3gpp_events_3panel.yaml`

```yaml
# 3GPP 事件圖表的預期詞彙
required:
  - "A3 Event"
  - "A4 Event"
  - "A5 Event"
  - "TTT"
  - "Hys"
  - "Serving beam"
  - "Neighbor/Target beam"
  - "Time (t)"
  - "t_start"
  - "t_trigger"
  - "Th_A4"
  - "Th1_A5"
  - "Th2_A5"

optional_patterns:
  - "^A[345].*triggered$"
  - "^Condition.*met$"
  - "^Report sent"
  - "Measured signal level"
```

#### 2.3 Font-weight 估算改進
**修改**: `extractor_text.py`

基於像素密度估算粗細：
```python
def _estimate_font_weight(rgba, bbox):
    """
    基於文字區域的像素密度估算 font-weight
    bold 文字通常有更高的 ink_ratio
    """
    region = _crop_region(rgba, bbox)
    ink_ratio = _compute_ink_ratio(region)

    if ink_ratio > 0.35:
        return "bold"
    return "normal"
```

### 交付物
- [ ] 修改 `extractor_text.py` - 增強過濾邏輯
- [ ] 新增 `config/vocabulary/*.yaml` - 模板詞彙表
- [ ] 修改 `ocr.py` - 添加詞彙表支援

---

## P0.3 幾何元素精確化

### 現狀分析
**文件**: `src/png2svg/extractor_geometry.py`

當前實現：
- 面板檢測基於列墨水分布
- 虛線檢測基於 run-length 分析
- 軸線檢測基於長線條檢測

問題：
1. 箭頭類型未分類 (三角形/線條)
2. 虛線模式固定為 [4,4]
3. 產生重複的幾何元素

### 改進方案

#### 3.1 箭頭類型識別器
**新增**: `src/png2svg/arrow_detector.py`

```python
@dataclass
class ArrowHead:
    type: Literal["triangle", "line", "dot", "none"]
    position: tuple[float, float]
    direction: float  # 角度
    size: float

def detect_arrow_type(mask: np.ndarray,
                      line_endpoint: tuple[float, float]) -> ArrowHead:
    """
    在線條端點附近檢測箭頭類型

    算法：
    1. 裁剪端點周圍區域
    2. 檢測三角形形狀 (凸包分析)
    3. 檢測線條形狀 (Hough 線檢測)
    4. 檢測圓點 (Hough 圓檢測)
    """
```

#### 3.2 虛線模式自動檢測
**修改**: `extractor_geometry.py` 的 `_detect_dashed_lines()`

```python
def _detect_dash_pattern(runs: list[tuple[int, int]]) -> tuple[int, int]:
    """
    從 run-length 數據推斷 dasharray 模式

    Returns:
        (dash_length, gap_length)
    """
    dash_lengths = [r[1] for r in runs]
    gaps = []
    for i in range(len(runs) - 1):
        gap = runs[i+1][0] - (runs[i][0] + runs[i][1])
        gaps.append(gap)

    median_dash = int(np.median(dash_lengths))
    median_gap = int(np.median(gaps)) if gaps else median_dash

    return (median_dash, median_gap)
```

#### 3.3 重複元素去除
**新增**: `src/png2svg/dedup.py`

```python
def deduplicate_lines(lines: list[dict],
                      distance_threshold: float = 3.0) -> list[dict]:
    """
    去除重複或重疊的線條

    算法：
    1. 計算所有線條對的距離
    2. 使用 DBSCAN 聚類相近線條
    3. 每個簇保留最長/最佳的線條
    """
```

### 交付物
- [ ] 新增 `src/png2svg/arrow_detector.py` - 箭頭識別
- [ ] 修改 `extractor_geometry.py` - 虛線模式檢測
- [ ] 新增 `src/png2svg/dedup.py` - 去重邏輯

---

## 實施順序

### Week 1: 基礎改進
1. 實現文字過濾增強 (P0.2.1) - 快速減少 OCR 噪音
2. 實現重複元素去除 (P0.3.3) - 清理 SVG 冗餘

### Week 2: 曲線改進
3. 實現貝茲曲線擬合器 (P0.1.1)
4. 整合到渲染流程

### Week 3: 精細化
5. 實現箭頭類型識別 (P0.3.1)
6. 實現虛線模式檢測 (P0.3.2)
7. 添加模板詞彙表 (P0.2.2)

### Week 4: 測試與調優
8. 更新回歸測試
9. 調整閾值參數
10. 文檔更新

---

## 成功指標

| 指標 | 當前值 | 目標值 |
|------|-------|-------|
| bad_pixel_ratio | 0.505 | < 0.30 |
| RMSE | 0.231 | < 0.20 |
| OCR 噪音文字數 | ~5-10 | 0 |
| 重複幾何元素 | ~12 | 0 |
| 曲線平滑度 | 分段線性 | 三次貝茲 |

---

## 技術風險

1. **貝茲擬合精度**: 可能需要多次調參
   - 緩解：提供可配置的 max_error 參數

2. **potrace 依賴**: 不是所有環境都有
   - 緩解：作為可選功能，無依賴時降級

3. **詞彙表維護**: 每個模板需要維護詞彙表
   - 緩解：從現有 expected.svg 自動生成

---

## 附錄：關鍵代碼位置

- 曲線提取: `src/png2svg/extractor_curves.py:155-197`
- 文字過濾: `src/png2svg/extractor_text.py:331-402`
- 虛線檢測: `src/png2svg/extractor_geometry.py:215-336`
- 渲染器: `src/png2svg/renderer.py`

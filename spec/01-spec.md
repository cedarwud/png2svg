# Spec — PNG→High-Quality Editable SVG (MVP v0)

## 1) Problem
使用者有大量論文圖（LEO satellite handover 領域常見圖型），希望：
- 由 PNG 產出「可在 Adobe Illustrator 內像原生向量一樣編輯」的輸出
- 尤其重視：文字可改、圖層好找、對齊乾淨、看不出是追圖
- 希望建立一套可擴充的 pipeline：之後能逐步支援更多圖型

## 2) Users & Primary Use Cases
- 使用者（研究者）：
  - 把論文圖轉成可編輯 SVG，後續改字、改線型、改配色、重排版
- 主要場景：
  1) 針對已知模板（3GPP events / flowchart / line plot）輸出高品質 SVG
  2) 對輸出做機器化驗收（禁止 `<image>`、文字必須 `<text>`、複雜度限制等）
  3) 回歸測試：確保加功能不會讓既有高品質模板變差

## 3) In Scope (MVP v0)
- CLI 工具：
  - png2svg（模板模式）
  - validate（契約驗收）
  - regress（回歸套件）
  - pdf2png（基本抽圖/抽頁，v0 可簡化）
- 三個模板的 renderer：
  - t_3gpp_events_3panel
  - t_procedure_flow
  - t_performance_lineplot
- Contract v1 + Thresholds v1（YAML）
- Regression dataset（10–15 張，小而精）

## 4) Out of Scope (MVP v0)
- 泛用任意 PNG 的「高品質」自動重建
- 強依賴 OCR/電腦視覺去理解未知圖型（可在 v1+ 再做）
- 自動從 PNG 推導 params（v0 允許人工準備 params.json）

## 5) Inputs & Outputs
### Input
- input.png（尺寸即輸出尺寸）
- params.json（模板參數，定義圖中元素幾何與文字）

### Output
- output.svg（符合 Figure Contract v1）
- validate report（JSON）
- regress summary（console + optional JSON）

## 6) Success Criteria (v0)
- 每個模板至少 3–5 個 regression case 全數通過：
  - validator：pass
  - visual diff：在門檻內（可允許少量抗鋸齒差異）
- Illustrator 中可用性（人工 spot check）：
  - 文字是可編輯文字
  - 圖層/群組命名可找
  - 沒有嵌入圖片

# Constitution (SDD v0) — PNG→High-Quality Editable SVG

## 0) Single Source of Truth
本專案的真理來源依序為：
1. spec/00-constitution.md（不可違反的原則）
2. spec/04-acceptance.md（驗收準則）
3. config/*.yaml（可機器化的規則與門檻）
4. 其餘文件（spec/01~03、AGENTS.md）

## 1) Non-negotiables（不可妥協）
- 產出 SVG 必須「看起來像 Illustrator 從零畫」：幾何乾淨、對齊一致、線條筆直、排版整齊。
- **之後要改字**：文字必須保留為 `<text>`（不可描邊成 path）。
- 禁止 raster 內嵌：SVG **不得出現 `<image>`**。
- 禁止濾鏡/漸層：不得使用 `<filter>`、`<linearGradient>`、`<radialGradient>` 與 `fe*`。
- 可重現：同一份 params + 同一版本程式 → 輸出必須 deterministic（不靠隨機）。
- MVP v0 以 **Template-based reconstruction** 取得高品質；不做泛用 trace 當主要路徑。

## 2) Quality Philosophy
- 追求「語意化向量」而非「像素追蹤」：
  - 直線→`<line>`
  - 方塊→`<rect>`
  - 圓點→`<circle>`
  - 曲線→少段數 `<path>`（Bezier）
  - 文字→`<text>`（含可讀 id）
- 人眼對齊敏感，所以必做：
  - snap to grid
  - baseline alignment
  - 嚴格限制顏色數與線寬集合

## 3) Scope Rule（MVP v0 範圍）
- v0 支援三大模板：
  1) 3-panel 3GPP A3/A4/A5 infographic
  2) procedure / flow / state diagram
  3) performance line plot
- 任何超出模板範圍的 PNG，在 v0 不承諾高品質「像手刻」，最多做到「不違反 contract」的中等輸出。

## 4) Engineering Rules
- 所有規則都必須可被 validator 檢查與量化。
- 每加一個功能或調整參數，都要加/更新 regression case，避免品質倒退。
- 不要先做「自動偵測萬能版」，先把模板品質做穩做可回歸。

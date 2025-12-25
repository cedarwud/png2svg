# Plan — Technical Implementation Plan (MVP v0)

## 1) Architecture Overview
以 Template-based renderer 為核心，形成三段 pipeline：

(1) Render
- png2svg --mode template
  - 讀 input.png 取得 canvas size（預設輸出同尺寸）
  - 讀 params.json → 呼叫指定 template renderer
  - renderer 用乾淨幾何 primitives + 少段數曲線產生 SVG
  - 保證 group/id/字體/顏色/線寬符合 contract

(2) Validate
- validate_svg
  - 解析 SVG XML
  - 檢查：結構/禁用元素/群組命名/字體/顏色數/線寬集合/path 複雜度
  - 產出 JSON report（可定位原因）

(3) Regression (visual + structural)
- regress
  - 對每個 case：render → validate → rasterize → diff expected.svg（必要時 fallback expected.png）
  - fail-fast + summary

## 2) Key Design Decisions
- 高品質來源 = 模板重建（不是 trace）
- “像 AI 手刻” 的核心技術點：
  - snap to grid（幾何吸附）
  - baseline alignment（文字基線對齊）
  - 限制線寬集合/顏色數
  - 曲線用少段數 Bezier（不是上百個點）
- Validator 是第一級產品（不是附屬）

## 3) Dependencies
### System (Linux)
- resvg（SVG rasterize for diff；若不可用，後備 cairosvg）

### Python
- typer（CLI）
- pillow（讀 PNG 尺寸）
- svgwrite 或 lxml（產 SVG）
- numpy（diff）
- pyyaml（讀 YAML）
- pytest（測試）

## 4) Template Param Conventions (v0)
- params.json 必含：
  - template: string
  - canvas: {width,height}（若缺，從 input.png 讀）
  - title: string（可選）
  - template-specific fields

### 4.1 t_3gpp_events_3panel（最重要）
- panels[3]：A3/A4/A5
- t_start_ratio / t_trigger_ratio：以 panel 區域比例定義
- curves: serving/neighbor 的 points_ratio（renderer 會平滑並轉 Bezier）

### 4.2 t_procedure_flow
- nodes[]：rect + text
- edges[]：line/polyline + arrowhead + label
- lanes[] 可選（swimlane）

### 4.3 t_performance_lineplot
- axes：ticks/labels
- series[]：data points → plot mapping → path/polyline（低複雜度）

## 5) Visual Regression Strategy
- rasterize SVG → PNG（1x）
- diff：
  - RMSE
  - bad_pixel_ratio（超過容忍差的像素比例）
- 允許小抗鋸齒差異，但要能抓出布局/線型/文字位置的大退步

## 6) Milestones
- M0：validator 核心（禁止 image/filter/gradient + required groups + 字體檢查）
- M1：Template #1 可回歸（3–5 cases）
- M2：加入 visual diff 回歸
- M3：Template #2 可回歸
- M4：Template #3 可回歸

# Tasks — MVP v0 Task Breakdown (Spec Kit style)

> 原則：每個任務都要有 DoD（Definition of Done），可由 regress/validate 證明。

## Phase 0 — Repo skeleton
- 建立目錄結構：spec/, config/, tools/, src/, datasets/
DoD:
- tree 結構符合 spec/01

## Phase 1 — Contract & Threshold configs
- 產出兩份 YAML：
  - config/figure_contract.v1.yaml
  - config/validator_thresholds.v1.yaml
DoD:
- validate 工具能載入 YAML（即使尚未完整檢查）

## Phase 2 — Validator core (no templates yet)
- tools/validate_svg.py
- 檢查項最低限度：
  - 禁止 <image>
  - 禁止 filter/gradient/fe*
  - 必備 group id：figure_root + g_axes/g_curves/g_annotations/g_text/g_markers
  - font-family 必須含 Arial/Liberation Sans/sans-serif
  - 顏色數 <= max_colors
  - stroke-width 在 allowed set（允許小容忍）
  - path command count 不超過 max_path_commands
- 產出 report.json
DoD:
- 對刻意違規的 svg 會 fail 並給清楚 error code

## Phase 3 — Rasterize + visual diff
- 加入 resvg rasterize（後備 cairosvg）
- diff 計算 RMSE + bad_pixel_ratio
DoD:
- validate 可以附帶 visual_diff metrics（若提供 expected.svg 或 expected.png）
- regress 工具可以跑完一個 case

## Phase 4 — png2svg template framework
- tools/png2svg.py：讀 input.png + params.json，呼叫 template
- src/common/svg_builder.py：建立標準 groups + style helpers
DoD:
- 用 “dummy template” 也能產出合規的空 SVG（僅 groups + title text）

## Phase 5 — Template #1 (t_3gpp_events_3panel)
- 完成 renderer：
  - 3 panels
  - curves（少段數）
  - shaded TTT window
  - t_start/t_trigger guide lines
  - text ids + panel grouping
- 建立 regression cases（至少 3–5）
DoD:
- regress 全通過
- validator pass
- Illustrator 人工檢查：文字可改、群組好找

## Phase 6 — Template #2 (t_procedure_flow)
- nodes/edges/lanes
- 箭頭 marker 或 polygon
- multiline text via tspan
- cases（至少 3）
DoD:
- regress 全通過

## Phase 7 — Template #3 (t_performance_lineplot)
- axes/ticks/legend/series
- cases（至少 3）
DoD:
- regress 全通過

## Phase 8 — Hardening
- 補齊錯誤訊息、文件、範例 params
- 把 thresholds 收斂（別太鬆）
DoD:
- 10–15 regression cases stable

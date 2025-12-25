# Acceptance — MVP v0 Definition of Done

## 1) Functional Acceptance
- 必須提供 CLI：
  - png2svg（template mode）
  - validate
  - regress
- 必須支援模板：
  - t_3gpp_events_3panel
  - t_procedure_flow
  - t_performance_lineplot

## 2) Contract Acceptance (hard fail)
- SVG 不得包含：
  - <image>
  - <filter>, <linearGradient>, <radialGradient>, fe*
- 文字必須為 <text>（不得把文字 outline/path）
- 必須存在 groups：
  - figure_root
  - g_axes, g_curves, g_annotations, g_text, g_markers
- 必須符合字體規則（Arial/Liberation Sans/sans-serif）
- 顏色數 <= max_colors
- stroke-width 必須落在允許集合（含容忍）
- path 複雜度（command count）不得超過門檻

## 3) Visual Regression Acceptance
- regression dataset 至少 10 個 cases（建議 12–15）
- regress 必須全通過才算綠燈
- visual diff 使用：
  - RMSE <= 門檻
  - bad_pixel_ratio <= 門檻

## 4) Usability Acceptance (manual spot check)
（每個模板至少抽 1 case 在 Illustrator 手動檢查）
- 能直接選取並編輯文字內容
- 群組/圖層命名能快速定位（如 panel_A3、txt_title）
- 沒有任何嵌入圖片
- 曲線不會是一大坨上百個節點（不呈現 trace 感）

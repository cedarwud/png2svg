# Codex CLI Runbook (Spec Kit style)

依 Spec Kit 的精神：先完成一個 phase 的 DoD，再進下一 phase。
（不要一次叫 agent 全做完，容易偏）

## Step 0 — Read specs
Prompt to Codex:
"Read spec/00-constitution.md through spec/05-codex-runbook.md and AGENTS.md. Follow the Constitution and Acceptance strictly."

## Step 1 — Scaffold
Prompt:
"Implement Phase 0: create folder structure, empty CLI entrypoints, and a minimal pyproject/requirements. Do not implement templates yet."

## Step 2 — Configs
Prompt:
"Implement Phase 1: create config/figure_contract.v1.yaml and config/validator_thresholds.v1.yaml consistent with spec/04 acceptance."

## Step 3 — Validator core
Prompt:
"Implement Phase 2 validator core. Provide clear error codes. Add pytest unit tests for forbidden tags and required groups."

## Step 4 — Rasterize + diff
Prompt:
"Implement Phase 3: rasterize SVG to PNG using resvg if available, else cairosvg. Implement visual diff metrics and integrate into regress."

## Step 5 — Template framework + Template #1
Prompt:
"Implement Phase 4 and Phase 5. Focus on template t_3gpp_events_3panel. Add 3–5 regression cases and make regress pass."

## Step 6 — Template #2
Prompt:
"Implement Phase 6 procedure_flow. Add cases and keep existing cases passing."

## Step 7 — Template #3
Prompt:
"Implement Phase 7 performance_lineplot. Add cases and keep suite green."

## Step 8 — Hardening
Prompt:
"Implement Phase 8: improve messages, docs, and sample params. Tighten thresholds carefully without breaking regress."

# 評估指標與驗證框架

本專案的評估邏輯集中在 [benchmark_suite](benchmark_suite) 與 [tools/ode_simulator.py](tools/ode_simulator.py)。工作流會先讓 Cello/ODE topology 帶有初步 metrics，再由 [benchmark_suite/benchmark_controller.py](benchmark_suite/benchmark_controller.py) 的 `evaluate_candidate()` 產生統一的 weighted total score。

## 1. 評估總覽

`evaluate_candidate(candidate)` 會呼叫下列 scorer：

- [benchmark_suite/functional_scorer.py](benchmark_suite/functional_scorer.py)：功能正確性。
- [benchmark_suite/kinetic_scorer.py](benchmark_suite/kinetic_scorer.py)：動力學與 Monte Carlo robustness。
- [benchmark_suite/static_plausibility_evaluator.py](benchmark_suite/static_plausibility_evaluator.py)：靜態可行性。
- [benchmark_suite/metabolic_scorer.py](benchmark_suite/metabolic_scorer.py)：gate count 與 metabolic burden。
- [benchmark_suite/temporal_scorer.py](benchmark_suite/temporal_scorer.py)：response/rise time。
- [benchmark_suite/cello_constraint_evaluator.py](benchmark_suite/cello_constraint_evaluator.py)：Cello assignment、orthogonality、toxicity、buildability。

`evaluate_candidate()` 會回傳：

- `score`
- `weighted_total_score`
- `grade`
- `component_scores`
- `score_weights`
- `details`
- 各類常用指標，如 `metabolic_burden_score`、`robustness_score`、`orthogonality_score`、`cello_assignment_score`。

## 2. Weighted Total Score

權重定義在 [benchmark_suite/benchmark_controller.py](benchmark_suite/benchmark_controller.py) 的 `SCORE_WEIGHTS`：

| Component | Weight |
| --- | ---: |
| `functional` | 0.22 |
| `kinetic` | 0.15 |
| `static_plausibility` | 0.08 |
| `metabolic_burden` | 0.15 |
| `robustness` | 0.15 |
| `temporal` | 0.05 |
| `orthogonality` | 0.10 |
| `cello_assignment` | 0.10 |

公式：

```text
weighted_total_score =
  0.22 * functional
+ 0.15 * kinetic
+ 0.08 * static_plausibility
+ 0.15 * metabolic_burden
+ 0.15 * robustness
+ 0.05 * temporal
+ 0.10 * orthogonality
+ 0.10 * cello_assignment
```

所有 component score 都會經 `_clamp_score()` 限制在 0 到 1。

等級定義：

- `Excellent`：score >= 0.80
- `Pass`：0.60 <= score < 0.80
- `Fail`：score < 0.60

## 3. Functional Scorer

實作位置：[benchmark_suite/functional_scorer.py](benchmark_suite/functional_scorer.py)

功能評估會嘗試從 candidate 讀取：

- `truth_table`
- `truth_table_or_logic_matrix`
- `verilog`
- `min_on`
- `max_off`
- `fold_change`

主要評估面向：

- Verilog 是否能用簡化的 combinational parser 模擬。
- truth table row 的 expected output 是否與 Verilog 輸出一致。
- ON/OFF fold change 與 margin 是否足夠。

支援的 Verilog 內容偏向簡單 combinational gates：

- `assign`
- `and`
- `or`
- `not`
- `nand`
- `nor`
- `xor`
- `xnor`
- `buf`

## 4. Kinetic Scorer

實作位置：[benchmark_suite/kinetic_scorer.py](benchmark_suite/kinetic_scorer.py)

Kinetic scorer 會優先使用 candidate 既有的模擬結果，例如：

- `kinetic_score`
- `robustness_score`
- `signal_to_noise_ratio` 或 `snr`
- `monte_carlo_runs` 或 `monte_carlo_samples`
- `monte_carlo_failure_rate`
- `monte_carlo_terminal_output_cv`
- `metrics_cv`
- `dynamic_margin`

若 candidate 有足夠 simulation inputs，也可執行 noisy response 評估。SNR 轉分數使用 `_snr_to_score()`。

## 5. ODE Simulation Engine

實作位置：[tools/ode_simulator.py](tools/ode_simulator.py)

`BatchODESimulator` 會對每個 topology 執行資源感知 ODE 模擬。重要類別：

- `WarmStartResourceSolver`：估算 free RNAP/ribosome，並回傳 occupancy。
- `ResourceAwareSimulation`：建立 mRNA/protein dynamics RHS。
- `BatchODESimulator`：批次模擬 topology、Monte Carlo perturbation、cache。

### 5.1 動力學模型

狀態向量包含每個 gene 的 mRNA 與 protein：

```text
y = [mRNA_1 ... mRNA_n, protein_1 ... protein_n]
```

RHS 會考慮：

- transcription rate
- translation rate
- mRNA degradation
- protein degradation
- RNAP resource occupancy
- ribosome resource occupancy
- Hill repression
- leak fraction

### 5.2 Solver

模擬器優先使用 SciPy：

1. `solve_ivp(..., method="BDF")`
2. `solve_ivp(..., method="Radau")`

若 SciPy 不存在或 solver 失敗，會使用內建 `_rk4_integrate()` 作為 fallback。

### 5.3 Monte Carlo

`BatchODESimulator` 支援：

- `monte_carlo_samples`
- `noise_fraction`
- `noise_level`

會 perturb：

- `transcription_rate`
- `translation_rate`
- `kd`
- `hill_coefficient`
- `leak_fraction`
- `mrna_degradation_rate`
- `protein_degradation_rate`
- `y_min`
- `ymax`
- `y_max`

輸出會包含：

- `monte_carlo_runs`
- `monte_carlo_failure_rate`
- `monte_carlo_terminal_output_cv`

### 5.4 ODE 輸出欄位

成功模擬時 topology 會被補上：

- `ode_status = "simulated"`
- `gene_count`
- `kinetic_score`
- `robustness_score`
- `signal_to_noise_ratio`
- `metrics_max_burden`
- `metrics_cv`
- `dynamic_margin`
- `resource_occupancy`
- `parameter_provenance`
- `benchmark_report`

失敗時：

- `ode_status = "failed"`
- `kinetic_score = 0.0`
- `robustness_score = 0.0`
- `score = 0.0`
- `benchmark_report.details` 會包含 `{"metric": "kinetic", "status": "ode_failed"}`

## 6. Metabolic Burden Scorer

實作位置：[benchmark_suite/metabolic_scorer.py](benchmark_suite/metabolic_scorer.py)

此 scorer 會從 Verilog 計算 logic gates 數量。支援 primitive gates：

- `and`
- `nand`
- `or`
- `nor`
- `xor`
- `xnor`
- `not`
- `buf`

分數函式：

```text
metabolic_burden_score = exp(-decay_rate * max(0, gate_count - free_gate_count))
```

目前預設：

- `free_gate_count = 3`
- `decay_rate = 0.35`

回傳欄位包括：

- `metabolic_burden_score`
- `gate_count`
- `complexity_penalty`

## 7. Temporal Scorer

實作位置：[benchmark_suite/temporal_scorer.py](benchmark_suite/temporal_scorer.py)

Temporal scorer 會計算或估算 rise time，並轉成 `temporal_score`。資料來源可能是：

- candidate 既有的 `rise_time`
- simulation trace 中的 `time` / `output`
- `logic_depth`
- `gate_count`

若可從 trace 找到 crossing threshold，會使用 trace；否則用 gate depth/數量估算。

## 8. Static Plausibility Evaluator

實作位置：[benchmark_suite/static_plausibility_evaluator.py](benchmark_suite/static_plausibility_evaluator.py)

此 evaluator 不跑 ODE，而是用 topology/Verilog 的靜態特徵估算可行性。它會檢查：

- part repetition
- logic depth
- 是否有過度複雜或可能降低可建構性的結構

輸出 component metric 為 `static_plausibility`。

## 9. Cello Constraint Evaluator

實作位置：[benchmark_suite/cello_constraint_evaluator.py](benchmark_suite/cello_constraint_evaluator.py)

此 evaluator 從 Cello mapping report、stdout/stderr、raw error log 或 topology 欄位提取：

- `orthogonality_score`
- `cello_assignment_score`
- `cello_buildable`
- `toxicity`
- `toxicity_score`

常見嚴重錯誤包含：

- not enough gates
- not enough orthogonal parts/repressors
- crosstalk / cross talk
- Cello mapping failure

`CelloWrapper` 也會呼叫 `evaluate_cello_constraints()`，把 constraint report 直接放進 topology。

## 10. Semantic Faithfulness

實作位置：[benchmark_suite/semantic_evaluator.py](benchmark_suite/semantic_evaluator.py)

`SemanticFaithfulnessEvaluator` 可用 LLM 檢查 Verilog 是否忠實滿足使用者需求，回傳：

- `semantic_faithfulness_score`
- `missed_edge_cases`

注意：目前 [benchmark_suite/benchmark_controller.py](benchmark_suite/benchmark_controller.py) 沒有直接呼叫 `score_semantic_faithfulness()`。`evaluate_candidate()` 會從 candidate 既有欄位讀取 `semantic_faithfulness_score` 與 `missed_edge_cases`，再放入總 report。若要讓 semantic evaluator 自動加入總分，需要在 controller 中新增 scorer 並調整權重。

## 11. Critic 門檻與路由

實作位置：[agents/critic_agent.py](agents/critic_agent.py)

Critic 使用 benchmark report 做 approve/reject 決策。重要 threshold：

| 常數 | 值 | 含義 |
| --- | ---: | --- |
| `PASS_SCORE_THRESHOLD` | 0.80 | 分數達此值才可能通過。 |
| `FAIL_SCORE_THRESHOLD` | 0.60 | 低於此值必須失敗。 |
| `METABOLIC_BURDEN_THRESHOLD` | 0.70 | 低於此值會強制不通過並路由到 Builder。 |
| `ROBUSTNESS_THRESHOLD` | 0.75 | 低於此值或出現 signal overlap 會強制不通過。 |
| `ORTHOGONALITY_THRESHOLD` | 0.20 | 低於或等於此值視為 Cello/UCF 嚴重問題。 |
| `SEMANTIC_FAITHFULNESS_THRESHOLD` | 0.90 | 低於此值且有 missed edge cases 時強制不通過。 |

路由原則：

- truth table、Boolean expression、需求語意、gate count、robustness、Cello buildability 類問題通常歸為 `LOGIC_ERROR`。
- 邏輯可接受但 mapping、part constraint、toxicity、ODE dynamics 不佳時歸為 `PART_ERROR`。
- 兩者皆有問題時歸為 `BOTH`。
- 都可接受時為 `NONE` 並 `is_approved=true`。

## 12. 指標同步到 SearchNode

[schemas/state.py](schemas/state.py) 的 `SearchNode.sync_evaluation_metrics()` 會把 topology 或 benchmark report 中的欄位同步到節點上，包含：

- `score`
- `metabolic_burden_score`
- `gate_count`
- `complexity_penalty`
- `robustness_score`
- `signal_to_noise_ratio`
- `monte_carlo_runs`
- `temporal_score`
- `rise_time`
- `orthogonality_score`
- `cello_assignment_score`
- `cello_buildable`
- `semantic_faithfulness_score`
- `missed_edge_cases`

此同步讓 UI、Critic feedback、failed attempt record 與 skill extraction 都能使用同一組欄位。

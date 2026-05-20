# 評估指標與驗證框架

本專案的評估邏輯集中在 [benchmark_suite](benchmark_suite) 與 [tools/ode_simulator.py](tools/ode_simulator.py)。工作流會先讓 Cello/ODE topology 帶有初步 metrics，再由 [benchmark_suite/benchmark_controller.py](benchmark_suite/benchmark_controller.py) 的 `evaluate_candidate()` 產生統一的 weighted total score。

本文列出目前程式實際使用的評分項目與計算公式，方便未來檢視或調整 scoring model。

## 1. 評估總覽

`evaluate_candidate(candidate)` 會呼叫下列 scorer：

- [benchmark_suite/functional_scorer.py](benchmark_suite/functional_scorer.py)：功能正確性。
- [benchmark_suite/kinetic_scorer.py](benchmark_suite/kinetic_scorer.py)：動力學與 Monte Carlo robustness。
- [benchmark_suite/static_plausibility_evaluator.py](benchmark_suite/static_plausibility_evaluator.py)：靜態可行性。
- [benchmark_suite/metabolic_scorer.py](benchmark_suite/metabolic_scorer.py)：gate count 與 metabolic burden。
- [benchmark_suite/temporal_scorer.py](benchmark_suite/temporal_scorer.py)：response/rise time。
- [benchmark_suite/cello_constraint_evaluator.py](benchmark_suite/cello_constraint_evaluator.py)：Cello assignment、orthogonality、toxicity、buildability。

所有 component score 進入總分前都會用下式限制在 0 到 1：

```text
clamp01(x) = max(0.0, min(1.0, x))
```

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

總分公式：

```text
weighted_total_score = round(
    0.22 * clamp01(functional)
  + 0.15 * clamp01(kinetic)
  + 0.08 * clamp01(static_plausibility)
  + 0.15 * clamp01(metabolic_burden)
  + 0.15 * clamp01(robustness)
  + 0.05 * clamp01(temporal)
  + 0.10 * clamp01(orthogonality)
  + 0.10 * clamp01(cello_assignment),
  10
)
```

其中 `robustness` 會優先使用 candidate 既有的 `robustness_score`；若缺少該欄位，才使用 kinetic scorer 回傳的 `robustness_score`。

等級定義：

- `Excellent`：`weighted_total_score >= 0.80`
- `Pass`：`0.60 <= weighted_total_score < 0.80`
- `Fail`：`weighted_total_score < 0.60`

## 3. Functional Scorer

實作位置：[benchmark_suite/functional_scorer.py](benchmark_suite/functional_scorer.py)

功能評估會嘗試從 candidate 讀取：

- `truth_table`
- `truth_table_or_logic_matrix`
- `logic_matrix`
- `verilog`
- `verilog_code`
- `verilog_draft`
- `min_on` 或 `on_min`
- `max_off` 或 `off_max`
- `fold_change`

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

實際計分公式：

```text
logic_compliance_score = correct_truth_table_rows / checked_truth_table_rows
```

若 truth table 沒有可檢查的 row：

```text
logic_compliance_score = 0.0
```

Fold change 會優先讀取 `fold_change`。若沒有 `fold_change`，但有 `min_on` 與 `max_off`：

```text
fold_change = min_on / max(max_off, 1e-9)
```

Fold change 分數：

```text
fold_change_score = clamp01(log1p(max(0.0, fold_change)) / log1p(100.0))
```

ON/OFF margin 分數：

```text
margin = min_on - max_off
scale = max(abs(min_on), abs(max_off), 1.0)
margin_score = clamp01(0.5 + 0.5 * margin / scale)
```

Functional 總分會平均所有可用的 component：

```text
functional_score = clamp01(mean(available_scores))

available_scores = [
  logic_compliance_score if truth_table and verilog are available,
  fold_change_score if fold_change can be computed,
  margin_score if min_on and max_off are available
]
```

若上述三項都無法計算，會 fallback：

```text
functional_score = clamp01(candidate.functional_score or candidate.score or 0.0)
```

## 4. Kinetic Scorer

實作位置：[benchmark_suite/kinetic_scorer.py](benchmark_suite/kinetic_scorer.py)

若 candidate 沒有 simulation inputs，也就是沒有 `verilog`、`verilog_code`、`gate_count`、`biokinetic_parameters` 任何一項，分數直接 fallback：

```text
kinetic_score = candidate.kinetic_score or candidate.score or 0.0
```

若 candidate 有 simulation inputs，會執行 noisy ODE response 評估。Monte Carlo 次數與 noise level：

```text
monte_carlo_runs = max(1, candidate.monte_carlo_runs or candidate.monte_carlo_samples or 20)
noise_level = candidate.noise_level or candidate.noise_fraction or 0.10
```

每次 noisy simulation 會得到 `on_value` 與 `off_value`。成功樣本集合如下：

```text
on_array = successful on_value samples
off_array = successful off_value samples
failed_runs = failed simulation count
```

訊號是否 collapse：

```text
min_signal = min(on_array)
max_noise = max(off_array)
collapsed = max_noise >= min_signal
```

SNR 與 SNR 分數：

```text
mean_on = mean(on_array)
mean_off = mean(off_array)
std_on = std(on_array)
std_off = std(off_array)
snr = max(0.0, (mean_on - mean_off) / max(std_on + std_off, 1e-9))
snr_score = clamp01(snr / (snr + 10.0))
```

Robustness 分數：

```text
success_rate = 0.0 if collapsed else successful_runs / monte_carlo_runs

robustness_score =
  0.0 if collapsed
  else 0.5 * success_rate + 0.5 * snr_score

if failed_runs > 0 and not collapsed:
  robustness_score = robustness_score * (monte_carlo_runs - failed_runs) / monte_carlo_runs

kinetic_score = clamp01(robustness_score)
```

若所有 noisy ODE robustness simulations 都失敗：

```text
kinetic_score = 0.0
robustness_score = 0.0
signal_to_noise_ratio = 0.0
```

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

Resource occupancy：

```text
rnap_occupancy = clamp01(1.0 - rnap_free / max(rnap_total, 1e-9))
ribosome_occupancy = clamp01(1.0 - ribosome_free / max(ribosome_total, 1e-9))
```

RNAP 與 ribosome resource factor：

```text
rnap_factor = rnap_free / (km_rnap + rnap_free)
ribosome_factor = ribosome_free / (km_ribosome + ribosome_free)
```

Hill repression。第 1 個 gene 不受上游 repressors 壓制，其餘 gene 使用上一層 protein：

```text
regulation[0] = 1.0
regulation[i] = leak + (1.0 - leak) / (1.0 + (protein[i - 1] / kd) ^ hill)
```

RHS：

```text
d_mRNA_i/dt =
  transcription_rate * regulation_i * rnap_factor
  - mrna_degradation_rate * mRNA_i

d_protein_i/dt =
  translation_rate * mRNA_i * ribosome_factor
  - protein_degradation_rate * protein_i
```

### 5.2 Solver

模擬器優先使用 SciPy：

1. `solve_ivp(..., method="BDF")`
2. `solve_ivp(..., method="Radau")`

若 SciPy 不存在或 solver 失敗，會使用內建 `_rk4_integrate()` 作為 fallback。RK4 更新式：

```text
k1 = rhs(t, y)
k2 = rhs(t + 0.5 * dt, max(y + 0.5 * dt * k1, 0.0))
k3 = rhs(t + 0.5 * dt, max(y + 0.5 * dt * k2, 0.0))
k4 = rhs(t + dt, max(y + dt * k3, 0.0))
y_next = max(y + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4), 0.0)
```

### 5.3 ODE Metrics

模擬後從 protein output 與 resource trace 計算：

```text
output = last protein trajectory
output_mean = mean(output)
output_std = std(output)
max_burden_nM = max(trace.burden_nM)
output_cv = output_std / max(output_mean, 1e-9)
signal_to_noise_ratio = output_mean / max(output_std, 1e-9)
```

若有多個 protein，dynamic margin 會比較最終 output 與上游 protein；若只有一個 protein，分母為 1：

```text
dynamic_margin =
  output_mean / (1.0 + max(upstream_protein_values))
```

Resource capacity factor：

```text
resource_capacity_factor = min(
  1.0,
  0.5 * rnap_total / default_rnap_total
  + 0.5 * ribosome_total / default_ribosome_total
)
```

Burden 與 toxicity penalty 使用 sigmoid：

```text
sigmoid_penalty(value, soft_limit, steepness) =
  1.0 / (1.0 + exp(clamp(steepness * (value - soft_limit), -60.0, 60.0)))

burden_penalty = sigmoid_penalty(max_burden_nM, burden_soft_limit, 0.00018)
toxicity_penalty = sigmoid_penalty(max_burden_nM, toxicity_threshold, 0.00022)
```

### 5.4 ODE Kinetic Score

ODE simulator 內部的 kinetic 分數與 [benchmark_suite/kinetic_scorer.py](benchmark_suite/kinetic_scorer.py) 的 noisy scorer 是兩個入口。ODE simulator 對 topology 寫入 `kinetic_score` 的公式如下：

```text
stability = 1.0 / (1.0 + output_cv)

if monte_carlo_terminal_output_cv exists:
  stability = stability * 1.0 / (1.0 + monte_carlo_terminal_output_cv)

margin = clamp01(dynamic_margin / 80.0)

resource_penalty =
  1.0 - 0.5 * (rnap_occupancy_max + ribosome_occupancy_max)

failure_penalty = 1.0 - monte_carlo_failure_rate

raw_kinetic_score =
  0.25 * stability
+ 0.20 * margin
+ 0.25 * burden_penalty
+ 0.20 * toxicity_penalty
+ 0.10 * clamp01(resource_penalty)

kinetic_score = clamp01(
  raw_kinetic_score
  * failure_penalty
  * (0.35 + 0.65 * resource_capacity_factor)
)

robustness_score = kinetic_score
```

Topology 原本若已有 `score`，ODE simulator 會將它與 ODE kinetic score 混合為 topology 的暫時分數：

```text
base_score = topology.score or (0.65 + topology_index * 0.02)
topology.score = clamp01(0.35 * base_score + 0.65 * kinetic_score)
```

### 5.5 Monte Carlo

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

參數 perturb 公式：

```text
sigma = abs(original_value) * noise_level
sampled_value = max(0.0, normal(original_value, sigma))
```

Monte Carlo 輸出：

```text
monte_carlo_failure_rate = failures / max(1, monte_carlo_samples)
monte_carlo_terminal_output_cv =
  std(terminal_outputs) / max(mean(terminal_outputs), 1e-9)
```

### 5.6 ODE 輸出欄位

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

此 scorer 會從 Verilog 或 candidate 的 `gate_count` 計算 logic gates 數量。支援 primitive gates：

- `and`
- `nand`
- `or`
- `nor`
- `xor`
- `xnor`
- `not`
- `buf`

Gate count 會計算兩種 Verilog primitive 寫法：

```text
gate_count =
  count(regex "\b(gate)\s*\(")
  + count(regex "\b(gate)\s+(optional_params)?instance_name\s*\(")
```

分數函式：

```text
excess_gates = max(0, gate_count - ideal_gate_limit)
metabolic_burden_score = exp(-decay_rate * excess_gates)
complexity_penalty = 1.0 - metabolic_burden_score
```

目前預設：

- `ideal_gate_limit = 3`
- `decay_rate = 0.35`

若沒有 Verilog source 或 `gate_count`，scorer 會跳過並給：

```text
metabolic_burden_score = 1.0
gate_count = 0
complexity_penalty = 0.0
```

若讀取或解析失敗：

```text
metabolic_burden_score = 0.0
gate_count = 0
complexity_penalty = 1.0
```

## 7. Temporal Scorer

實作位置：[benchmark_suite/temporal_scorer.py](benchmark_suite/temporal_scorer.py)

Temporal scorer 會計算或估算 rise time，並轉成 `temporal_score`。資料來源優先序：

1. candidate 既有的 `rise_time` 或 `response_time`
2. simulation trace 中的 `time` / `t` 與 `output` / `y` / `output_trace`
3. `logic_depth` / `depth`
4. `gate_count`

Trace rise time：

```text
rise_time = first time where output >= threshold_on
threshold_on = candidate.threshold_on or 0.5
```

Depth estimate：

```text
rise_time = max(0.0, depth * gate_delay_seconds)
gate_delay_seconds = candidate.gate_delay_seconds or 35.0
```

Temporal 分數：

```text
target = candidate.target_rise_time or 180.0
temporal_score = clamp01(exp(-max(0.0, rise_time - target) / max(target, 1e-9)))
```

若無法取得或估算 rise time，scorer 會跳過並給：

```text
temporal_score = 1.0
rise_time = None
```

## 8. Static Plausibility Evaluator

實作位置：[benchmark_suite/static_plausibility_evaluator.py](benchmark_suite/static_plausibility_evaluator.py)

此 evaluator 不跑 ODE，而是用 topology/Verilog 的靜態特徵估算可行性。它會檢查：

- part repetition
- logic depth
- 是否有過度複雜或可能降低可建構性的結構

重複元件數量：

```text
repeated_part_count = sum(count(part_id) - 1 for each part_id where count(part_id) > 1)
```

若沒有 `part_ids`、`assigned_parts`、`components`，會從 Verilog 註解或名稱 token 推估 part：

```text
// part: <id>
// component: <id>
// cello_constraint: <id>
promoter_<id>
rbs_<id>
terminator_<id>
repressor_<id>
```

Logic depth 優先使用 `logic_depth` 或 `depth`，否則從 Verilog dependency graph 推估；若沒有 Verilog，使用 `gate_count`。

Penalty 與分數：

```text
repeat_penalty = 1.0 - exp(-0.18 * repeated_part_count)
depth_excess = max(0, logic_depth - 4)
depth_penalty = 1.0 - exp(-0.22 * depth_excess)

structural_score = clamp01((1.0 - repeat_penalty) * (1.0 - depth_penalty))
```

若有 candidate 既有的 `plausibility_score` 且沒有任何 structural inputs：

```text
static_plausibility = clamp01(plausibility_score)
```

若同時有 `plausibility_score` 與 structural inputs：

```text
static_plausibility = clamp01(0.5 * plausibility_score + 0.5 * structural_score)
```

若沒有 explicit score：

```text
static_plausibility = structural_score
```

## 9. Cello Constraint Evaluator

實作位置：[benchmark_suite/cello_constraint_evaluator.py](benchmark_suite/cello_constraint_evaluator.py)

此 evaluator 從 Cello mapping report、stdout/stderr、raw error log 或 topology 欄位提取：

- `orthogonality_score`
- `cello_assignment_score`
- `cello_buildable`
- `toxicity`
- `toxicity_score`

Cello buildability：

```text
cello_buildable =
  coerce_bool(candidate.cello_buildable, default = mapping_status == "mapped")
```

嚴重 orthogonality constraint error 包含：

- not enough gates
- not enough orthogonal parts/repressors
- crosstalk / cross talk

Orthogonality 分數：

```text
if severe_constraint_error:
  orthogonality_score = 0.05
  cello_buildable = false
else if cello_buildable:
  orthogonality_score = candidate.orthogonality_score or 1.0
else:
  orthogonality_score = candidate.orthogonality_score or 0.25

orthogonality_score = clamp01(orthogonality_score)
```

Assignment score 會從 report 的 `gate_assignment_score`、`assignment_score`、`score` 或文字 regex 取出。Normalize 公式：

```text
if assignment_score is None:
  cello_assignment_score = 0.0
else if assignment_score > 1.0:
  cello_assignment_score = clamp01(assignment_score / 100.0)
else:
  cello_assignment_score = clamp01(assignment_score)
```

若 `cello_buildable=false` 且 normalized assignment 為 0，會嘗試使用 candidate 既有的 `cello_assignment_score`：

```text
if not cello_buildable and cello_assignment_score == 0.0:
  cello_assignment_score = candidate.cello_assignment_score or 0.0
```

Toxicity score：

```text
toxicity_score = 1.0 if toxicity is None else clamp01(1.0 - normalize_score(toxicity))
```

Cello constraint scorer 的 component score：

```text
cello_constraint_score =
  0.5 * orthogonality_score
+ 0.5 * cello_assignment_score

if not cello_buildable:
  cello_constraint_score = cello_constraint_score * 0.5

cello_constraint_score = clamp01(cello_constraint_score)
```

注意：總分中的 `orthogonality` 與 `cello_assignment` 不是直接使用 `cello_constraint_score`，而是分別使用：

```text
component_scores["orthogonality"] = clamp01(orthogonality_score)
component_scores["cello_assignment"] = clamp01(cello_assignment_score)
```

`cello_constraint_score` 會保留在 `details`，但目前沒有對應的 `SCORE_WEIGHTS["cello_constraints"]`，因此不直接進入 weighted total。

## 10. Semantic Faithfulness

實作位置：[benchmark_suite/semantic_evaluator.py](benchmark_suite/semantic_evaluator.py)

`SemanticFaithfulnessEvaluator` 可用 LLM 檢查 Verilog 是否忠實滿足使用者需求，回傳：

- `semantic_faithfulness_score`
- `missed_edge_cases`

LLM 必須回傳 JSON：

```json
{
  "score": 0.0,
  "missed_conditions": []
}
```

分數公式：

```text
semantic_faithfulness_score = clamp01(float(parsed.score))
missed_edge_cases = parsed.missed_conditions
```

若缺少 original prompt 或 Verilog：

```text
semantic_faithfulness_score = 0.0
missed_edge_cases = ["Semantic evaluation skipped because original prompt or Verilog is missing."]
```

若 LLM 回傳不能解析為 JSON：

```text
semantic_faithfulness_score = 0.0
missed_edge_cases = ["Semantic evaluator returned non-JSON output: ..."]
```

注意：目前 [benchmark_suite/benchmark_controller.py](benchmark_suite/benchmark_controller.py) 沒有直接呼叫 `score_semantic_faithfulness()`。`evaluate_candidate()` 只會從 candidate 既有欄位讀取：

```text
semantic_faithfulness_score = candidate.semantic_faithfulness_score or 1.0
missed_edge_cases = candidate.missed_edge_cases or candidate.missed_conditions or []
```

這些欄位會放入總 report，但不進入目前的 weighted total score。若要讓 semantic evaluator 自動加入總分，需要在 controller 中新增 scorer 並調整權重。

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

Critic 後處理會覆寫 LLM 可能不一致的判斷。主要公式與規則：

```text
if score < 0.60:
  is_approved = false

if is_approved and error_type != "NONE":
  is_approved = false

if error_type == "NONE" and not is_approved:
  error_type = "PART_ERROR"
```

強制失敗條件：

```text
metabolic_failed =
  metabolic_burden_score is not None
  and metabolic_burden_score < 0.70

robustness_failed =
  signal_overlap
  or (robustness_score is not None and robustness_score < 0.75)

cello_ucf_failed =
  cello_buildable is false
  or (orthogonality_score is not None and orthogonality_score <= 0.20)

semantic_failed =
  semantic_faithfulness_score is not None
  and semantic_faithfulness_score < 0.90
  and missed_edge_cases is not empty
```

若任一強制失敗條件成立：

```text
is_approved = false
if error_type in {"NONE", "PART_ERROR"}:
  error_type = "LOGIC_ERROR"
routing_target = "Builder"
```

Signal overlap 偵測：

```text
signal_overlap = report.collapsed or report.robustness_collapsed

if min_signal and max_noise exist in kinetic/robustness/monte_carlo details:
  signal_overlap = max_noise >= min_signal
```

一般路由：

```text
if error_type in {"LOGIC_ERROR", "BOTH"}:
  routing_target = "Builder"
else if error_type == "PART_ERROR":
  routing_target = "Translator"
else:
  routing_target = "Consolidator"
```

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

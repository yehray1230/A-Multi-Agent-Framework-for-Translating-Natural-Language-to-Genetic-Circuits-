# 工作流程與設定指南

本文件說明如何使用目前程式碼中的 multi-agent genetic circuit workflow。核心實作位於 [workflows/reflexion_controller.py](workflows/reflexion_controller.py)，互動式入口位於 [app.py](app.py)。

## 1. 安裝與啟動

建議使用 Python 3.11 以上環境。

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
streamlit run app.py
```

主要依賴列在 [requirements.txt](requirements.txt)，包含：

- `streamlit`
- `litellm`
- `pydantic`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `joblib`
- `chromadb`
- `requests`
- `beautifulsoup4`
- `sympy`

## 2. Streamlit UI 使用方式

啟動 [app.py](app.py) 後，可在側邊欄設定：

- 使用者需求 `user_intent`
- 宿主 `host_organism`
- compute budget
- API key、model name、API base
- 是否啟用 RAG、ODE simulation、multi-agent search、caching

UI 內有兩種主要執行路徑：

- Demo 路徑：`_run_demo_iteration()`，使用內建假資料，不需要外部 LLM。
- BYOK 路徑：`_run_byok_workflow()`，依使用者提供的 API key/model 建立 agent 並呼叫真正工作流。

## 3. 標準工作流

完整工作流由 [workflows/reflexion_controller.py](workflows/reflexion_controller.py) 的 `run_reflexion_workflow()` 控制。

### 3.1 初始化

若 `state.tree_nodes` 與 `state.active_frontier` 都是空的，controller 會建立 root 節點：

```text
SearchNode(node_id="root", search_mode="Exploration")
```

root 會被加入 `active_frontier`，之後 controller 使用 `active_frontier.pop(0)` 取出節點，因此目前搜尋行為接近 BFS。

### 3.2 RAG context

若目前模式是 `Exploration` 或 `Repair` 且有傳入 `skill_retriever`，controller 會呼叫：

```python
skill_retriever.retrieve_skills(state.user_intent, mode="Exploration")
skill_retriever.retrieve_skills(state.user_intent, mode="Repair")
```

結果寫入 `state.rag_context`，再由 [agents/builder_agent.py](agents/builder_agent.py) 與 [agents/translator_agent.py](agents/translator_agent.py) 放進 prompt。

`Exploitation` 模式目前不主動重新檢索 RAG context，而是偏向沿用父節點邏輯與 Critic feedback。

### 3.3 Builder

非 `Exploitation` 模式會執行 Builder：

```python
state = builder.run(state)
```

Builder 會輸出三個策略：

- `gate_count_optimization`
- `depth_optimization`
- `robustness_strategy`

成功後，proposal 會寫入目前節點的 `node.logic_proposals`，並同步到 `state.logic_proposals`。

若 Builder 失敗，`state.last_error` 會被設定，節點會標記為 `Dead_End`。

### 3.4 Translator

Translator 會把 `logic_proposals` 轉為 Verilog：

```python
state = translator.run(state)
```

驗證規則在 [agents/translator_agent.py](agents/translator_agent.py) 的 `_validate_verilog_ast()`。通過條件是：

- 有 `module` 與 `endmodule`
- 有 `input` 與 `output`
- 有 combinational logic：`assign` 或 primitive gates

不允許：

- `always`
- `reg`
- `#` delay

所有 proposal 都翻譯失敗時，`state.last_error = "ERROR: all Verilog translations failed."`。

### 3.5 Cello mapping

Cello wrapper 執行：

```python
state = cello_wrapper.run(state)
```

若 `cello_command` 未設定，會走 mock topology。若有設定，會呼叫外部 Cello command，並把 `{input_netlist}`、`{output_dir}`、`{ucf_path}` template 代入命令。

Mapping 結果寫入：

- `node.candidate_topologies`
- `state.candidate_topologies`

### 3.6 Biokinetic data mining

若有傳入 `data_miner`，controller 會執行：

```python
state = data_miner.run(state)
```

DataMiner 會為每個 topology 補入：

- host
- gene count
- RNAP/ribosome resource parameters
- transcription/translation/degradation parameters
- unit system
- source summary

預設參數在 [agents/data_miner_agent.py](agents/data_miner_agent.py) 的 `DEFAULT_BIOKINETIC_PARAMETERS`。

### 3.7 ODE simulation

ODE 模擬器執行：

```python
state = batch_ode_simulator.run(state)
```

每個 topology 會被補上：

- `ode_status`
- `kinetic_score`
- `robustness_score`
- `signal_to_noise_ratio`
- `monte_carlo_runs`
- `metrics_max_burden`
- `metrics_cv`
- `dynamic_margin`
- `resource_occupancy`
- `benchmark_report`

若 ODE solver 失敗，分數會歸零，並在 `benchmark_report.details` 記錄 `ode_failed`。

### 3.8 Benchmark

controller 會對每個 topology 呼叫：

```python
_apply_weighted_benchmark(topo)
```

此函式再呼叫 [benchmark_suite/benchmark_controller.py](benchmark_suite/benchmark_controller.py) 的 `evaluate_candidate()`，產生 weighted total score 與 component scores。

最高 `score` 的 topology 會成為：

- `node.best_topology`
- `state.best_topology`

節點也會呼叫 `node.sync_evaluation_metrics(best_topo)`，把常用評估欄位同步到 `SearchNode` 上。

### 3.9 Critic

Critic 執行：

```python
state = critic.run(state)
```

Critic 根據 proposal、best topology、benchmark report、failed attempt history 產生回饋與路由：

- `NONE`：通過。
- `LOGIC_ERROR`：邏輯、語意、gate count、robustness、Cello buildability 等需要 Builder 重新設計。
- `PART_ERROR`：邏輯大致可用，但 mapping、part、toxicity 或 ODE dynamics 需要物理層修正。
- `BOTH`：邏輯與物理實作都需要修正。

## 4. 搜尋模式

### Exploration

用途：產生新的設計方向。

程式行為：

- Builder temperature 設為 `0.7`。
- 若有 skill retriever，會以 `mode="Exploration"` 檢索 context。
- Builder 會重新產生三種 proposal。

### Repair

用途：修正邏輯或架構錯誤。

程式行為：

- Builder temperature 設為 `0.1`。
- 若有 skill retriever，會以 `mode="Repair"` 檢索 context。
- Builder prompt 會加入 Critic feedback。

### Exploitation

用途：沿用既有邏輯，集中調整 mapping/part constraints。

程式行為：

- 不執行 Builder。
- 子節點繼承父節點 `logic_proposals`。
- Translator 收到 exploitation mode 指令：不要改邏輯架構，只根據 feedback 調整 part/mapping constraints。

## 5. 分支規則

Critic 不通過後，controller 會先增加 `used_budget` 並呼叫 `_record_failed_attempt()`，再依錯誤類型分支。

| `error_type` | 下一步 |
| --- | --- |
| `LOGIC_ERROR` | 建立 `Repair` 子節點；若 budget 允許，也建立 `Exploration` 子節點。 |
| `BOTH` | 同 `LOGIC_ERROR`。 |
| `PART_ERROR` | 建立 `Exploitation` 子節點，並繼承父節點 proposals。 |
| `NONE` 但未通過 | 暫停，要求人工提供更多限制。 |

若連續同一種 `error_type` 達到 `MAX_CONSECUTIVE_ERROR_TYPE = 3`，工作流會暫停，避免無限迴圈。

## 6. Compute Budget 與人工介入

`DesignState.compute_budget` 控制最多可使用的修正/分支次數。當：

- `used_budget >= compute_budget`
- Critic 標記 `requires_human_input`
- Critic 標記 unrecoverable
- 連續同類錯誤太多
- frontier 全部耗盡

controller 會呼叫 `_pause_for_human_input()`，並設定：

- `state.requires_human_input = True`
- `state.pause_reason`
- `state.human_feedback_prompt`
- 節點狀態可能變成 `Needs_Human_Input`

UI 可利用這些欄位提示使用者補充限制、接受 trade-off，或選擇 fallback topology。

## 7. 輸出與可追蹤性

每個節點會保留：

- proposal
- Verilog
- candidate topologies
- best topology
- score 與 component metrics
- critic feedback
- failed attempt record
- parent-child lineage

因此可以從 `tree_nodes` 回看任一設計為何被接受、修正、或淘汰。

## 8. 測試

測試檔位於 [tests](tests)：

- [tests/test_reflexion_architecture.py](tests/test_reflexion_architecture.py)
- [tests/test_external_tools_and_skill_loop.py](tests/test_external_tools_and_skill_loop.py)
- [tests/test_physical_simulation_and_data_miner.py](tests/test_physical_simulation_and_data_miner.py)

可用以下命令執行：

```powershell
pytest
```

若環境沒有安裝 pytest，先安裝 [requirements-dev.txt](requirements-dev.txt)。

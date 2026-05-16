# 專案架構說明

本專案是一個以多代理人工作流為核心的遺傳電路設計原型。系統把自然語言需求轉成 Cello-compatible Verilog，再經由 Cello mapping、ODE/DAE 近似模擬、benchmark scoring 與 Critic 回饋，逐步搜尋可用的 genetic circuit topology。

主要入口有兩個：

- Streamlit 互動介面：[app.py](app.py)
- 程式化 Reflexion 工作流：[workflows/reflexion_controller.py](workflows/reflexion_controller.py)

## 1. 系統總覽

核心資料流如下：

1. 使用者在 Streamlit UI 輸入設計需求與 API/model 設定。
2. `DesignState` 保存整體任務狀態、搜尋樹、候選 topology、評估結果與人工介入狀態。
3. `BuilderAgent` 產生三種設計策略。
4. `TranslatorAgent` 將策略轉成 Cello-compatible Verilog。
5. `CelloWrapper` 產生或呼叫外部 Cello mapping 結果。
6. `DataMinerAgent` 補入 biokinetic parameters。
7. `BatchODESimulator` 模擬資源競爭、動態表現與 Monte Carlo robustness。
8. `benchmark_suite` 彙整功能、動力學、負擔、時間、Cello constraint 等分數。
9. `CriticAgent` 判斷是否通過，並以 `LOGIC_ERROR`、`PART_ERROR`、`BOTH`、`NONE` 導向下一個搜尋分支。
10. `ConsolidatorAgent` 產出最後摘要，`SkillExtractorAgent` 可把成功或失敗經驗寫成可重用 skill。

## 2. 主要目錄與責任

| 路徑 | 內容 |
| --- | --- |
| [app.py](app.py) | Streamlit UI、demo workflow、BYOK workflow、圖表與節點檢視器。 |
| [schemas/state.py](schemas/state.py) | `DesignState`、`SearchNode`、搜尋模式、節點狀態、錯誤類型，以及評估欄位同步。 |
| [workflows/reflexion_controller.py](workflows/reflexion_controller.py) | 真正的 multi-agent loop、frontier expansion、compute budget、人機介入暫停。 |
| [agents/](agents) | Builder、Translator、Critic、DataMiner、Consolidator、SkillExtractor。 |
| [tools/](tools) | Cello wrapper、ODE simulator、skill/vector retrieval。 |
| [benchmark_suite/](benchmark_suite) | 各類 evaluator 與 weighted total score controller。 |
| [exporters/](exporters) | Obsidian skill card 格式化與寫檔。 |
| [utils/](utils) | LLM 呼叫、tool calling、biokinetic unit conversion。 |
| [tests/](tests) | Reflexion 架構、外部工具、物理模擬與資料探勘測試。 |

## 3. 狀態模型

狀態模型定義在 [schemas/state.py](schemas/state.py)。

### `DesignState`

`DesignState` 是整個任務的共享狀態，重要欄位包括：

- `user_intent`：自然語言設計目標。
- `host_organism`：目標宿主，預設為 `Escherichia coli`。
- `tree_nodes`：以 `node_id` 為 key 的搜尋節點表。
- `active_frontier`：等待處理的節點 queue，目前採 `pop(0)`，行為接近 BFS。
- `current_node_id`：目前正在處理的節點。
- `compute_budget`、`used_budget`：控制搜尋分支數量。
- `rag_context`、`skill_library_context`：Builder/Translator 可用的歷史 skill context。
- `biokinetic_context`：DataMiner 補入的參數摘要。
- `logic_proposals`、`verilog_codes`、`candidate_topologies`、`best_topology`：目前全域候選結果。
- `failed_attempts`：Critic 拒絕後記錄的失敗嘗試。
- `requires_human_input`、`pause_reason`、`human_feedback_prompt`：工作流暫停並要求人工補充限制時使用。

### `SearchNode`

`SearchNode` 表示搜尋樹上的一個設計嘗試，重要欄位包括：

- `node_id`、`parent_id`、`children_ids`：搜尋樹 lineage。
- `search_mode`：`Exploration`、`Repair` 或 `Exploitation`。
- `status`：`Pending`、`Evaluated`、`Pass`、`Dead_End`、`Needs_Human_Input`。
- `logic_proposals`：Builder 輸出的三種策略。
- `verilog_codes`：Translator 產生的 Cello-compatible Verilog。
- `candidate_topologies`：Cello/ODE/benchmark 後的候選 topology。
- `best_topology`、`score`：此節點最佳結果。
- `critic_feedbacks`、`error_type`、`failed_attempts`：Critic 回饋與路由依據。
- `metabolic_burden_score`、`robustness_score`、`signal_to_noise_ratio`、`orthogonality_score`、`cello_assignment_score` 等：由 `sync_evaluation_metrics()` 從 topology 或 benchmark report 同步。

## 4. Agent 層

### BuilderAgent

實作位置：[agents/builder_agent.py](agents/builder_agent.py)

Builder 會要求 LLM 輸出一個 JSON object，且必須包含三個 top-level strategy：

- `gate_count_optimization`
- `depth_optimization`
- `robustness_strategy`

每個 strategy 需要包含 truth table 或 logic matrix、logic blueprint、Verilog draft 與 translator directives。Builder 也會把 `rag_context`、skill retrieval 結果、human constraints、critic feedback 放入 prompt。

### TranslatorAgent

實作位置：[agents/translator_agent.py](agents/translator_agent.py)

Translator 將 Builder proposal 轉成 Verilog，並用 `_validate_verilog_ast()` 做基本檢查。允許：

- `module` / `endmodule`
- `input`、`output`
- primitive gates：`and`、`or`、`not`、`nand`、`nor`、`xor`、`xnor`
- `assign`

禁止：

- `always`
- `reg`
- delay syntax `#`
- sequential logic、clock、memory、latch 類設計

當節點模式為 `Exploitation` 時，Translator 會收到額外指令：不要改變邏輯架構，只依 Critic feedback 調整 mapping/part constraints。

### CelloWrapper

實作位置：[tools/cello_wrapper.py](tools/cello_wrapper.py)

`CelloWrapper` 支援兩種模式：

- `cello_command is None`：回傳 mock topology，方便本地 demo 與測試。
- `cello_command` 有設定：把 Verilog 寫入 temporary directory，呼叫外部 Cello command，收集 stdout/stderr 與 mapping 結果。

Mapping 失敗時會分類常見錯誤，例如 `UCF_INCOMPATIBLE`、`VERILOG_SYNTAX_ERROR`、`UNSUPPORTED_GATE`、`PART_UNAVAILABLE`、`TIMEOUT`。

### DataMinerAgent

實作位置：[agents/data_miner_agent.py](agents/data_miner_agent.py)

DataMiner 會為 topology 補上 `biokinetic_parameters`。若有 `vector_retriever`，會從 local records 搜尋參數；否則使用 `DEFAULT_BIOKINETIC_PARAMETERS`。參數會經 [utils/unit_conversion.py](utils/unit_conversion.py) 正規化到 nM 與 seconds。

### BatchODESimulator

實作位置：[tools/ode_simulator.py](tools/ode_simulator.py)

模擬器以資源感知模型估算 mRNA/protein dynamics。主要特性：

- 使用 `WarmStartResourceSolver` 估算 free RNAP/ribosome。
- `ResourceAwareSimulation` 建立 ODE RHS。
- 優先使用 SciPy `solve_ivp` 的 `BDF` 與 `Radau`。
- SciPy 不可用或 solver 失敗時，使用內建 RK4 fallback。
- 支援 Monte Carlo perturbation、local cache，以及選用 joblib `Memory` 的 cache 介面。

### CriticAgent

實作位置：[agents/critic_agent.py](agents/critic_agent.py)

Critic 讀取 topology 與 benchmark report，輸出 JSON 決策：

- `is_approved`
- `error_type`
- `routing_target`
- `recoverable`
- `requires_human_input`
- `feedback`

程式內建多個硬性 threshold：

- `PASS_SCORE_THRESHOLD = 0.80`
- `FAIL_SCORE_THRESHOLD = 0.60`
- `METABOLIC_BURDEN_THRESHOLD = 0.70`
- `ROBUSTNESS_THRESHOLD = 0.75`
- `ORTHOGONALITY_THRESHOLD = 0.20`
- `SEMANTIC_FAITHFULNESS_THRESHOLD = 0.90`

即使 LLM 回傳可通過，只要 metabolic burden、robustness、Cello buildability 或 semantic faithfulness 違反門檻，程式仍會強制改成不通過並補上對應 guidance。

### ConsolidatorAgent

實作位置：[agents/consolidator_agent.py](agents/consolidator_agent.py)

Consolidator 目前負責把最佳 topology、分數與回饋整理成最後摘要，並保持狀態可供 UI 顯示。

### SkillExtractorAgent

實作位置：[agents/skill_extractor_agent.py](agents/skill_extractor_agent.py)

SkillExtractor 會從 `DesignState`、best topology、critic feedback、failed attempts 中擷取可重用經驗，寫入 `state.extracted_skills`，也可透過 Obsidian exporter 輸出 Markdown skill card。

## 5. Reflexion 搜尋控制

搜尋控制在 [workflows/reflexion_controller.py](workflows/reflexion_controller.py)。

初次執行時若沒有節點，controller 會建立：

```text
SearchNode(node_id="root", search_mode="Exploration")
```

每輪處理流程：

1. 檢查 `used_budget >= compute_budget`，超過則呼叫 `_pause_for_human_input()`。
2. 從 `active_frontier.pop(0)` 取出節點。
3. 依 `search_mode` 設定 LLM temperature 與 RAG context。
4. 非 `Exploitation` 模式先跑 Builder。
5. 跑 Translator。
6. 跑 CelloWrapper。
7. 選用 DataMiner。
8. 跑 BatchODESimulator。
9. 對每個 topology 套用 `evaluate_candidate()`。
10. 挑出最高分 topology，更新 node 與 state。
11. 跑 Critic。
12. 若通過，節點標記 `Pass` 並結束。
13. 若不通過，記錄 failed attempt，依 `error_type` 建立下一批子節點。

分支規則：

- `LOGIC_ERROR` 或 `BOTH`：建立 `Repair` 子節點；若 budget 足夠，也建立新的 `Exploration` 子節點。
- `PART_ERROR`：建立 `Exploitation` 子節點，並繼承原本的 `logic_proposals`。
- 連續同類錯誤達 `MAX_CONSECUTIVE_ERROR_TYPE = 3`：暫停並要求人工輸入。
- Critic 標示 unrecoverable 或 requires human input：暫停。

## 6. 記憶與 Graph RAG

Skill retrieval 實作在 [tools/skill_retriever.py](tools/skill_retriever.py)。目前使用 JSON skill library；程式中的預設 skill file path 仍是亂碼字串，專案根目錄實際存在的檔案是 [邏輯設計skill.json](邏輯設計skill.json)。若要正式啟用，建議在呼叫端明確傳入正確 `skill_file_path`。

`SkillRetriever` 排序會考慮：

- query term overlap
- `confidence_score`
- search mode bonus
- tags/backlinks graph bonus
- `recency_score`
- dead-end / avoid 類負向記憶 penalty

向量檢索另有簡化實作：

- [vector_db.py](vector_db.py)：`InMemoryVectorDB`
- [tools/vector_retriever.py](tools/vector_retriever.py)：`VectorRetriever`

## 7. UI 架構

Streamlit UI 在 [app.py](app.py)。主要區塊：

- `_render_sidebar()`：模型、API key、host、budget、workflow toggle。
- `_render_status_strip()`：完成、暫停、目前節點、最佳分數等狀態。
- `_render_byok_controls()`：BYOK/API 設定。
- `_render_pipeline()`：顯示 Builder、Translator、Cello、ODE、Critic 等步驟。
- `_render_chart_overview()`：節點分數與 topology 指標圖。
- `_render_tree_workspace()`：搜尋樹與節點清單。
- `_render_inspector()`：目前節點、Verilog、topology、Critic feedback 與 raw JSON。
- `_run_demo_iteration()`：不呼叫外部 LLM 的 demo 模式。
- `_run_byok_workflow()`：建立實際 agent/tool 並呼叫 `run_reflexion_workflow()`。

## 8. 已知注意事項

- 部分 Python source 內仍有亂碼字串，主要出現在 prompt guidance 與預設 skill file path；這不影響本次文檔重建，但可能影響 LLM prompt 可讀性。
- `CelloWrapper` 預設 mock mode 不會產生真正可建構 DNA 設計；若要接 Cello，需要設定 `cello_command` 與 UCF path。
- `BatchODESimulator` 是專案內建的近似模型，不等同完整生物物理驗證。
- `SemanticFaithfulnessEvaluator` 未被 `benchmark_controller.evaluate_candidate()` 直接列入 results；目前 semantic score 是從 candidate 既有欄位讀取並放入 report。

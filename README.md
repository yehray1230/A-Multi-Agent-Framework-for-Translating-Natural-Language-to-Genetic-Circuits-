# A Multi-Agent Framework for Translating Natural Language to Genetic Circuits

本專案是一個以多代理人工作流為核心的遺傳電路設計原型。它嘗試把自然語言需求轉換成 Cello-compatible Verilog，並透過 Cello mapping、資源感知 ODE 模擬、Monte Carlo robustness 測試與多維度 benchmark scoring，逐步搜尋較可行的 genetic circuit topology。

更完整的技術文件請參考：

- [ARCHITECTURE.md](ARCHITECTURE.md)：系統架構、狀態模型、Agent 與工具層。
- [WORKFLOW.md](WORKFLOW.md)：安裝、UI、Reflexion workflow、搜尋模式與分支規則。
- [EVALUATION_METRICS.md](EVALUATION_METRICS.md)：評分器、權重、ODE 模擬與 Critic 門檻。

## 核心目標

本系統不是只要求 LLM 直接寫出一段 Verilog，而是把「設計、翻譯、物理映射、動態模擬、批判修正」拆成可追蹤的工作流：

1. `BuilderAgent` 根據需求產生三種邏輯設計策略。
2. `TranslatorAgent` 將策略轉成 Cello-compatible combinational Verilog。
3. `CelloWrapper` 產生 mock topology 或呼叫外部 Cello 進行 mapping。
4. `DataMinerAgent` 補入宿主與 biokinetic parameters。
5. `BatchODESimulator` 估算動態表現、資源占用與 robustness。
6. `benchmark_suite` 彙整功能、動力學、負擔、時間與 Cello constraint 分數。
7. `CriticAgent` 判斷通過或修正方向，並驅動 `Exploration`、`Repair`、`Exploitation` 搜尋分支。

## 快速開始

建議使用 Python 3.11 以上。

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
streamlit run app.py
```

啟動後可在 Streamlit 介面中輸入設計需求、宿主、compute budget、model/API 設定，並切換 RAG、ODE simulation、multi-agent search 與 caching。

## 專案結構

| 路徑 | 說明 |
| --- | --- |
| [app.py](app.py) | Streamlit 互動介面、demo workflow、BYOK workflow、圖表與節點檢視器。 |
| [schemas/state.py](schemas/state.py) | `DesignState`、`SearchNode`、搜尋模式、節點狀態與評估欄位同步。 |
| [workflows/reflexion_controller.py](workflows/reflexion_controller.py) | Multi-agent Reflexion loop、frontier expansion、compute budget 與人工介入暫停。 |
| [agents](agents) | Builder、Translator、Critic、DataMiner、Consolidator、SkillExtractor。 |
| [tools](tools) | Cello wrapper、ODE simulator、skill/vector retrieval。 |
| [benchmark_suite](benchmark_suite) | 功能、動力學、代謝負擔、時間、Cello constraint 等 evaluator。 |
| [exporters](exporters) | Obsidian skill card 輸出。 |
| [tests](tests) | 架構、外部工具、物理模擬與資料探勘測試。 |

## 工作流概念

工作流的共享狀態由 [schemas/state.py](schemas/state.py) 定義。`DesignState` 保存使用者需求、搜尋樹、候選 topology、RAG context、評估結果與人工介入狀態；`SearchNode` 表示搜尋樹上的單次設計嘗試，包含 proposal、Verilog、candidate topologies、best topology、Critic feedback 與錯誤類型。

控制器位於 [workflows/reflexion_controller.py](workflows/reflexion_controller.py)。若尚未有節點，系統會建立 `root` 節點並以 `Exploration` 模式開始。每輪會依序執行 Builder、Translator、Cello、DataMiner、ODE simulator、benchmark、Critic。若 Critic 不通過，系統會依錯誤類型建立下一個搜尋分支：

- `LOGIC_ERROR`：建立 `Repair` 節點，必要時另開新的 `Exploration` 節點。
- `PART_ERROR`：建立 `Exploitation` 節點，保留原邏輯並集中改善 mapping 或物理約束。
- `BOTH`：同時視為邏輯與物理層問題，優先進入修正。
- `NONE`：代表設計可接受並結束搜尋。

## 生物學合理性的設計取向

本專案的模擬與評分不是把邏輯閘當成純數位電路處理，而是刻意把數位正確性、宿主負擔、資源競爭、動態穩定性、mapping 可建構性與雜訊 robustness 放進同一個評估框架。這些設計讓系統更接近合成生物學中的實際限制。

### 1. 不只檢查 Boolean truth table

[benchmark_suite/functional_scorer.py](benchmark_suite/functional_scorer.py) 會檢查 Verilog 是否符合 truth table、ON/OFF margin 與 fold change。這能避免設計只在語意上看似合理，實際邏輯卻不符合需求。

同時，Translator 在 [agents/translator_agent.py](agents/translator_agent.py) 中限制 Verilog 必須是 Cello-compatible combinational logic，禁止 `always`、`reg`、delay syntax、clock、memory 等不適合 Cello mapping 的語法。

### 2. 以資源感知 ODE 模擬近似細胞內動態

[tools/ode_simulator.py](tools/ode_simulator.py) 的 `BatchODESimulator` 不只用固定輸出分數，而是建立 mRNA/protein dynamics。狀態向量包含各 gene 的 mRNA 與 protein，並將下列因素納入 RHS：

- transcription rate
- translation rate
- mRNA degradation
- protein degradation
- Hill repression
- leak fraction
- free RNAP 與 free ribosome
- RNAP/ribosome occupancy

`WarmStartResourceSolver` 會估算 RNAP 與 ribosome 的可用量，讓多 gene circuit 的資源競爭反映在轉錄與轉譯速率上。這比只看 gate count 更接近細胞內 circuit 會互相搶奪轉錄與轉譯資源的現象。

### 3. 保留宿主與生物動力學參數來源

[agents/data_miner_agent.py](agents/data_miner_agent.py) 會為 topology 補入 biokinetic parameters。若有 local vector retriever，可從資料記錄中取得參數；否則使用保守預設值，例如 RNAP/ribosome total、transcription/translation rate、degradation rate、`kd`、Hill coefficient、leak fraction、burden/toxicity threshold。

這些參數會附帶 source、confidence 與 unit system。透過 [utils/unit_conversion.py](utils/unit_conversion.py) 正規化後，ODE 模擬使用一致的 nM 與 seconds 單位，降低混用單位造成的錯誤。

### 4. 用 Monte Carlo 測試參數不確定性

生物系統的 kinetic parameters 不會是精準常數。`BatchODESimulator` 支援 Monte Carlo perturbation，會對 transcription rate、translation rate、`kd`、Hill coefficient、leak fraction、degradation rates 等參數加入雜訊，並輸出：

- `monte_carlo_runs`
- `monte_carlo_failure_rate`
- `monte_carlo_terminal_output_cv`
- `signal_to_noise_ratio`
- `robustness_score`

這讓高分設計不只是單次模擬剛好成功，也需要在參數擾動下維持可分辨的 ON/OFF 狀態。

### 5. 把 metabolic burden 納入總分

[benchmark_suite/metabolic_scorer.py](benchmark_suite/metabolic_scorer.py) 會統計 Verilog 中的 primitive gate 數量，並用 exponential penalty 估算 metabolic burden：

```text
metabolic_burden_score = exp(-0.35 * max(0, gate_count - 3))
```

這個機制讓過度複雜的電路被扣分，因為更多 gates 往往意味著更多 genetic parts、更多轉錄轉譯負擔、更高宿主壓力，也可能降低可建構性與穩定性。

### 6. 檢查 Cello mapping 與 part constraints

[tools/cello_wrapper.py](tools/cello_wrapper.py) 可連接外部 Cello，並分類 mapping error，例如 UCF 不相容、Verilog syntax、unsupported gate、part unavailable、timeout。即使在 mock mode 下，topology 仍會保留 `cello_buildable`、`orthogonality_score`、`cello_assignment_score` 等欄位。

[benchmark_suite/cello_constraint_evaluator.py](benchmark_suite/cello_constraint_evaluator.py) 會從 Cello report、stdout/stderr 或 topology 欄位提取：

- orthogonality
- gate assignment quality
- toxicity
- buildability
- severe crosstalk 或 part shortage

這能讓系統避免只在抽象邏輯層最佳，卻無法用現有 parts 建構。

### 7. 用多目標加權，而非單一分數

[benchmark_suite/benchmark_controller.py](benchmark_suite/benchmark_controller.py) 將不同 evaluator 整合為 weighted total score：

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

因此，一個候選設計需要同時在邏輯正確性、動態行為、負擔、穩定性、Cello mapping 與 part compatibility 上表現合理，才會得到高分。

### 8. Critic 以硬門檻阻止不合理設計通過

[agents/critic_agent.py](agents/critic_agent.py) 不完全信任 LLM 的主觀判斷。即使 LLM 回答可通過，只要以下指標不達門檻，程式仍會強制 reject 並導向修正：

- `score < 0.60`：必須失敗。
- `metabolic_burden_score < 0.70`：導向 Builder 簡化設計。
- `robustness_score < 0.75` 或 ON/OFF signal overlap：導向 Builder 改善訊號 margin。
- `cello_buildable = false` 或 `orthogonality_score <= 0.20`：導向架構或 part composition 修正。
- `semantic_faithfulness_score < 0.90` 且存在 `missed_edge_cases`：導向語意與邏輯修正。

這些硬門檻讓系統比較不容易把看似漂亮但生物學上不穩定、不可建構或負擔過高的設計誤判為成功。

## 目前限制

- `CelloWrapper` 預設使用 mock topology；若要取得實際 Cello mapping，需要設定 `cello_command` 與 UCF/part library。
- ODE 模型是近似模型，適合做設計階段篩選，不等同實驗驗證。
- 目前 semantic faithfulness evaluator 已有實作，但 weighted benchmark controller 只讀取 candidate 既有的 semantic 欄位，尚未直接把 LLM semantic scorer 納入 scorer list。
- 部分 agent prompt guidance 仍有原始亂碼字串，建議後續清理，避免影響 LLM 回饋品質。

## 測試

```powershell
pytest
```

測試涵蓋 Reflexion 架構、外部工具與 skill loop、物理模擬與 DataMiner 行為。

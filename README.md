# 自然語言轉基因電路的多代理框架

這個專案提供一套將自然語言設計需求轉換為基因電路的實驗框架。系統會使用多個代理共同完成需求解析、邏輯提案、Cello 相容 Verilog 產生、拓樸 mapping、ODE 動態模擬、Monte Carlo 穩健性分析，以及 benchmark 評分。

主要介面由 Streamlit 提供，可用示範模式快速產生 deterministic 節點，也可使用 BYOK 模式接上自己的 LLM API key 執行完整工作流程。

## 快速開始

請使用 Python 3.11。

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
streamlit run app.py
```

啟動後，在側邊欄輸入設計需求，例如：

```text
設計一個只有在 A 高、B 低時才啟動 GFP 的基因電路。
```

接著可以選擇：

- **執行示範迭代**：建立或推進一個可重現的範例搜尋節點。
- **執行示範搜尋**：依照計算預算連續推進示範樹狀搜尋。
- **執行 BYOK 工作流程**：使用你提供的 API key 執行真實多代理流程。
- **匯出狀態 JSON**：下載目前 `DesignState`，方便保存或除錯。

## 系統流程

工作流程大致分為以下階段：

1. **需求輸入**：使用者以自然語言描述目標基因電路。
2. **RAG 檢索**：從技能庫或歷史規則中取得與設計相關的脈絡。
3. **Builder**：產生多個邏輯設計提案。
4. **Translator**：將邏輯提案轉換成 Cello 相容 Verilog。
5. **Cello Mapping**：評估 Verilog 是否能映射到可用基因元件。
6. **ODE 模擬**：估計動態反應、訊號裕度與穩健性。
7. **Benchmark 評分**：整合功能正確性、代謝負擔、正交性、元件相容性等分數。
8. **Critic 回饋**：判斷錯誤類型，並決定進入探索、修正或最佳化分支。
9. **Consolidator**：整理目前最佳拓樸與最終結果。

## 主要檔案

| 路徑 | 說明 |
| --- | --- |
| [app.py](app.py) | Streamlit 中文介面、示範流程與 BYOK 工作流程入口。 |
| [schemas/state.py](schemas/state.py) | `DesignState` 與 `SearchNode`，定義工作流程狀態。 |
| [workflows/reflexion_controller.py](workflows/reflexion_controller.py) | 多代理 Reflexion loop 與樹狀搜尋控制。 |
| [agents](agents) | Builder、Translator、Critic、DataMiner、Consolidator、SkillExtractor 等代理。 |
| [tools](tools) | Cello wrapper、ODE simulator、技能與向量檢索工具。 |
| [benchmark_suite](benchmark_suite) | 功能、動態、代謝負擔、穩健性、Cello constraint 等評分器。 |
| [exporters](exporters) | Obsidian skill card 匯出工具。 |
| [tests](tests) | Reflexion 架構、工具整合、模擬與資料探勘測試。 |

## 評分重點

Benchmark controller 會彙整多個 evaluator 的結果，常見指標包括：

- `functional`：布林邏輯與真值表是否符合需求。
- `kinetic`：動態反應與 ODE 模擬品質。
- `static_plausibility`：靜態設計合理性。
- `metabolic_burden`：元件數與生物負擔。
- `robustness`：Monte Carlo 擾動下的穩健性。
- `temporal`：反應時間與時間行為。
- `orthogonality`：元件正交性與 cross-talk 風險。
- `cello_assignment`：Cello mapping 與元件指派品質。

## BYOK 模式

BYOK 代表 Bring Your Own Key。你可以在側邊欄的 **BYOK 模型設定** 中設定：

- 服務提供者
- LiteLLM 模型名稱
- API key
- API base URL

API key 只會用於目前 Streamlit session，不會被匯出到狀態 JSON。

## 測試

```powershell
pytest
```

測試涵蓋 Reflexion 架構、外部工具與技能迴圈、物理模擬與資料探勘流程。

## 相關文件

- [ARCHITECTURE.md](ARCHITECTURE.md)：系統架構說明。
- [WORKFLOW.md](WORKFLOW.md)：AI Reflexion workflow 與執行流程。
- [EVALUATION_METRICS.md](EVALUATION_METRICS.md)：評分指標與 evaluator 設計。

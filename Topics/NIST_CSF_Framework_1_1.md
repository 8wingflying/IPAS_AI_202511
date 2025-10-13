# 🧭 NIST CSF（Cybersecurity Framework）網路安全框架說明

> 文件名稱：**Framework for Improving Critical Infrastructure Cybersecurity**  
> 版本：NIST CSF 2.0（最新版於 2024 年 2 月發布）  
> 制定機構：美國國家標準與技術研究院（NIST）  
> 適用對象：政府、企業、關鍵基礎設施與民間組織  

---

## 📘 一、什麼是 NIST CSF？

**NIST CSF（Cybersecurity Framework）** 是一套協助組織  
**識別、評估、管理與降低網路安全風險** 的治理框架。

它最初於 2014 年推出，用以強化美國「關鍵基礎設施（Critical Infrastructure）」的網路防護，  
後來被廣泛應用於全球企業資訊安全與風險治理領域。

> 💡 NIST CSF 的特點：  
> - 非強制性（Voluntary Framework）  
> - 可依組織規模與行業彈性採用  
> - 可與 ISO 27001、RMF、COBIT、CIS Controls 等標準整合

---

## 🎯 二、NIST CSF 的目標（Framework Goals）

1. **提升組織的網路安全成熟度與風險意識**  
2. **協助跨部門與外部供應鏈進行一致性風險管理**  
3. **建立可量化的安全績效衡量機制（Security Outcomes）**  
4. **推動「安全即治理（Security as Governance）」文化**

---

## 🧩 三、NIST CSF 的結構（Framework Structure）

NIST CSF 由 **三個核心要素（Core Components）** 組成：

| 要素 | 說明 |
|------|------|
| **Framework Core（核心功能）** | 五大功能 + 子類別（Categories & Subcategories）構成的通用模型 |
| **Implementation Tiers（實施層級）** | 組織的安全成熟度分級（Tier 1–4） |
| **Profiles（風險概況）** | 組織現況與目標安全狀態的對照表 |

---

## 🧭 四、五大核心功能（Five Core Functions）

| 核心功能 | 英文名稱 | 說明 |
|------------|------------|------|
| **1️⃣ 識別（Identify）** | Identify | 了解組織資產、環境、風險與治理結構。 |
| **2️⃣ 保護（Protect）** | Protect | 建立防禦措施以確保服務運作與資料安全。 |
| **3️⃣ 偵測（Detect）** | Detect | 監測與發現異常活動或網路威脅。 |
| **4️⃣ 回應（Respond）** | Respond | 在事件發生後迅速反應、通報與緩解。 |
| **5️⃣ 復原（Recover）** | Recover | 於事件後恢復業務運作與學習改進。 |

> 💡 這五大功能形成一個「安全事件生命週期循環（Cybersecurity Lifecycle）」。

---

## ⚙️ 五、五大功能下的 23 類別（Categories）

| 功能 | 類別（Category） | 說明 |
|------|----------------|------|
| **Identify** | Asset Management（ID.AM） | 管理硬體、軟體與資料資產 |
| | Business Environment（ID.BE） | 定義組織任務與安全依賴性 |
| | Governance（ID.GV） | 建立安全政策與法規遵循 |
| | Risk Assessment（ID.RA） | 執行風險識別與分析 |
| | Supply Chain Risk Management（ID.SC） | 管理供應鏈與第三方風險 |
| **Protect** | Identity Management & Access Control（PR.AC） | 控制使用者身分與授權 |
| | Awareness and Training（PR.AT） | 培養員工資安意識 |
| | Data Security（PR.DS） | 資料保護與加密 |
| | Information Protection Processes（PR.IP） | 制定資安標準作業程序 |
| | Maintenance（PR.MA） | 系統維護與更新 |
| | Protective Technology（PR.PT） | 技術性防護（防火牆、EDR、SIEM） |
| **Detect** | Anomalies and Events（DE.AE） | 偵測異常行為 |
| | Security Continuous Monitoring（DE.CM） | 建立持續監測機制 |
| | Detection Processes（DE.DP） | 事件偵測與通報流程 |
| **Respond** | Response Planning（RS.RP） | 事件回應計畫 |
| | Communications（RS.CO） | 事件溝通與外部協調 |
| | Analysis（RS.AN） | 威脅分析與根因調查 |
| | Mitigation（RS.MI） | 緩解與修復行動 |
| | Improvements（RS.IM） | 回饋與改進 |
| **Recover** | Recovery Planning（RC.RP） | 災難復原與業務持續性 |
| | Improvements（RC.IM） | 經驗回饋與改善 |
| | Communications（RC.CO） | 危機後溝通與公關協調 |

---

## 🧮 六、實施成熟度等級（Implementation Tiers）

| 層級 | 名稱 | 特徵 |
|------|------|------|
| **Tier 1 – Partial（部分實施）** | 非正式、無系統性風險管理流程 |
| **Tier 2 – Risk Informed（風險導向）** | 已識別主要風險但執行不一致 |
| **Tier 3 – Repeatable（可重複）** | 建立標準流程並定期改進 |
| **Tier 4 – Adaptive（自適應）** | 整合 AI、自動化與威脅情報反應 |

> 🚀 **建議目標：** 組織應至少達到 Tier 3「可重複」水準，  
> 並逐步邁向 Tier 4「自適應」以達到高成熟度。

---

## 🧾 七、Profile（風險概況）

**Profile（風險概況）** 是組織用來比較「目前狀態」與「目標狀態」的工具。  

| 類型 | 說明 |
|------|------|
| **Current Profile（現況概況）** | 組織目前的安全實施情形 |
| **Target Profile（目標概況）** | 預期達成的安全目標與改進計畫 |
| **Gap Analysis（差距分析）** | 找出現況與目標的落差，制定改善策略 |

---

## 🧠 八、NIST CSF 的核心原則

| 原則 | 中文說明 |
|------|----------|
| **Risk-Based** | 以風險為導向管理安全活動 |
| **Outcome-Focused** | 以成果（安全狀態）衡量，而非僅遵循規範 |
| **Flexible and Scalable** | 可依行業、規模、國家彈性採用 |
| **Collaborative** | 強調跨部門、跨產業協作 |
| **Continuous Improvement** | 推動持續改進與成熟度提升 |

---

## 🔗 九、CSF 與其他框架對照（Alignment）

| 框架 | 焦點 | 與 NIST CSF 關係 |
|------|------|----------------|
| **NIST RMF** | 資安與隱私風險治理流程 | CSF 為 RMF 的功能化實踐層，RMF 為治理層 |
| **NIST SP 800-53** | 控制項細節 | CSF 各類別可對應 SP 800-53 控制項 |
| **ISO/IEC 27001** | ISMS 資訊安全管理系統 | CSF 對應 Annex A 控制項 |
| **COBIT 2019** | IT 治理與控制 | CSF 提供更具操作性的網安維度 |
| **CIS Controls v8** | 技術安全控制基準 | 可用於落實 CSF Protect 與 Detect 功能 |
| **NIST AI RMF** | AI 系統風險治理 | AI RMF 延伸 CSF 的風險導向原則至 AI 領域 |

---

## 💡 十、實務應用範例（Example Applications）

| 行業 | CSF 應用場景 | 重點功能 |
|------|---------------|-----------|
| **金融業** | 管理詐欺與交易系統安全 | Identify、Protect、Respond |
| **醫療產業** | 電子病歷與病患隱私保護 | Protect、Detect、Recover |
| **教育機構** | 學生資料與遠端教學平台安全 | Identify、Protect、Detect |
| **政府單位** | 雲端與供應鏈資安治理 | Identify、Respond、Recover |

---

## 📊 十一、CSF、RMF、SP 800-53 對照表

| 項目 | CSF | RMF | SP 800-53 |
|------|------|------|------------|
| 性質 | 功能框架 | 管理流程 | 控制細節 |
| 主要焦點 | 安全成果（Outcome） | 風險流程（Process） | 控制措施（Controls） |
| 層級 | 高層策略 | 組織與系統層 | 技術實施層 |
| 使用時機 | 安全治理與成熟度評估 | 系統開發與授權 | 實際落地與稽核 |
| 關聯性 | 指引「做什麼」 | 說明「怎麼做」 | 提供「具體控制」 |

---

## ✅ 十二、NIST CSF 2.0 的重點更新（2024）

| 更新方向 | 說明 |
|-----------|------|
| **新增「Govern」功能** | 將治理（Govern）正式納入核心功能，使功能擴展為六項（Govern、Identify、Protect、Detect、Respond、Recover） |
| **強化供應鏈風險管理（C-SCRM）** | 將第三方安全納入標準化要求 |
| **明確連結隱私與 AI 風險** | 將 AI、隱私、韌性納入核心指引 |
| **改版架構更模組化、可量測** | 增加量化安全績效指標（Outcome Metrics） |

---

## 🧾 十三、CSF 的價值總結（Key Takeaways）

- 是最被全球採用的 **網路安全治理框架**。  
- 採 **風險導向、成果導向、可量測化** 的設計。  
- 能與 **RMF、SP 800-53、ISO 27001** 無縫整合。  
- 適合任何規模與產業的組織。  
- 最新版（CSF 2.0）已納入 **Govern（治理）** 與 **AI / 供應鏈風險** 概念。  

---

📎 **參考文件**
- NIST (2024). *Framework for Improving Critical Infrastructure Cybersecurity, Version 2.0*  
- NIST SP 800-37 Rev.2: *Risk Management Framework (RMF)*  
- NIST SP 800-53 Rev.5: *Security and Privacy Controls*  
- 官方網站：https://www.nist.gov/cyberframework  

---

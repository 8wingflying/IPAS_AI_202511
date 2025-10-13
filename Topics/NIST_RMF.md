# 🧭 NIST RMF（Risk Management Framework）風險管理框架 說明

> 出處：NIST SP 800-37 Rev. 2  
> 文件全名：**Risk Management Framework for Information Systems and Organizations: A System Life Cycle Approach for Security and Privacy**  
> 作者：National Institute of Standards and Technology（NIST）

---

## 📘 一、什麼是 NIST RMF？

**NIST RMF（風險管理框架）** 是美國國家標準與技術研究院（NIST）所制定的  
一套用於 **資訊系統安全與隱私風險管理** 的整合性流程。

它將「風險管理（Risk Management）」與「系統生命週期（System Development Life Cycle, SDLC）」結合，  
確保組織能在設計、開發、運行與維護各階段，持續管理安全與隱私風險。

---

## 🎯 二、主要目標（Objectives）

1. 將安全與隱私融入系統生命週期（SDLC）。  
2. 以風險為核心，協助決策者管理與授權資訊系統。  
3. 確保安全控制措施（Security Controls）具備可持續性與可稽核性。  
4. 建立全組織層級的安全治理文化（Security Governance Culture）。

---

## 🧩 三、NIST RMF 的七大步驟（Seven Core Steps）

| 步驟 | 名稱 | 中文說明 | 主要任務 |
|------|------|-----------|------------|
| **Step 1** | **Prepare** | 準備階段 | 建立風險治理結構、政策、角色與資源。 |
| **Step 2** | **Categorize** | 系統分類 | 根據資料與系統的重要性評定影響等級（高 / 中 / 低）。 |
| **Step 3** | **Select** | 控制選擇 | 根據 NIST SP 800-53 選擇適用的安全與隱私控制措施。 |
| **Step 4** | **Implement** | 控制實施 | 落實選定的控制措施於系統與組織流程中。 |
| **Step 5** | **Assess** | 控制評估 | 評估控制的實際效果與合規性。 |
| **Step 6** | **Authorize** | 授權運作 | 由高層決策者依風險可接受度核准系統上線（ATO）。 |
| **Step 7** | **Monitor** | 持續監控 | 持續追蹤風險變化、系統更新與安全事件。 |

---

## 🔄 四、RMF 運作流程圖（Lifecycle Integration）

```mermaid
flowchart LR
    A[Step 1: Prepare 準備] --> B[Step 2: Categorize 分類]
    B --> C[Step 3: Select 選擇]
    C --> D[Step 4: Implement 實施]
    D --> E[Step 5: Assess 評估]
    E --> F[Step 6: Authorize 授權]
    F --> G[Step 7: Monitor 監控]
    G --> B

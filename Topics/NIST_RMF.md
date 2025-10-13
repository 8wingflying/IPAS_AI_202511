# 📊 NIST RMF 七大步驟 × NIST AI RMF 四大功能 對照表（中英對照）

> 本表說明傳統資訊安全導向的 **NIST RMF（Risk Management Framework）**  
> 如何對應至人工智慧治理導向的 **NIST AI RMF（AI Risk Management Framework）**。  
> 兩者在結構與理念上高度一致，皆強調「風險導向、全生命週期、持續改進」。

---

## 🧭 對照總覽表

| NIST RMF 七大步驟 (Information System Focus) | 中文名稱 | 對應的 NIST AI RMF 四大功能 (AI System Focus) | 中文名稱 | 關聯說明 |
|----------------------------------------------|------------|-----------------------------------------------|-------------|-------------|
| **Step 1: Prepare** | 準備階段 | **Govern** | 治理（建立政策與角色） | 建立治理架構、風險容忍度與責任分工，確保管理文化與資源到位。 |
| **Step 2: Categorize** | 系統分類 | **Govern / Map** | 治理與映射 | 根據資料與系統重要性分類，映射出風險來源與影響範圍。 |
| **Step 3: Select** | 控制選擇 | **Map** | 映射（風險識別） | 選擇適合的安全與倫理控制措施，分析模型與資料風險。 |
| **Step 4: Implement** | 控制實施 | **Map / Measure** | 映射與衡量 | 實作控制措施，同時收集性能與可信度資料。 |
| **Step 5: Assess** | 效能評估 | **Measure** | 衡量（測試與評估） | 檢測系統風險指標、性能、公平性、隱私與安全性。 |
| **Step 6: Authorize** | 授權運作 | **Manage** | 管理（決策與授權） | 管理階層依據風險可接受度批准系統部署與使用。 |
| **Step 7: Monitor** | 持續監控 | **Manage** | 管理（持續監控與改進） | 監控 AI 系統行為與偏差，建立回饋與改進循環。 |

---

## 🔄 架構映射圖（RMF → AI RMF）

```mermaid
flowchart LR
    A[Prepare 準備] --> B[Categorize 分類]
    B --> C[Select 選擇]
    C --> D[Implement 實施]
    D --> E[Assess 評估]
    E --> F[Authorize 授權]
    F --> G[Monitor 監控]
    G --> B

    subgraph AI_RMF [NIST AI RMF 四大功能]
        G1[Govern 治理]
        G2[Map 映射]
        G3[Measure 衡量]
        G4[Manage 管理]
    end

    A -. 對應 .-> G1
    B -. 對應 .-> G1
    C -. 對應 .-> G2
    D -. 對應 .-> G2
    D -. 延伸 .-> G3
    E -. 對應 .-> G3
    F -. 對應 .-> G4
    G -. 對應 .-> G4

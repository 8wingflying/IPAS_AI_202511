### 🧭 NIST CSF（Cybersecurity Framework）網路安全框架

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

## 🔄 五、CSF 架構圖（Five Functions Integration）

```mermaid
flowchart LR
    A[Identify 識別] --> B[Protect 保護]
    B --> C[Detect 偵測]
    C --> D[Respond 回應]
    D --> E[Recover 復原]
    E --> A

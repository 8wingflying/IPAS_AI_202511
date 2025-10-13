#### 🧭 NIST SP 800-53：Security and Privacy Controls for Information Systems and Organizations

> 美國國家標準與技術研究院（NIST）  
> 文件代碼：SP 800-53 Rev.5  
> 最新版本發布：2020 年 12 月  
> 文件名稱：**Security and Privacy Controls for Information Systems and Organizations**  
> 中文名稱：**資訊系統與組織的安全與隱私控制指南**

---

## 📘 一、文件目的（Purpose）

NIST SP 800-53 的主要目的在於：
- 為 **聯邦政府、關鍵基礎設施及企業** 提供一套統一的 **安全與隱私控制措施（Security and Privacy Controls）**。  
- 支援 **NIST RMF（Risk Management Framework）** 第三步「Select」與第四步「Implement」階段。  
- 幫助組織在系統生命週期中 **防禦威脅、保護資訊、強化隱私與彈性**。

> 🚀 簡言之：  
> **NIST SP 800-53 = RMF 的「控制措施資料庫」**。

---

## 🧱 二、SP 800-53 在 RMF 架構中的角色

| RMF 步驟 | 功能 | 與 SP 800-53 的關係 |
|-----------|------|--------------------|
| Step 2: Categorize | 系統分類 | 決定所需控制層級（高 / 中 / 低） |
| **Step 3: Select** | **控制選擇** | 從 SP 800-53 控制集中挑選適用項目 |
| **Step 4: Implement** | **控制實施** | 實作 SP 800-53 規定的技術與管理控制 |
| Step 5: Assess | 控制評估 | 檢測所選控制項的實際效能 |

---

## ⚙️ 三、控制家族結構（Control Families）

SP 800-53 Rev.5 將安全與隱私控制分為 **20 大控制家族（Control Families）**。  
每一項控制家族（Control Family）以 **兩位字母代碼** 表示，包含多個具體控制項（Control）。

| 類別 | 控制家族 | 英文代碼 | 說明 |
|------|-----------|------------|------|
| **1️⃣ 管理類（Management）** | Program Management | **PM** | 整體安全與隱私治理計畫 |
|  | Risk Assessment | **RA** | 風險評估與威脅分析 |
|  | Planning | **PL** | 系統安全與隱私計畫 |
|  | Security Assessment and Authorization | **CA** | 控制評估與授權 |
|  | System and Services Acquisition | **SA** | 系統採購與供應鏈安全 |
| **2️⃣ 操作類（Operational）** | Awareness and Training | **AT** | 安全意識與訓練 |
|  | Incident Response | **IR** | 資安事件回應 |
|  | Configuration Management | **CM** | 組態管理 |
|  | Contingency Planning | **CP** | 災難復原與持續運作計畫 |
|  | Maintenance | **MA** | 系統維護與變更控制 |
|  | Media Protection | **MP** | 資料媒體保護與銷毀 |
|  | Physical and Environmental Protection | **PE** | 實體與環境安全 |
|  | Personnel Security | **PS** | 人員安全與身分驗證 |
|  | System and Information Integrity | **SI** | 系統完整性與防惡意軟體 |
| **3️⃣ 技術類（Technical）** | Access Control | **AC** | 存取控制與身分管理 |
|  | Audit and Accountability | **AU** | 稽核與可追蹤性 |
|  | Identification and Authentication | **IA** | 使用者身分識別與驗證 |
|  | System and Communications Protection | **SC** | 通訊安全與加密防護 |
|  | System and Information Integrity | **SI** | 系統完整性與錯誤偵測 |
| **4️⃣ 隱私類（Privacy）** | Privacy Controls | **PT** | 個資保護與透明性要求 |

> 📘 每個控制項通常以格式「AC-1、AC-2、AC-3」表示。  
> 例如：**AC-2（Account Management）** — 定義使用者帳號建立、維護與撤銷的安全要求。

---

## 🧩 四、控制項（Controls）結構範例

每個控制項包含：
1. **控制目標（Control Statement）**  
2. **控制增強項（Enhancements）**  
3. **適用等級（Baseline Impact Level: Low / Moderate / High）**  
4. **相關控制（Related Controls）**  
5. **實施指導（Supplemental Guidance）**

---

### 📖 範例：AC-2（Account Management）

| 項目 | 說明 |
|------|------|
| **控制目標** | 管理使用者帳號的建立、修改、停用與移除。 |
| **增強項** | AC-2(3): 自動停用長期未使用帳號。 |
| **適用等級** | Low, Moderate, High |
| **相關控制** | IA-2（身分驗證）、CM-5（組態變更） |
| **指導原則** | 應定期審查帳號權限並限制共用帳號。 |

---

## 🧠 五、SP 800-53 Rev.5 的新特點

| 新增重點 | 說明 |
|-----------|------|
| **1. 整合隱私控制（Privacy Controls）** | 與安全控制並列，確保個資與透明性要求。 |
| **2. 支援所有組織（非僅聯邦政府）** | 擴大應用於企業、學術與民間組織。 |
| **3. 模組化結構** | 控制項可依行業、風險、系統層級調整。 |
| **4. 與 NIST CSF、RMF、ISO 27001 對齊** | 增強跨框架兼容性。 |
| **5. 新增「供應鏈風險（Supply Chain Risk）」控制** | 因應第三方與外包風險增加。 |
| **6. 支援自動化（Automation-Ready）** | 提供控制識別碼，利於自動化稽核工具整合。 |

---

## 🧩 六、控制選擇流程（Control Selection Process）

```mermaid
flowchart TD
    A[1️⃣ 系統分類 FIPS 199] --> B[2️⃣ 選擇控制基準（Baseline Controls）]
    B --> C[3️⃣ 調整控制（Tailoring）]
    C --> D[4️⃣ 文件化控制選擇（SSP）]
    D --> E[5️⃣ 控制實施與驗證（Implement & Assess）]

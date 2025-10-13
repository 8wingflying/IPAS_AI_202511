# 🧭 NIST RMF Step 1：Prepare（準備階段）詳細說明

> 來源：NIST SP 800-37 Rev.2《Risk Management Framework for Information Systems and Organizations》  
> 準備階段（Prepare Step）是 RMF 的起點，確保組織在開始任何資訊系統風險管理前，  
> 已具備完善的「治理結構、角色職責、風險文化與資源基礎」。

---

## 🎯 一、準備階段的核心目標（Purpose）

**目的：**  
在風險管理開始前建立「組織級別的基礎設施與文化」，確保後續 RMF 流程能有效執行。

**主要任務：**
1. 建立組織層級的風險管理政策與責任分工。  
2. 確立風險容忍度（Risk Appetite）與影響門檻。  
3. 定義資訊系統的安全與隱私管理角色。  
4. 建立跨部門的溝通、稽核與回饋機制。  
5. 為系統分類與控制選擇（Step 2、Step 3）奠定基礎。

---

## 🧩 二、準備階段的兩大層級（Two Levels of Preparation）

| 層級 | 說明 | 主要參與者 |
|------|------|------------|
| **組織層（Organization-Level Preparation）** | 全公司層面的政策、架構與資源配置。 | CISO、CIO、法遵長、風險治理委員會 |
| **系統層（System-Level Preparation）** | 針對特定系統或專案建立風險與控制規劃。 | 系統管理員、安全工程師、專案經理 |

---

## ⚙️ 三、主要任務（Key Tasks in Prepare Step）

| 任務編號 | 任務名稱 | 中文說明 | 輸出成果 |
|-----------|-----------|------------|------------|
| **P-1** | Risk Management Roles | 定義風險管理職責與授權層級 | 組織風險責任矩陣 |
| **P-2** | Risk Management Strategy | 建立整體風險管理策略 | 風險容忍度聲明書 |
| **P-3** | Organization-wide Risk Assessment | 執行組織層風險評估 | 風險登錄表（Risk Register） |
| **P-4** | Common Control Identification | 識別可重複使用的通用控制（Common Controls） | 控制對照清單 |
| **P-5** | Information Types and Classification | 確認系統資訊類型與分類 | FIPS 199 分級報告 |
| **P-6** | Security and Privacy Framework Alignment | 對齊其他標準（ISO 27001、CSF、GDPR 等） | 對應矩陣 |
| **P-7** | Continuous Monitoring Strategy | 設定持續監控策略 | 監控政策與頻率表 |
| **P-8** | Stakeholder Engagement | 與內外部利害關係人協調 | 風險溝通計畫 |

---

## 🏛 四、準備階段的治理焦點（Governance Focus）

| 治理面向 | 說明 |
|-----------|------|
| **政策層（Policy Level）** | 建立組織資訊安全與隱私政策，明確風險管理原則。 |
| **角色與責任（Roles & Responsibilities）** | 指定系統擁有者（System Owner）、授權官（Authorizing Official, AO）、安全官（ISO）。 |
| **風險容忍度（Risk Appetite）** | 定義可接受的風險程度（例如：低風險業務容忍中等技術風險）。 |
| **跨部門整合（Integration）** | 將安全、隱私、法規遵循整合入組織治理流程。 |
| **資源配置（Resources）** | 為風險管理活動分配人力、預算與技術工具。 |

---

## 🧠 五、與後續步驟的關聯（Dependencies）

| 相關步驟 | 關聯說明 |
|-----------|-----------|
| **Step 2: Categorize** | 使用準備階段的分類依據與風險評估結果。 |
| **Step 3: Select** | 依據組織風險容忍度選擇控制措施。 |
| **Step 6: Authorize** | 準備階段建立的治理機制決定授權流程。 |
| **Step 7: Monitor** | 準備階段的監控策略決定持續監測頻率與責任。 |

---

## 🧩 六、實務應用範例（Practical Example）

### 🎓 教育機構範例
大學導入學籍資料管理系統：
- **P-1**：校方指定資安長（CISO）與系統擁有者。  
- **P-2**：訂定學生個資為「中高風險資訊」。  
- **P-4**：沿用全校通用的身分驗證控制。  
- **P-7**：設定每季隱私稽核與弱點掃描。

### 💰 金融機構範例
銀行導入雲端客戶分析平台：
- **P-2**：定義金融資料屬高風險類別。  
- **P-3**：建立風險登錄表並進行第三方供應商評估。  
- **P-5**：對客戶資料分類與加密。  
- **P-8**：與法遵部門協調跨國資料流動合規。

---

## 🤖 七、延伸：AI RMF 中的對應關聯

| NIST RMF：Prepare | 對應 NIST AI RMF 功能 | 說明 |
|--------------------|------------------------|------|
| 建立風險治理文化 | **Govern（治理）** | 建立 AI 治理結構與倫理政策 |
| 明確風險責任與決策流程 | **Govern** | 確保 AI 系統問責（Accountability） |
| 定義風險容忍度與指標 | **Map（映射）** | 釐清 AI 系統風險來源與影響 |
| 建立持續監控機制 | **Manage（管理）** | 支援 AI 模型的持續改進與審查 |

---

## 🧾 八、準備階段常見文件（Outputs / Artifacts）

| 文件名稱 | 中文說明 |
|------------|------------|
| 風險管理策略（Risk Management Strategy） | 定義組織風險治理方式與容忍度。 |
| 風險責任矩陣（Risk Responsibility Matrix） | 清楚劃分風險角色與任務。 |
| 資訊分類報告（Information Categorization Report） | 依 FIPS 199 對系統資訊分類。 |
| 監控策略文件（Monitoring Strategy Document） | 定義持續監控頻率、指標與責任人。 |
| 利害關係人溝通計畫（Stakeholder Engagement Plan） | 確保內外部人員風險認知一致。 |

---

## 🧩 九、AI 系統應用示例：生成式 AI 專案的準備階段

| 任務 | 說明 |
|------|------|
| **定義角色責任** | 指定模型擁有者（Model Owner）、資料保護官（DPO）、倫理審查委員會。 |
| **建立風險治理策略** | 明定生成式 AI 禁止用於歧視、仇恨或錯誤資訊生成。 |
| **資料分類** | 將訓練資料依隱私等級分類（公開、內部、敏感）。 |
| **合規對齊** | 對應 AI Act、GDPR、ISO 42001 等國際標準。 |
| **風險溝通** | 建立模型透明度報告與外部審查機制。 |

---

## ✅ 十、重點摘要（Key Takeaways）

- 「準備階段」是整個 RMF 成功的基礎。  
- 它將組織從「零散防護」轉為「系統治理」。  
- 核心精神是：**文化、責任、資源、政策** 四要素先行。  
- 在 AI 風險治理中，對應 **Govern（治理）功能**，  
  是建構「可信任與問責的 AI 系統」的第一步。

---

📎 **參考文件**
- NIST SP 800-37 Rev.2 (2018): *Risk Management Framework for Information Systems and Organizations*  
- NIST SP 800-53 Rev.5 (2020): *Security and Privacy Controls*  
- NIST AI 100-1 (2023): *Artificial Intelligence Risk Management Framework*  
- 官方網站：https://csrc.nist.gov/publications/detail/sp/800-37/rev-2/final  

---

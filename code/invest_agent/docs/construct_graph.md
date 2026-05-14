# 1 概览与关键依据（来自报告）

* 调查样本显示个人（自然人）投资者在样本中占绝对多数，而机构样本较少但资金规模显著（样本：个人问卷 ~84,807 份，机构问卷 409 份；机构可用平均资金规模 ≈ 546.9 亿元）。因此在网络上应设定**少量机构 + 大量散户**的结构性不均衡。
* 投资者多“相信自己的判断”（71.6% 表示“自己分析决定”），但外界信息（网上/媒体 47.3%、朋友推介 45.7%）对决策也有显著影响——说明**并非所有节点都完全跟随他人，但有明显的群体影响通道**。这支持把“羊群系数”建模为随机正值、且与节点类型（散户/机构）与信息来源有关。

以上事实决定了：网络要是**有向带权**（注意关注是单向的、强弱不同），节点类型至少分为“散户（retail）”与“机构（institution）”，并且要让机构在网络中具有较大“影响力/被关注度”的先验。

---

# 2 网络结构（顶层设计）

设定总节点数为 (N)（用户可自定，典型仿真取 (N=2000) 到 (N=20000)）。令机构占比 (p_{inst})，散户占比 (1-p_{inst})。

**建议默认参数（可调整）**

* (p_{inst}=0.05)（5%）：机构数量少但每个权重与资金影响较大。理由：报告机构样本远少于个人，但机构资金规模大，5% 是在“可模拟多数散户行为同时保留足够机构节点用于网络效应”间的折衷。可按需要设为 1%–10%。

**节点标记**

* 每个节点 (i) 有类型 (T_i \in {\text{retail},\text{inst}})。

**有向带权关注图 (G=(V,E,W))**

* 边 (i \to j) 表示“i 关注 j” —— 在决策时会部分采纳 j 的行为/意见。边权 (w_{ij}) 即“羊群系数”，定义域 ([0,\infty))。权重为越大表示越强的跟随倾向（更高的采纳比重）。

**生成拓扑原则（直观）**

1. **异质度/偏好**：散户更倾向关注机构（报告显示散户常受媒体/他人影响且更依赖渠道信息），因此边从散户指向机构的概率及权重大于散户→散户。
2. **优先连接（preferential attachment / in-degree）**：被关注节点（尤其机构）更容易被新的或其他节点关注（现实中名气/资金/业绩形成“吸引力”）。
3. **稀疏性**：实用网络应当稀疏（平均出度 (k_{out}) 小于 N），以便计算与解释。

---

# 3 羊群系数（边权）分布设计（数学化）

权重非负且通常具长尾分布（少数强影响，多数弱影响）。建议使用 **Gamma 分布** 或 **对数正态（lognormal）** 两类之一（均在 ([0,\infty))）。下面给出分类型参数推荐（可直接采样）——这些参数反映：散户更容易被机构影响、机构相互间影响力较强但更“理性”且波动性不同。

设 (w_{ij}) 的分布按边端点类型分类：

1. **散户（i=retail） → 机构（j=inst）**（散户倾向关注机构）

   * 建议：Gamma(k=2.0, θ=1.5)；均值 (=k\theta=3.0)，方差 (=k\theta^2=4.5)。
   * 说明：期望偏高、方差中等，表示散户向机构的跟随强度总体偏大但有差异。

2. **散户 → 散户**（同类之间的跟风，通常较弱）

   * 建议：Gamma(k=1.0, θ=0.5)；均值 0.5，方差 0.25。
   * 说明：多数散户之间的羊群系数小，偶尔会有较强跟随（通过长尾或混合机制产生极端值）。

3. **机构 → 散户**（机构关注散户通常较少；但有时机构会监测散户情绪）

   * 建议：Gamma(k=0.8, θ=0.25)；均值 ≈0.2，方差 ≈0.05。
   * 说明：小平均值，但保留一定噪声。

4. **机构 → 机构**（机构间互相参考、协同或抄袭策略）

   * 建议：Gamma(k=1.2, θ=0.8)；均值 ≈0.96，方差 ≈0.768。
   * 说明：均值接近 1，但方差更高以允许个别机构高度跟随或被跟随。

**可选对数正态替代**：若你希望更明显的长尾（极端大影响），对上面均值做 lognormal 建模，例如 (w\sim\text{LogNormal}(\mu,\sigma)) 并使 (\exp(\mu+\sigma^2/2)=) 目标均值。

**归一化/剪切**：在决策更新时，通常会将邻居影响按比例归一化（例如将所有 (w_{ij}) 对被关注者的影响量化为比例权重），或引入“基线自我权重” (w_{ii}) 来反映“相信自己判断”的比重（报告中 71.6% 自信自己判断，因此可设置平均自重较高，见下）。

---

# 4 节点属性（分布与数值化）

每个节点 (i) 有属性向量 (\theta_i = (\alpha_i,\lambda_i,m_i))：

* (\alpha_i)：风险厌恶系数（风险厌恶；定义域 ([0,\infty))；数值越大越风险厌恶）。
* (\lambda_i)：损失厌恶系数（prospect theory 中常用；(\lambda>1) 表示损失的痛苦 > 同等收益）。
* (m_i)：过度自信程度（定义域 ([1,\infty))），表示对自身信息/信号精度的高估倍数（1=不过度自信）。

依据报告中的群体差异（机构资金更大、更偏向专业、散户教育程度高但依然容易受媒体/朋友影响），建议如下分布（可直接用于仿真初始化）：

## 4.1 散户（retail）参数建议

* 风险厌恶 (\alpha)：**Gamma(k=2.0, θ=1.0)** → 均值 2.0（中等偏高），方差 2.0。理由：报告显示多数个人偏向“产生一定收益、可承担一定风险”，但也有较保守群体，因而分布应有中等右尾。
* 损失厌恶 (\lambda)：**Truncated Normal(mean=2.25, sd=0.5, lower=1.0)** → 多数在 1.5–3.0 区间，均值 ~2.25（文献/行为金融常见值约 2）。理由：报告指出亏损引发焦虑（10%–50% 区间显著），说明损失敏感性较高。
* 过度自信 (m)：**1 + LogNormal(mu= -1.3, sigma=0.35)**（使 m 的均值约 1.2–1.3）或更直接：**Shifted-LogNormal(shift=1, μ=0.2, σ=0.25)** → 平均 ~1.25。理由：虽多数相信自己判断（71.6%），但报告同时显示朋友/媒体影响显著 —— 既有自信又有跟风，故自信系数应略高于 1。

## 4.2 机构（inst）参数建议

* 风险厌恶 (\alpha)：**Gamma(k=1.2, θ=0.6)** → 均值 ≈0.72（明显低于散户），机构整体更风险中性/承受力强。理由：机构资金大、专业性高，报告显示机构更偏权益配置（股票偏好高）。
* 损失厌恶 (\lambda)：**Truncated Normal(mean=1.8, sd=0.4, lower=1.0)** → 较散户略低。机构更注重风险管理，但仍有损失规避。
* 过度自信 (m)：**1 + LogNormal(mu= -2.2, sigma=0.12)** → 均值 ≈1.1。机构也会过度自信但幅度通常小于散户（他们更依赖研究、风控流程）。

---

# 5 决策采纳模型（如何把“羊群系数”与节点属性映射到行为）

一个简单且常用的决策更新（离散时间 (t)）规则：

每个节点 (i) 在时刻 (t) 拥有信念/意见 (s_i(t))（例如“买入强度”或“保有/赎回概率”的实数）。更新可以用加权平均 + 自重的形式：

[
s_i(t+1) = \underbrace{\beta_i \cdot s_i^\text{self}(t)}*{\text{自我信息}} ;+; (1-\beta_i)\cdot\frac{\sum*{j\in\mathcal{N}*i^\text{in}} w*{ij}, s_j(t)}{\sum_{j\in\mathcal{N}*i^\text{in}} w*{ij} + \epsilon}
]

* (\beta_i\in(0,1)) 为“自我权重” —— 可设为与过度自信 (m_i) 正相关（例如 (\beta_i = \frac{m_i}{m_i + c})），并且与报告中的“自己判断比例 71.6%”对应总体水平（可校准使平均 (\beta) ≈0.7）。
* 分母加 (\epsilon) 防止除零。
* (s_i^\text{self}(t)) 是节点基于自身信息/信号的私有判断（可按其风险厌恶与损失厌恶从当前价格/收益信号计算）。
* 当 (w_{ij}) 很大时，邻居意见占主导（羊群效应）。

**关于 (w_{ii})**：你也可以把自信映射为自环权重 (w_{ii})，并把更新写成带归一化的邻居和自环权重的线性组合。

---

# 6 网络生成伪代码（Python 风格伪代码，便于直接实现）

```python
import numpy as np
import networkx as nx

def generate_investor_network(N=5000, p_inst=0.05, k_out_mean=10):
    # 1. assign types
    n_inst = max(1, int(N * p_inst))
    types = np.array(['inst']*n_inst + ['retail']*(N-n_inst))
    np.random.shuffle(types)

    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, type=types[i])
    # 2. sample node attributes according to type (use recommended distributions)
    for i in range(N):
        if types[i]=='retail':
            alpha = np.random.gamma(shape=2.0, scale=1.0)
            lam = max(1.0, np.random.normal(2.25, 0.5))
            m = 1.0 + np.random.lognormal(mean=0.2, sigma=0.25)
        else:
            alpha = np.random.gamma(shape=1.2, scale=0.6)
            lam = max(1.0, np.random.normal(1.8, 0.4))
            m = 1.0 + np.random.lognormal(mean=-2.2, sigma=0.12)
        G.nodes[i].update({'alpha':alpha,'lambda':lam,'m':m})

    # 3. create edges: for each node sample out-degree from Poisson(k_out_mean),
    #    choose targets with preferential attachment to in-degree and type bias.
    in_deg = np.ones(N)  # initial small attractiveness
    for i in range(N):
        k = np.random.poisson(lam=k_out_mean)
        for _ in range(k):
            # candidate sampling: mix of preferential attachment and type bias
            # probability vector proportional to (in_deg) * type_bias
            type_bias = np.where(types=='inst', 3.0, 1.0)  # prefer institutions
            probs = in_deg * type_bias
            probs[i] = 0.0  # avoid self-target (or allow for self-loop separately)
            probs = probs / probs.sum()
            j = np.random.choice(N, p=probs)
            # sample weight w_ij according to type pair using Gamma parameters above
            pair = (types[i], types[j])
            if pair==('retail','inst'):
                w = np.random.gamma(2.0, 1.5)
            elif pair==('retail','retail'):
                w = np.random.gamma(1.0, 0.5)
            elif pair==('inst','inst'):
                w = np.random.gamma(1.2, 0.8)
            else:
                w = np.random.gamma(0.8, 0.25)
            G.add_edge(i,j,weight=float(w))
            in_deg[j] += 1.0
    return G
```

> 注：上述伪代码中的超参数（如 type_bias、k_out_mean、Gamma 参数）按上文建议给出，实际仿真时应根据目标规模与校验结果调参。

---

# 7 校准建议（如何用报告数据检验/微调）

* **机构比例 (p_{inst})**：可参考样本（机构问卷 409 份 vs 个人样本量），并结合你想模拟的“机构渗透率”（现实中机构在交易量/资金上占比远高于数量占比），默认 5% 可作为起点。
* **自重 (\beta)** 的均值：报告“71.6% 自己分析决定” → 使仿真总体上 (\mathbb{E}[\beta]\approx0.7)。可把 (\beta_i) 映射为 (\beta_i = \mathrm{sigmoid}(a\cdot(m_i - 1) + b)) 并调 (a,b) 使均值合适。
* **朋友/媒体影响参数**：报告里的 47.3%（网上/媒体）与 45.7%（朋友）可以用来设定散户→散户与散户→机构 边的平均出度与平均权重比例（即使平均被关注权重占总权重的比重与这两个数字相符）。

---

# 8 将 LLM 用于节点决策的映射建议（少量 LLM 驱动大市场）

* **策略**：把 LLM 在每个“投资者类型”上做为**策略模板 + 随机化参数化器**：

  * 对于每个节点 (i)，构造一个 prompt 模板包含：节点属性（(\alpha_i,\lambda_i,m_i)）、历史价格序列摘要、邻居意见摘要（按 (w_{ij}) 加权平均），再让 LLM 输出 “买/卖/观望 强度” 或概率。
  * 为减少 LLM 数量：把节点按**属性簇**聚类（例如按风险厌恶+损失厌恶+过度自信分 10–50 个簇），每个簇使用同一 LLM 实例或相同 prompt 模板与不同随机种子来生成行为，从而用“少量 LLM”模拟大量节点。
* **信息聚合**：每次仿真步把每个节点的邻居信号压缩为统计量（加权均值、方差、top-k 意见），把这些作为 LLM 的输入而不是每一位邻居的完整信息，以节约 token 与运算。
* **频次**：令 LLM 只在**重要决策时刻**（如价格大幅变动、新闻事件、或每隔 T 步）来生成策略，其它时刻使用快速的线性更新规则（第 5 节）推进，以混合模拟精度与计算成本。

---

# 9 输出示例（参数表 — 便于直接复制）

| 项目              |                                   建议分布/参数 | 说明 / 参考   |
| --------------- | ----------------------------------------: | --------- |
| 机构比例 (p_{inst}) |                      0.05（可 0.01–0.10 调整） | 机构少量但资金大。 |
| 散户→机构 (w)       |             Gamma(k=2.0, θ=1.5), mean=3.0 | 散户更倾向关注机构 |
| 散户→散户 (w)       |             Gamma(k=1.0, θ=0.5), mean=0.5 | 同类影响较弱    |
| 机构→机构 (w)       |            Gamma(k=1.2, θ=0.8), mean≈0.96 | 机构间互相参考   |
| 机构→散户 (w)       |            Gamma(k=0.8, θ=0.25), mean≈0.2 | 机构少量关注散户  |
| 散户 (\alpha)     |                  Gamma(2.0,1.0), mean=2.0 | 风险厌恶偏中偏高  |
| 散户 (\lambda)    | TruncNormal(mean=2.25, sd=0.5, lower=1.0) | 损失厌恶明显    |
| 散户 (m)          |    1 + LogNormal(μ=0.2,σ=0.25), mean≈1.25 | 过度自信略高    |
| 机构 (\alpha)     |                 Gamma(1.2,0.6), mean≈0.72 | 更风险中性     |
| 机构 (\lambda)    |  TruncNormal(mean=1.8, sd=0.4, lower=1.0) | 损失厌恶略低    |
| 机构 (m)          |     1 + LogNormal(μ=0.1,σ=0.12), mean≈1.1 | 轻微过度自信    |

（以上表格参数为推荐起点；仿真时请按目标校准。）
from msi.msi import MSI
from diff_prof.diffusion_profiles import DiffusionProfiles
import multiprocessing
import numpy as np
import pickle
import networkx as nx

msi = MSI()
msi.load()

dp = DiffusionProfiles(alpha = 0.9, max_iter = 1000, tol = 1e-06, weights = {'down_biological_function': 4, 'indication': 3.5, 
'biological_function': 6, 'up_biological_function': 2, 'protein': 4, 'drug': 3.5}, num_cores = 4, 
save_load_file_path = "results/")
# dp = DiffusionProfiles(alpha = 0.8595436247434408, max_iter = 1000, tol = 1e-06, weights = {'down_biological_function': 4.4863053901688685, 'indication': 3.541889556309463, 
# 'biological_function': 6.583155399238509, 'up_biological_function': 2.09685000906964, 'protein': 4.396695660380823, 'drug': 3.2071696595616364}, num_cores = 4, 
# save_load_file_path = "results/") #原始参数
# dp = DiffusionProfiles(alpha = 0.7, max_iter = 100, tol = 1e-06, weights = {'down_biological_function': 1, 'indication': 1, 
# 'biological_function': 1, 'up_biological_function': 1, 'protein': 1, 'drug': 1}, num_cores = 4, 
# save_load_file_path = "results/")
    
# dp.get_initial_M(msi)  # ✅ 必须先构建 M
# dp.calculate_diffusion_profile(msi, selected_drugs_and_indications = ["DB01098"]) #可以指定药物或疾病

#这个自动构建M
dp.calculate_diffusion_profiles(msi) #处理所有 drug 和 indication 节点


# #-----------------------计算药物和疾病（其实是细胞类型）扩散谱间的相似度------------------
# import numpy as np
# import os
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle
# import re

# # 从结果文件夹加载扩散谱
# def load_diffusion_profiles(file_path):
#     """
#     从指定路径加载扩散谱。
#     """
#     diffusion_profiles = dict()
#     drugs = []
#     indications = []
    
#     # 正则表达式：匹配以 C 后面跟数字的文件名
#     indication_pattern = re.compile(r"^C\d+")
    
#     # 读取所有药物和疾病的扩散谱文件
#     for filename in os.listdir(file_path):
#         if filename.endswith("_p_visit_array.npy"):
#             drug_or_indication = filename.split("_p_visit_array.npy")[0]
#             diffusion_profile = np.load(os.path.join(file_path, filename))
#             diffusion_profiles[drug_or_indication] = diffusion_profile
            
#             # 根据文件名判断是药物还是疾病
#             if indication_pattern.match(drug_or_indication):  # 如果文件名以 C 后跟数字，则认为是疾病
#                 indications.append(drug_or_indication)
#             else:
#                 drugs.append(drug_or_indication)

#     return diffusion_profiles, drugs, indications

# # 计算药物与疾病之间的扩散谱相似度（余弦相似度）
# def calculate_diffusion_similarity(diffusion_profiles, drugs, indications):
#     """
#     计算药物与疾病之间的扩散谱相似度（余弦相似度）。
#     """
#     similarity_results = []

#     for drug in drugs:
#         for indication in indications:
#             # 提取药物和疾病的扩散谱
#             drug_diffusion = diffusion_profiles.get(drug)
#             indication_diffusion = diffusion_profiles.get(indication)

#             if drug_diffusion is not None and indication_diffusion is not None:
#                 # 计算余弦相似度
#                 similarity = cosine_similarity([drug_diffusion], [indication_diffusion])[0][0]
#                 similarity_results.append((drug, indication, similarity))
#             else:
#                 similarity_results.append((drug, indication, None))  # 如果未找到扩散谱，则为None

#     return similarity_results

# # 保存计算结果到CSV文件
# def save_similarity_results_to_csv(similarity_results, output_dir):
#     """
#     保存药物与疾病之间的扩散谱相似度到指定目录的CSV文件。
#     """
#     # 创建DataFrame
#     df = pd.DataFrame(similarity_results, columns=["Drug", "Indication", "Similarity"])
    
#     # 保存为CSV文件
#     csv_file_path = os.path.join(output_dir, "drug_indication_similarity.csv")
#     df.to_csv(csv_file_path, index=False)
#     print(f"Similarity results saved to {csv_file_path}")


# # 加载药物和疾病的扩散谱
# print("Loading diffusion profiles...")
# diffusion_profiles, drugs, indications = load_diffusion_profiles("results/")
    
# # 计算药物与疾病之间的扩散谱相似度
# print("Calculating diffusion similarities between drugs and indications...")
# similarity_results = calculate_diffusion_similarity(diffusion_profiles, drugs, indications)

# # 保存结果到CSV文件
# print("Saving results to CSV...")
# save_similarity_results_to_csv(similarity_results, "MM_outputs/")




#------------------------------药物综合得分-----------------------------
import pandas as pd
import numpy as np
from pathlib import Path

# ========== 配置 ==========
COMM_PATH = "MM/MCDM_joint_AIA_own_colors_pearson_summed.txt"
SIM_PATH  = "MM_outputs/drug_indication_similarity.csv"
MAP_PATH  = "MM/2_indication_to_protein_converted.tsv"
OUTPUT    = "MM_outputs/drug_overall_score.csv"

# λ：直击肿瘤 vs 重塑微环境 的权衡（0~1）
LAMBDA = 0.5

# 通讯强度归一化的分位点（更稳健；避免极值牵引）
LOW_Q, HIGH_Q = 0.05, 0.95

# 肿瘤细胞类型的名字（与你文件一致）
TUMOR_CT = "Malignant"

# ========== 工具函数 ==========
def quantile_minmax_norm(values, low_q=0.05, high_q=0.95):
    """分位点截断 + 线性映射到[0,1]，更稳健。"""
    v = pd.Series(values).astype(float)
    if len(v) == 0:
        return v  # 空
    lo, hi = v.quantile(low_q), v.quantile(high_q)
    # 处理极端：当所有值都相同
    if hi - lo == 0:
        # 若全为0 => 直接返回0；否则全部置为1（避免除0）
        return pd.Series([0.0 if (v.iloc[0] == 0) else 1.0] * len(v), index=v.index)
    v_clip = v.clip(lower=lo, upper=hi)
    return (v_clip - lo) / (hi - lo)

# ========== 1) 读映射：C1..C10 -> 细胞类型 ==========
map_df = pd.read_csv(MAP_PATH, sep="\t")
# 兼容不同列名大小写/空格
map_df.columns = [c.strip() for c in map_df.columns]
# 需要的两列：node_1 (C1..C10) 和 node_1_name (细胞类型名)
if not {"node_1", "node_1_name"}.issubset(set(map_df.columns)):
    raise ValueError("映射文件缺少列 node_1 / node_1_name")

code2name = dict(zip(map_df["node_1"].astype(str), map_df["node_1_name"].astype(str)))

# ========== 2) 读药物-细胞类型相似度 ==========
sim_df = pd.read_csv(SIM_PATH)
sim_df.columns = [c.strip() for c in sim_df.columns]
if not {"Drug", "Indication", "Similarity"}.issubset(set(sim_df.columns)):
    raise ValueError("相似度文件需要包含列：Drug, Indication, Similarity")

# 把 Indication 从 C1..C10 映射为细胞类型名称；若本来就是名称则保留原值
sim_df["Indication_name"] = sim_df["Indication"].astype(str).map(code2name).fillna(sim_df["Indication"].astype(str))
# 保留 [0,1] 范围；如有越界可按需裁剪
sim_df["Similarity"] = sim_df["Similarity"].astype(float).clip(0, 1)

# 每个(Drug, 细胞类型) 可能有多行（不同来源/重复），取平均更稳健
sim_avg = (sim_df
           .groupby(["Drug", "Indication_name"], as_index=False)["Similarity"]
           .mean())

# 为了后续便捷，做成：每药物一行，列=细胞类型；缺失补0
sim_pivot = (sim_avg
             .pivot_table(index="Drug", columns="Indication_name", values="Similarity", fill_value=0.0))

# ========== 3) 读细胞通讯强度（pearson），只取 Target=Malignant ==========
comm_df = pd.read_csv(COMM_PATH, sep="\t")
comm_df.columns = [c.strip() for c in comm_df.columns]
need_cols = {"pearson", "Sender", "Target"}
if not need_cols.issubset(set(comm_df.columns)):
    raise ValueError(f"细胞通讯文件需要包含列：{need_cols}")

comm_sel = comm_df[(comm_df["Target"] == TUMOR_CT) & (comm_df["Sender"] != TUMOR_CT)][["Sender","pearson"]].copy()
# 该文件名含 'pearson_summed'，但仍以 Sender 聚合以防重复
# comm_by_sender = (comm_sel.groupby("Sender", as_index=False)["pearson"].sum())

# 归一化到 [0,1] 作为权重 Ĉ_tumor,i
#comm_sel["C_norm"] = quantile_minmax_norm(comm_sel["pearson"], LOW_Q, HIGH_Q)
#softmax归一化
x = comm_sel["pearson"].clip(lower=0).to_numpy(float)  # 允许负值时先截断
tau = 1.0  # 温度，越小越“尖锐”，越大越接近均匀
w = np.exp(x / tau)
comm_sel["C_norm"] = w / w.sum()   # >0 且 sum=1；你的第二项就成加权平均

# 建立 {细胞类型 -> Ĉ} 字典
C_norm_map = dict(zip(comm_sel["Sender"].astype(str), comm_sel["C_norm"].astype(float)))

# ========== 4) 计算 S_drug,tumor 以及第二项的加权平均 ==========
# 确定肿瘤细胞是否在相似度矩阵的列中；若没有则作为0处理
if TUMOR_CT not in sim_pivot.columns:
    # 也尝试看看 mapping 里是否有 TUMOR_CT 的 code（比如 Ck 对应 Malignant）
    tumor_codes = [k for k, v in code2name.items() if v == TUMOR_CT]
    # 如果 Indication 是 code 存在但未被 pivot 成列（极少数特殊情况），此处可再并一次；简单起见统一补0
    sim_pivot[TUMOR_CT] = 0.0

# 第二项只在“有通讯权重的细胞类型”上做加权；并与相似度矩阵列对齐
senders = list(C_norm_map.keys())
# 某些 Sender 在相似度表里可能没有列（例如没算该细胞类型的相似度），统一补0列
for ct in senders:
    if ct not in sim_pivot.columns:
        sim_pivot[ct] = 0.0

# 取出矩阵视图
S_tumor = sim_pivot[TUMOR_CT].copy()                       # Series: Drug -> S_{drug,tumor}
S_env_mat = sim_pivot[senders].copy()                       # DataFrame: Drug x Sender
C_vec = pd.Series({ct: C_norm_map[ct] for ct in senders})   # Series: Sender -> Ĉ

# 计算第二项：加权平均 (保证落在[0,1])
# 分子：每药物与各 Sender 的相似度 * 该 Sender 的 Ĉ，再对 Sender 求和
num = (S_env_mat * C_vec).sum(axis=1)
den = C_vec.sum()
S_env_weighted = num / (den if den != 0 else 1.0)

# ========== 5) 组合主分数 ==========
score_main = LAMBDA * S_tumor + (1.0 - LAMBDA) * S_env_weighted

# ========== 6) 导出结果 ==========
out = pd.DataFrame({
    "Drug": sim_pivot.index,
    "S_drug_tumor": S_tumor.values,
    "S_env_weighted": S_env_weighted.values,
    "lambda": LAMBDA,
    "Score": score_main.values
}).sort_values("Score", ascending=False)

# 同时附上用于第二项的 Ĉ（方便审计）
for ct in senders:
    out[f"S_{ct}"] = sim_pivot.loc[out["Drug"], ct].values  # 该药物对该细胞类型的相似度（0~1）
out["Cnorm_sum"] = den
out.to_csv(OUTPUT, index=False)

print(f"[OK] 计算完成，共 {len(out)} 个药物。")
print(f"已保存到: {OUTPUT}")
print("Top-10:\n", out.head(10))


#-----------------画出扩散谱的路径图（论文风格）
# ===== Diffusion subgraph export (concise version) =====
import os, json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
from msi.msi import MSI

# ---- config ----
drug, cell = "DBMM_COM", "C13"
k_drug, k_cell = 20, 35
out_dir = "MM_outputs"
os.makedirs(out_dir, exist_ok=True)
pref = os.path.join(out_dir, f"{drug}_vs_{cell}_d{k_drug}_c{k_cell}")

# ---- load graph + diffusion ----
msi = MSI()
msi.load()
G, idx2n, n2idx = msi.graph, msi.idx2node, msi.node2idx
d = np.load(f"results/{drug}_p_visit_array.npy")
c = np.load(f"results/{cell}_p_visit_array.npy")

# ---- select nodes ----
top_drug_idx = d.argsort()[-k_drug:]
top_cell_idx = c.argsort()[-k_cell:]
top_idx = np.unique(np.r_[top_drug_idx, top_cell_idx])
nodes = [idx2n[i] for i in top_idx]

subG = G.subgraph(nodes).copy()
subG = subG.subgraph([n for n in subG if subG.degree(n) > 0]).copy()

top_drug_set = set(top_drug_idx.tolist())
top_cell_set = set(top_cell_idx.tolist())

# ---- annotate nodes ----
for n in subG:
    i = n2idx[n]
    subG.nodes[n].update(dict(
        type=G.nodes[n].get("type", "unknown"),
        drug=float(d[i]),
        cell=float(c[i]),
        sum=float(d[i] + c[i]),
        is_top_drug=(i in top_drug_set),
        is_top_cell=(i in top_cell_set),
    ))

# ---- export node table ----
pd.DataFrame(
    [{"node": n, **subG.nodes[n]} for n in subG]
).to_csv(f"{pref}_nodes.csv", index=False)

# ---- export directed/original edge table ----
edges_df = pd.DataFrame(
    [{"source": u, "target": v, **e} for u, v, e in subG.edges(data=True)]
)
edges_df.to_csv(f"{pref}_edges.csv", index=False)

# ---- export undirected edge table (deduplicated) ----
undirected_rows = []
seen = set()

for u, v, e in subG.edges(data=True):
    key = tuple(sorted([str(u), str(v)]))
    if key in seen:
        continue
    seen.add(key)
    row = {"source": key[0], "target": key[1], **e}
    undirected_rows.append(row)

edges_undirected_df = pd.DataFrame(undirected_rows)
edges_undirected_df.to_csv(f"{pref}_edges_undirected.csv", index=False)

# ---- export adjacency ----
pd.DataFrame(
    nx.to_numpy_array(subG, nodelist=list(subG)),
    index=list(subG),
    columns=list(subG)
).to_csv(f"{pref}_adjacency.csv")

# ---- export graph formats ----
nx.write_graphml(subG, f"{pref}.graphml")
with open(f"{pref}.json", "w") as f:
    json.dump(json_graph.node_link_data(subG), f, indent=2)

# ---- plot (with full type->color mapping) ----
color_map = {
    "drug": "orange",
    "protein": "skyblue",
    "biological_function": "lightgreen",
    "indication": "red",
}
node_colors = [color_map.get(subG.nodes[n].get("type", "unknown"), "gray") for n in subG]

pos = nx.spring_layout(subG, seed=42)
fig, ax = plt.subplots(figsize=(8, 6))
nx.draw(
    subG,
    pos,
    ax=ax,
    node_size=[300 + 3000 * subG.nodes[n]["sum"] for n in subG],
    node_color=node_colors,
    with_labels=True,
    font_size=8
)
fig.tight_layout()
fig.savefig(f"{pref}.png", dpi=300)
plt.show()


#-------------每个节点的访问概率
import numpy as np
import pandas as pd
from msi.msi import MSI

# 加载 MSI 图对象及索引映射
msi = MSI()
msi.load()
idx2node = msi.idx2node
G = msi.graph

# 加载扩散谱向量（药物的访问频率分布）
visit_array = np.load("results/acefylline_p_visit_array.npy")

# 收集每个节点的详细信息
records = []
for idx, score in enumerate(visit_array):
    node = idx2node[idx]
    node_type = G.nodes[node].get("type", "unknown")
    node_name = G.nodes[node].get("name", "")
    records.append((node, node_type, node_name, score))

# 排序并保存为 CSV
df = pd.DataFrame(records, columns=["node_id", "node_type", "node_name", "visit_score"])
df.sort_values("visit_score", ascending=False, inplace=True)
df.to_csv("acefylline_diffusion_vector.csv", index=False)


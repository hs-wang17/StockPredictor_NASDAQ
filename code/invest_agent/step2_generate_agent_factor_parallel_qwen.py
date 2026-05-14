import json
import pandas as pd
from copy import deepcopy
import logging
import os
import re
import random
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from srcs.agent_api_inference import agent_api_inference

# --- 配置 ---
DATA_DIR = "/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/"
FACTOR_DIR = os.path.join(DATA_DIR, "daily_factor_data/")
OUTPUT_DIR = os.path.join(DATA_DIR, "invest_agent_data/")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "agent_factors.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "dag_agent_simulation.log")
NODE_LIST_FILE = os.path.join(OUTPUT_DIR, "graph_data/investor_node_list.json")

BATCH_SIZE = 10
MAX_WORKERS = 10  # 根据你的API配额调整并发数

logger = logging.getLogger("DAGAgentSim")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler(LOG_FILE, mode="a")
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

def generate_agent_prompt(node, node_list, pi_record, day_df, stock_order, date):
    """保留原始 Prompt 所有符号与逻辑"""
    stock_list = []
    for idx, row in day_df.iterrows():
        stock_code = row['stock']
        factors = row.drop(['stock', 'date']).dropna().to_dict()
        stock_list.append({"stock_code": stock_code, "factors": factors})
    stocks_json = json.dumps(stock_list, ensure_ascii=True)

    neighbors_info = []
    for neighbor in node.get("targets", []):
        neighbor_id = neighbor["id"]
        neighbor_pi_vector = []
        # 由于是按 Agent 串行，当前 Agent 可以看到之前已完成 Agent 的决策
        if neighbor_id in pi_record and date in pi_record[neighbor_id]:
            neighbor_pi_vector = pi_record[neighbor_id][date]
        
        eta = neighbor.get("weight", 1.0) # 默认为1.0避免KeyError
        neighbors_info.append({"id": int(neighbor_id), "pi_vector": neighbor_pi_vector, "eta": float(eta)})
    
    neighbors_json = json.dumps(neighbors_info, ensure_ascii=True)
    node_type_en = "retail investor" if node["type"] == "retail" else "institutional investor"

    prompt_template = f"""
You are a Chinese A-share investor. Your type is: {node_type_en}.

Your attributes:
- Risk aversion alpha={node['alpha']}, representing your risk preference in the CARA utility framework.
- Loss aversion lambda={node['lambda']}, representing that a loss causes lambda times more discomfort than a gain.
- Overconfidence m={node['m']}, representing that you believe your private information is amplified by m times.

Your neighbors and their decision vectors (for herd effect, but you don't need to fully mimic them):
{neighbors_json}

Stock factors for all stocks today (to focus on when making your decision, do not blindly follow others' decisions, do not output identical values):
{stocks_json}

Stock order: {stock_order}

According to optimal investment theory and factor models, your investment proportion pi for each stock (fraction of total capital invested in each stock today):
Output a vector in stock order, each element in [0,1], separated by commas, without any explanation or text. For example: 0.1,0.2,0.05,0.3,0.35
"""
    return prompt_template

def safe_parse_pi_vector(pi_str, expected_length):
    """保留原始解析逻辑与随机填充机制"""
    try:
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", str(pi_str))
        if not matches:
            values = [random.random() for _ in range(expected_length)]
        else:
            values = []
            for match in matches[:expected_length]:
                val = float(match)
                val = max(0.0, min(1.0, val))
                values.append(val)
            while len(values) < expected_length:
                values.append(random.random())

        total = sum(values)
        if total > 0:
            values = [val / total for val in values]
        else:
            values = [1.0 / expected_length] * expected_length
        return values
    except Exception:
        values = [random.random() for _ in range(expected_length)]
        total = sum(values)
        return [val / total for val in values] if total > 0 else [1.0 / expected_length] * expected_length

def load_factor_data(factor_dir):
    factor_files = glob.glob(os.path.join(factor_dir, "*.csv"))
    factor_files = [f for f in factor_files if f[-12:-4] >= '20150101']
    factor_files.sort()
    
    factor_dfs = []
    for factor_file in factor_files:
        try:
            date_str = os.path.basename(factor_file).replace('.csv', '')
            df = pd.read_csv(factor_file)
            if len(df.columns) >= 2 and df.columns[0] == 'stock' and df.columns[1] == 'stock':
                df = df.iloc[:, 1:]
            df.columns = ['stock'] + [col for col in df.columns[1:]]
            df['date'] = date_str
            factor_dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {factor_file}: {e}")
    return pd.concat(factor_dfs, ignore_index=True) if factor_dfs else None

def process_agent_on_specific_date(node, node_list, pi_record, date, date_group_df):
    """
    单个线程处理：一个 Agent 在某一天的所有 Batch
    """
    stock_order = date_group_df['stock'].tolist()
    num_stocks = len(stock_order)
    day_pi_vectors = []

    for batch_start in range(0, num_stocks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_stocks)
        batch_stock_order = stock_order[batch_start:batch_end]
        batch_day_df = date_group_df.iloc[batch_start:batch_end]

        try:
            prompt = generate_agent_prompt(node, node_list, pi_record, batch_day_df, batch_stock_order, date)
            pi_str = agent_api_inference(model_id="Qwen3-Coder-480B-A35B-Instruct", content=prompt)
            batch_vector = safe_parse_pi_vector(pi_str, len(batch_stock_order))
            day_pi_vectors.extend(batch_vector)
        except Exception as e:
            logger.error(f"Error: Agent {node['id']} Date {date} Batch {batch_start}: {e}")
            day_pi_vectors.extend([0.0] * (batch_end - batch_start))

    # 归一化这一天的总向量
    total = sum(day_pi_vectors)
    if total > 0:
        day_pi_vectors = [v / total for v in day_pi_vectors]
    else:
        day_pi_vectors = [1.0 / num_stocks] * num_stocks
    
    return date, day_pi_vectors

def run_simulation(node_list, factor_df):
    node_list_template = deepcopy(node_list)
    agent_ids = [node["id"] for node in node_list_template]
    dates = factor_df['date'].unique()
    
    # 结果 DataFrame 初始化
    result_df = factor_df[['stock', 'date']].copy()
    for node_id in agent_ids:
        result_df[f"agent_{node_id}"] = None

    pi_record = {node["id"]: {} for node in node_list_template}

    # 1. 外部循环：按 Agent 串行（保证 Agent 间的依赖）
    for node in node_list_template:
        node_id = node["id"]
        logger.info(f">>> Starting Agent {node_id} ({node['type']})")

        # 2. 内部并行：同时处理该 Agent 的所有日期
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_date = {}
            for date in dates:
                date_group_df = factor_df[factor_df['date'] == date].copy().reset_index(drop=True)
                if date_group_df.empty: continue
                
                future = executor.submit(process_agent_on_specific_date, node, node_list_template, pi_record, date, date_group_df)
                future_to_date[future] = date

            for future in as_completed(future_to_date):
                target_date, final_pi_vector = future.result()
                
                # 存入 record 供后续 Agent 参考
                pi_record[node_id][target_date] = final_pi_vector
                
                # 写入结果集
                # 找到对应日期的行索引进行填充
                date_mask = (result_df['date'] == target_date)
                result_df.loc[date_mask, f"agent_{node_id}"] = final_pi_vector
                
                logger.info(f"Agent {node_id} | Date {target_date} done.")

        # 每个 Agent 完成后保存，防止崩盘
        result_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Agent {node_id} data saved to {OUTPUT_CSV}")

    logger.info("Simulation finished.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    f_df = load_factor_data(FACTOR_DIR)
    with open(NODE_LIST_FILE) as f:
        nodes = json.load(f)
    
    logger.info(f"Loaded {len(nodes)} agents. Parallelizing by Date...")
    run_simulation(nodes, f_df)
    
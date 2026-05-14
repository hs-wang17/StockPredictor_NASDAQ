import json
import pandas as pd
from copy import deepcopy
import logging
import os
import re
import random
import glob
from srcs.agent_api_inference import agent_api_inference

DATA_DIR = "/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/"
FACTOR_DIR = os.path.join(DATA_DIR, "daily_factor_data/")
OUTPUT_DIR = os.path.join(DATA_DIR, "invest_agent_data/")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "agent_factors.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "dag_agent_simulation.log")
NODE_LIST_FILE = os.path.join(OUTPUT_DIR, "graph_data/investor_node_list.json")

BATCH_SIZE = 500

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
    """Generate LLM prompt using neighbor decisions for all stocks on the same day"""

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
        if neighbor_id in pi_record and date in pi_record[neighbor_id]:
            neighbor_pi_vector = pi_record[neighbor_id][date]
        if "weight" in neighbor:
            eta = neighbor["weight"]
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
    """
    Extract a vector of float numbers from LLM output string.
    If extraction fails or numbers are out of [0,1], return random values.
    Final vector is normalized to sum to 1.
    """
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
        if total > 0:
            values = [val / total for val in values]
        else:
            values = [1.0 / expected_length] * expected_length
        return values


def load_valid_stocks(meta_file_path):
    """
    Load valid stock symbols from metadata file, excluding ETFs.
    Returns a set of valid stock symbols.
    """
    try:
        meta_df = pd.read_csv(meta_file_path)
        logger.info(f"Loaded {len(meta_df)} records from metadata file")
        
        valid_stocks = meta_df[
            (meta_df['Listing Exchange'] == 'Q') &
            (meta_df['ETF'] == 'N') & 
            (meta_df['Test Issue'] == 'N') &
            (meta_df['Financial Status'] == 'N') &
            (meta_df['Nasdaq Traded'] == 'Y')
        ]['Symbol'].tolist()
        
        logger.info(f"Found {len(valid_stocks)} valid non-ETF stocks")
        return set(valid_stocks)
    except Exception as e:
        logger.error(f"Error loading metadata file {meta_file_path}: {e}")
        return None


def load_factor_data(factor_dir):
    """
    Load factor data from CSV files organized by date.
    Each file name represents a date (e.g., 20110103.csv).
    Returns a combined DataFrame with 'stock', 'date', and factor columns,
    filtered to include only valid non-ETF stocks.
    """
    META_FILE = "/root/autodl-tmp/.autodl/StockPredictor_NASDAQ/data/stock_market_data/symbols_valid_meta.csv"
    
    valid_stocks = load_valid_stocks(META_FILE)
    if valid_stocks is None:
        logger.error("Cannot proceed without valid stock list")
        return None
    
    factor_files = glob.glob(os.path.join(factor_dir, "*.csv"))
    factor_files = [f for f in factor_files if f[-12:-4] >= '20150101']
    
    logger.info(f"Found {len(factor_files)} factor files")
    
    def extract_date_from_filename(filepath):
        basename = os.path.basename(filepath)
        date_str = basename.replace('.csv', '')
        return date_str
    
    factor_files.sort(key=extract_date_from_filename)
    logger.info(f"Files sorted by date: {extract_date_from_filename(factor_files[0])} to {extract_date_from_filename(factor_files[-1])}")
    
    factor_dfs = []
    for factor_file in factor_files:
        date_str = extract_date_from_filename(factor_file)
        
        try:
            df = pd.read_csv(factor_file)
            df = df.iloc[:, :7]  # 只取前7列
            
            if len(df.columns) >= 2 and df.columns[0] == 'stock' and df.columns[1] == 'stock':
                df = df.iloc[:, 1:]
            
            df.columns = ['stock'] + [col for col in df.columns[1:]]
            df['date'] = date_str
            
            # 筛选有效的非ETF股票
            df = df[df['stock'].isin(valid_stocks)]
            logger.debug(f"Loaded {len(df)} valid records for date {date_str}")
            
            if len(df) > 0:
                factor_dfs.append(df)
            
        except Exception as e:
            logger.error(f"Error loading {factor_file}: {e}")
    
    if not factor_dfs:
        logger.error("No valid factor data loaded")
        return None
    
    factor_df = pd.concat(factor_dfs, ignore_index=True)
    logger.info(f"Loaded {len(factor_df)} total valid records from {len(factor_dfs)} files")
    
    return factor_df


def run_simulation_all_agents(node_list, factor_df):
    """
    Loop over each agent first, then over each day (processing stocks in batches to reduce memory usage).
    After each day finishes, save a separate CSV file for that day.
    """
    node_list_template = deepcopy(node_list)
    agent_ids = [node["id"] for node in node_list_template]

    # Create daily output directory
    DAILY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "daily_agent_factors")
    os.makedirs(DAILY_OUTPUT_DIR, exist_ok=True)

    dates = factor_df['date'].unique()
    logger.info(f"Processing {len(dates)} unique dates")

    pi_record = {node["id"]: {} for node in node_list_template}

    for date in dates:
        logger.info(f"Processing date: {date}")
        
        day_df = factor_df[factor_df['date'] == date].copy().reset_index(drop=True)
        stock_order = day_df['stock'].tolist()
        num_stocks = len(stock_order)
        
        if num_stocks == 0:
            logger.warning(f"No stocks found for date {date}, skipping")
            continue

        # Initialize daily result dataframe
        daily_result_df = day_df[['stock', 'date']].copy()
        for node_id in agent_ids:
            daily_result_df[f"agent_{node_id}"] = None

        for node in node_list_template:
            node_id = node["id"]
            node["pi"] = 0.0

            batch_pi_vectors = []

            for batch_start in range(0, num_stocks, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_stocks)
                batch_stock_order = stock_order[batch_start:batch_end]
                batch_day_df = day_df.iloc[batch_start:batch_end]

                logger.info(
                    f"Agent {node_id} | Date {date} | Processing batch {batch_start//BATCH_SIZE + 1}/{(num_stocks-1)//BATCH_SIZE + 1} ({len(batch_stock_order)} stocks)"
                )
                try:
                    prompt = generate_agent_prompt(node, node_list_template, pi_record, batch_day_df, batch_stock_order, date)
                    pi_str = agent_api_inference(model_id="Qwen3.5-Plus", content=prompt)
                    batch_pi_vector = safe_parse_pi_vector(pi_str, len(batch_stock_order))

                    batch_pi_vectors.append(batch_pi_vector)

                except Exception as e:
                    logger.error(f"Error for agent {node_id} | Date {date} | Batch {batch_start//BATCH_SIZE + 1}: {e}")
                    batch_pi_vector = [0.0] * len(batch_stock_order)
                    batch_pi_vectors.append(batch_pi_vector)

            try:
                combined_pi_vector = []
                for batch_vector in batch_pi_vectors:
                    combined_pi_vector.extend(batch_vector)

                total = sum(combined_pi_vector)
                if total > 0:
                    combined_pi_vector = [val / total for val in combined_pi_vector]
                else:
                    combined_pi_vector = [1.0 / num_stocks] * num_stocks

                pi_record[node_id][date] = combined_pi_vector

                for i in range(len(day_df)):
                    daily_result_df.loc[i, f"agent_{node_id}"] = combined_pi_vector[i]

                logger.info(f"Agent {node_id} | Date {date} | Final normalized vector: {combined_pi_vector[:3]}... (total: {num_stocks} stocks)")

            except Exception as e:
                logger.error(f"Error combining batches for agent {node_id} | Date {date}: {e}")
                pi_record[node_id][date] = [0.0] * num_stocks
                daily_result_df[f"agent_{node_id}"] = None

        # Save daily CSV file
        daily_output_file = os.path.join(DAILY_OUTPUT_DIR, f"{date}.csv")
        daily_result_df.to_csv(daily_output_file, index=False)
        logger.info(f"Date {date} | Saved to {daily_output_file}")

    logger.info("All dates completed.")
    return None


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    factor_df = load_factor_data(FACTOR_DIR)
    
    with open(NODE_LIST_FILE) as f:
        node_list = json.load(f)
    
    logger.info(f"Loaded {len(node_list)} investor nodes")
    
    run_simulation_all_agents(node_list, factor_df)

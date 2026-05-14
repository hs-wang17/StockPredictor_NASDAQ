screen -dmS train_lgbm_session bash -c 'python /root/autodl-tmp/.autodl/StockPredictor_NASDAQ/code/basic_framework/step6_train_model_lgbm.py > /root/autodl-tmp/.autodl/StockPredictor_NASDAQ/logs/train_lgbm.log 2>&1'

screen -dmS train_mlp_MSEloss_session bash -c 'python /root/autodl-tmp/.autodl/StockPredictor_NASDAQ/code/basic_framework/step6_train_model_mlp_MSEloss.py > /root/autodl-tmp/.autodl/StockPredictor_NASDAQ/logs/train_mlp_MSEloss.log 2>&1'

screen -dmS train_mlp_L1loss_session bash -c 'python /root/autodl-tmp/.autodl/StockPredictor_NASDAQ/code/basic_framework/step6_train_model_mlp_L1loss.py > /root/autodl-tmp/.autodl/StockPredictor_NASDAQ/logs/train_mlp_L1loss.log 2>&1'

screen -dmS generate_agent_factor_qwen_session bash -c 'python /root/autodl-tmp/.autodl/StockPredictor_NASDAQ/code/invest_agent/step2_generate_agent_factor_qwen.py > /root/autodl-tmp/.autodl/StockPredictor_NASDAQ/logs/generate_agent_factor_qwen.log 2>&1'

screen -dmS generate_agent_factor_parallel_qwen_session bash -c 'python /root/autodl-tmp/.autodl/StockPredictor_NASDAQ/code/invest_agent/step2_generate_agent_factor_parallel_qwen.py > /root/autodl-tmp/.autodl/StockPredictor_NASDAQ/logs/generate_agent_factor_parallel_qwen.log 2>&1'

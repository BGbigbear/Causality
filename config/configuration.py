from config.prompt_list import *

db_path = "C:/Users/sibin/worksp/COMPETITION/Causality Extraction/database/faiss_bce"

test_file = './data/initial/train1.json'
causality_file = './data/reconstruction/train1_1.json'

analysis_file = "./result/causality_train1_analysis_rougeSFT_full_1.json"
pred_file = './result/causality_train1_predict_rougeSFT_full_1.json'
# analysis_file = "./result/causality_analysis_test.json"
# pred_file = './result/causality_predict_test.json'

event_roles = ['cause_event', 'effect_event']
global_prompts = causality_prompts_3shots

model_path = "../Models/google/gemma-2-27b-it"

from config.prompt_list import *

db_path = "C:/Users/sibin/worksp/COMPETITION/Causality Extraction/database/faiss_bce"

test_file = './data/initial/train2.json'
causality_file = './data/reconstruction/train2_cot.json'

analysis_file = "./data/fewshot/causality_train2cot_analysis_rouge_full_0.json"
pred_file = './data/fewshot/causality_train2cot_predict_rouge_full_0.json'
# analysis_file = "./result/causality_analysis_test.json"
# pred_file = './result/causality_predict_test.json'

event_roles = ['cause_event', 'effect_event']
global_prompts = causality_prompts_3shots

model_path = "../Models/google/gemma-2-27b-it"

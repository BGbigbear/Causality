from config.prompt_list import *

db_path = "C:/Users/sibin/worksp/COMPETITION/Causality Extraction/database/faiss_bce"

test_file = 'C:/Users/sibin/worksp/COMPETITION/Causality Extraction/test1.json'
causality_file = 'C:/Users/sibin/worksp/COMPETITION/Causality Extraction/merge1_1.json'

# analysis_file = "./result/causality_valid_analysis_rouge.json"
# pred_file = './result/causality_valid_predict_rouge.json'

analysis_file = "./result/causality_test1_analysis_rouge_555.json"
pred_file = './result/causality_test1_predict_rouge_555.json'

event_roles = ['cause_event', 'effect_event']
global_prompts = causality_prompts_3shots

model_path = "../Models/google/gemma-2-27b-it"

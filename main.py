from inference.fast_inference import generate
from util.data_expand import multi_test_data


def run():
    # preprocess_mode: 0 - No preprocess; 1 - Use rouge; 2- Use rag
    # inference_mode: 0 - Use API; 1 - Use local model; 2 - Use vllm
    generate(start_point=0, end_point=0, preprocess_mode=1, max_workers=10, inference_mode=2, recheck=False)


if __name__ == '__main__':
    run()

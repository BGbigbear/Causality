from inference.fast_inference import generate


def run():
    generate(start_point=0, end_point=1, rouge=True, using_api=True, recheck=False)


if __name__ == '__main__':
    run()

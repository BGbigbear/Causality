from inference.fast_inference import generate


def run():
    generate(start_point=0, end_point=0, rouge=True, using_api=True)


if __name__ == '__main__':
    run()

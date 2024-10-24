from inference.api_inference import generate


def run():
    generate(start_point=0, end_point=100, rouge=True)


if __name__ == '__main__':
    run()

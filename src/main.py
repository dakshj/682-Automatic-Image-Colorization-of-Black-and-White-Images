from architecture.model import init_model
from data.image_data import load_raw_image_data


def train():
    X_train = load_raw_image_data('../dataset/train', normalize=True)


if __name__ == '__main__':
    train()

    model = init_model()

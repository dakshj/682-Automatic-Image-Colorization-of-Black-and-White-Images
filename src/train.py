from load_images import load_images_as_numpy


def train():
    X = load_images_as_numpy(
        r"D:\Projects\682 Automatic Image Colorization of Black-and-White Images\dataset\train")
    print(X)


if __name__ == '__main__':
    train()

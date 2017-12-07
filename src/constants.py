class Constants:
    class Global:
        H = 256  # Image height
        W = 256  # Image width

    class Encoder:
        KERNEL_SIZE = (3, 3)
        ACTIVATION_RELU = 'relu'
        PADDING_SAME = 'same'
        STRIDES = 2

    class Decoder:
        ACTIVATION_TANH = 'tanh'

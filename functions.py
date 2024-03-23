# Function to create directory
def create_dir(path):
    #path = os.path.join("/content/drive/MyDrive/Colab Notebooks", path)
    if not os.path.exists(path):
      print(path)
      os.makedirs(path)


# Function to load test dataset names and split into various sets
def load_dataset(path, split=0.1):
    images = sorted(glob(os.path.join(path, "Transversal", "*.png")))
    masks = sorted(glob(os.path.join(path, "Transversal-Mask", "*.png")))

    split_size = int(len(masks) * split)

    x_train, x_validation = train_test_split(images, test_size=split_size, random_state=42)
    y_train, y_valiadtion = train_test_split(masks, test_size=split_size, random_state=42)

    x_train, x_test = train_test_split(x_train, test_size=split_size, random_state=42)
    y_train, y_test = train_test_split(y_train, test_size=split_size, random_state=42)

    return (x_train, y_train), (x_validation, y_valiadtion), (x_test, y_test)


# Function to read the training input images and normalise them
def read_x(set):
    arr=[]
    for i in set:
        x = cv2.imread(i, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)
        arr.append(x)
    data=np.array(arr)
    return data


#  Function to read the training target images and normalise them
def read_y(set):
    arr=[]
    for i in set:
        x = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)    ## (h, w)
        x = np.expand_dims(x, axis=-1)## (h, w, 1)
        arr.append(x)
    data=np.array(arr)
    return data


# Function defining one convolution block consisting of two conv layers
def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


# Function to create a single encoder block consisting of one endcoder and maxpool layer
def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


# Function creating a decoder block consisting on one upsampling layer and 1 conv block
def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


# Function to create one U-Net
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # print(s1.shape, s2.shape, s3.shape, s4.shape)
    # print(p1.shape, p2.shape, p3.shape, p4.shape)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="UNET")
    return model


smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# Function to read test input images
def read_x_test(set):
    arr=[]
    for i in set:
        x = cv2.imread(i, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (W, H))
        arr.append(x)
    data=np.array(arr)
    return data


# Function to read the test target images
def read_y_test(set):
    arr=[]
    for i in set:
        x = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (W, H))
        arr.append(x)
    data=np.array(arr)
    return data

# Function to save the output images
def save_results(image, mask, y_pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    line = np.ones((H, 10, 3)) * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)
    
    

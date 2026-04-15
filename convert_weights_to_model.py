import tensorflow as tf
from tensorflow.keras import layers, Model

# -----------------------------
# EXACT COPY FROM YOUR NOTEBOOK
# -----------------------------
def conv_block(x, filters):
    y = layers.Convolution2D(filters, (3, 3), padding='same', activation='relu')(x)
    y = layers.Convolution2D(filters, (3, 3), padding='same', activation='relu')(y)
    return y

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(f)
    return f, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)

    def crop_and_concat_fn(inputs):
        target_tensor, skip_tensor = inputs

        target_h = tf.shape(target_tensor)[1]
        target_w = tf.shape(target_tensor)[2]
        skip_h = tf.shape(skip_tensor)[1]
        skip_w = tf.shape(skip_tensor)[2]

        h_diff = skip_h - target_h
        w_diff = skip_w - target_w

        cropped_skip = tf.image.crop_to_bounding_box(
            skip_tensor,
            offset_height=(h_diff // 2),
            offset_width=(w_diff // 2),
            target_height=target_h,
            target_width=target_w
        )
        return tf.concat([target_tensor, cropped_skip], axis=-1)

    x = layers.Lambda(crop_and_concat_fn)([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(300, 300, 3)):
    inputs = layers.Input(input_shape)

    f1, p1 = encoder_block(inputs, 64)
    f2, p2 = encoder_block(p1, 128)
    f3, p3 = encoder_block(p2, 256)
    f4, p4 = encoder_block(p3, 512)

    bottleneck = conv_block(p4, 1024)

    d1 = decoder_block(bottleneck, f4, 512)
    d2 = decoder_block(d1, f3, 256)
    d3 = decoder_block(d2, f2, 128)
    d4 = decoder_block(d3, f1, 64)

    d4_resized = layers.Resizing(300, 300)(d4)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4_resized)

    model = Model(inputs, outputs, name="U-Net")
    return model


# -----------------------------
# BUILD MODEL (IMPORTANT)
# -----------------------------
model = build_unet()

# Force model to build (CRITICAL)
model(tf.zeros((1, 300, 300, 3)))

print("Model built successfully")

# -----------------------------
# LOAD WEIGHTS
# -----------------------------
model.load_weights(
    "/home/tondamanati-abhinay/Desktop/capstone/Capstone/DataSet/DS/default_unet_model.h5"
)

print("Weights loaded successfully")

# -----------------------------
# SAVE FULL MODEL
# -----------------------------
model.save(
    "/home/tondamanati-abhinay/Desktop/capstone/Capstone/DataSet/DS/final_unet.keras"
)

print("Saved as final_unet.keras")
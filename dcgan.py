import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, ReLU, BatchNormalization, Dropout, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt

# ============================ Configuration ============================ #
train_folder = r"C:\Users\cclab\PycharmProjects\PythonProject\DOP_CNN\original_dataset\train"
latent_dim = 100
img_shape = (1024, 1024, 3)  # Updated resolution
batch_size = 2
epochs = 10000
save_interval = 100
target_count = 1000

# ============================ Convert BMP to JPG ============================ #
def convert_bmp_to_jpg(input_folder, output_folder, quality=85):
    os.makedirs(output_folder, exist_ok=True)
    supported_extensions = ('.bmp', '.dib')
    converted_count = 0
    failed_count = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            bmp_path = os.path.join(input_folder, filename)
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpg_path = os.path.join(output_folder, jpg_filename)
            try:
                with Image.open(bmp_path) as img:
                    rgb_img = img.convert('RGB')
                    rgb_img.save(jpg_path, 'JPEG', quality=quality)
                print(f"Converted: {filename} -> {jpg_filename}")
                converted_count += 1
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")
                failed_count += 1

    print("\nConversion Completed.")
    print(f"Total .bmp files found: {converted_count + failed_count}")
    print(f"Successfully converted: {converted_count}")
    print(f"Failed conversions: {failed_count}")

# ============================ Models ============================ #
def build_generator():
    model = Sequential(name="Generator")
    model.add(Input(shape=(latent_dim,)))
    model.add(Dense(16 * 16 * 512))  # Start with a larger feature map
    model.add(Reshape((16, 16, 512)))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Upsampling layers
    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))  # 16x16 -> 32x32
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))  # 32x32 -> 64x64
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))   # 64x64 -> 128x128
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(32, kernel_size=5, strides=2, padding='same'))   # 128x128 -> 256x256
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(16, kernel_size=5, strides=2, padding='same'))   # 256x256 -> 512x512
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh'))  # 512x512 -> 1024x1024
    return model

def build_discriminator():
    model = Sequential(name="Discriminator")
    model.add(Input(shape=img_shape))
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same'))  # 1024x1024 -> 512x512
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))  # 512x512 -> 256x256
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))  # 256x256 -> 128x128
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))  # 128x128 -> 64x64
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(1024, kernel_size=5, strides=2, padding='same'))  # 64x64 -> 32x32
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# ============================ Utility Functions ============================ #
def load_images(input_folder):
    images = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(img_path).convert('RGB').resize((img_shape[1], img_shape[0]))
                img_array = np.array(img) / 127.5 - 1.0
                images.append(img_array)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
    if not images:
        raise ValueError("No images found in the input folder.")
    return np.array(images)

def save_generated_images(generator, epoch, output_folder, num_images=10):
    """
    Save generated images individually during each save interval.
    """
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = generator.predict(noise)
    # Normalize pixel values to [0, 255] for saving
    generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)
    os.makedirs(output_folder, exist_ok=True)

    # Save each image separately
    for i, img in enumerate(generated_images):
        img_filename = f"epoch_{epoch}_image_{i+1}.png"
        img_path = os.path.join(output_folder, img_filename)
        Image.fromarray(img).save(img_path)

# ============================ Training Step ============================ #
@tf.function
def train_step(real_images):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        real_labels = tf.random.uniform(minval=0.9, maxval=1.0, shape=tf.shape(real_output))
        fake_labels = tf.random.uniform(minval=0.0, maxval=0.1, shape=tf.shape(fake_output))
        real_loss = cross_entropy(real_labels, real_output)
        fake_loss = cross_entropy(fake_labels, fake_output)
        disc_loss = (real_loss + fake_loss) / 2

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    with tf.GradientTape() as gen_tape:
        fake_images = generator(noise, training=True)
        fake_output = discriminator(fake_images, training=True)
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return disc_loss, gen_loss

# ============================ Main Training Loop ============================ #
def main():
    global generator, discriminator
    generator = build_generator()
    discriminator = build_discriminator()

    global discriminator_optimizer, generator_optimizer, cross_entropy
    discriminator_optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.999)
    generator_optimizer = Adam(0.0002, beta_1=0.5, beta_2=0.999, clipvalue=1.0)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    for subfolder in os.listdir(train_folder):
        if not subfolder.endswith("augmented"):
            continue

        input_folder = os.path.join(train_folder, subfolder)
        if not os.path.isdir(input_folder):
            continue

        print(f"Processing folder: {input_folder}")
        convert_bmp_to_jpg(input_folder, input_folder)

        X_train = load_images(input_folder)
        print(f"Loaded {X_train.shape[0]} images for training from {input_folder}.")

        current_count = 0
        for epoch in range(1, epochs + 1):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_images = X_train[idx]

            d_loss, g_loss = train_step(real_images)

            if epoch % save_interval == 0 or epoch == epochs:
                print(f"Epoch {epoch} [D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}]")
                save_generated_images(generator, epoch, input_folder, num_images=10)
                current_count += 10
                if current_count >= target_count:
                    print(f"Reached target of {target_count} images for {subfolder}.")
                    break

if __name__ == "__main__":
    main()

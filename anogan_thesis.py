import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Model
import time
import pandas as pd
import os

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_train = np.expand_dims(X_train, axis=-1)
X_test = X_test.astype('float32') / 255.0
X_test = np.expand_dims(X_test, axis=-1)

# Define common hyperparameters
latent_dim = 100
epochs = 30
batch_size = 128
input_shape = (28, 28, 1)

# Define normal and abnormal classes
normal_class = 3  # Define the class you want to treat as normal

# Filter data for normal class only
x_train_normal = X_train[y_train != normal_class]
x_test_normal = X_test[y_test != normal_class]
x_test_abnormal = X_test[y_test == normal_class]

# Create labels for the test set (0 for normal, 1 for abnormal)
y_test_abnormal = np.where(y_test[y_test == normal_class] == normal_class, 0, 1)

# Define ANOGAN generator
def build_anogan_generator():
    generator_input = Input(shape=(latent_dim,))
    x = Dense(7 * 7 * 128, activation='relu')(generator_input)
    x = Reshape((7, 7, 128))(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    generator_output = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')(x)
    generator = Model(generator_input, generator_output)
    return generator

# Define ANOGAN discriminator
def build_anogan_discriminator():
    discriminator_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(discriminator_input)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    discriminator_output = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, discriminator_output)
    return discriminator

# Build ANOGAN generator and discriminator
anogan_generator = build_anogan_generator()
anogan_discriminator = build_anogan_discriminator()

# Compile ANOGAN discriminator
anogan_discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Build ANOGAN GAN
def build_anogan_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = Model(gan_input, gan_output)
    return gan

anogan_gan = build_anogan_gan(anogan_generator, anogan_discriminator)
anogan_gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Lists to store loss values and AUC values
d_loss_values = []
g_loss_values = []
auc_values = []

# Training loop for ANOGAN
start_time = time.time()
for epoch in range(epochs):
    for step in range(x_train_normal.shape[0] // batch_size):
        batch_start = step * batch_size
        batch_end = batch_start + batch_size
        real_images = x_train_normal[batch_start:batch_end]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = anogan_generator.predict(noise)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train the discriminator
        d_loss_real = anogan_discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = anogan_discriminator.train_on_batch(generated_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))

        # Train the generator
        g_loss = anogan_gan.train_on_batch(noise, valid_labels)

    # Calculate AUC for anomaly detection between normal test data and generated abnormal data
    real_scores = anogan_discriminator.predict(x_test_normal)
    fake_scores = anogan_discriminator.predict(x_test_abnormal)
    auc = roc_auc_score(np.concatenate((np.ones_like(real_scores), np.zeros_like(fake_scores))),
                       np.concatenate((real_scores, fake_scores)))

    # Print and save progress
    print(f"ANOGAN - Epoch {epoch + 1}/{epochs} - D Loss: {d_loss} - G Loss: {g_loss} - AUC: {auc}")

    # Append loss and AUC values to lists
    d_loss_values.append(d_loss)
    g_loss_values.append(g_loss)
    auc_values.append(auc)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")

# Create a DataFrame to store loss and AUC values
data = {'Epoch': range(1, epochs + 1),
        'D Loss': d_loss_values,
        'G Loss': g_loss_values,
        'AUC': auc_values}
df = pd.DataFrame(data)

# Specify the directory path where you want to save the file
output_directory = '/C:/Users/Admin/Desktop/Master Thesis'

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Now you can save the file to this directory
output_file = os.path.join(output_directory, 'anogan_results.xlsx')
df.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")

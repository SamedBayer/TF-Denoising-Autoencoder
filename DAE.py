import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow_datasets as tfds


class DenoiseAutoEncoder(tf.keras.Model):
  """
  A class to represent a denoising autoencoder which inherits from Keras's Model.

  ...

  Attributes
  ----------
  encoder: tf.keras.Sequential
    Convolutional Encoder network.
  decoder: tf.keras.Sequential
    Convolutional Decoder network.

  Methods
  -------
  call(self, x):
    Performs forward propagation through the autoencoder on the input x.
  """
  def __init__(self):
    super(DenoiseAutoEncoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(32, 32, 3)), 
      tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      tf.keras.layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')]) 

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


def normalize(images, labels):
    """
    Normalizes the input images.

    Parameters
    ----------
    images: Tensor
      The input images to be normalized.
    labels: Tensor
      The labels corresponding to the images. Not necessary for this task

    Returns
    -------
    images: Tensor
      The normalized images.
    """
    images = tf.cast(images, tf.float32) / 255.0
    return images

    
def add_noise(images, mean=0.0, std=0.5, noise_factor=1):
    """
    Adds Gaussian noise to the input images.

    Parameters
    ----------
    images: Tensor
      The input images to which noise will be added.
    mean: float, optional
      The mean of the Gaussian noise.
    std: float, optional
      The standard deviation of the Gaussian noise.
    noise_factor: float, optional
      A factor by which the noise will be multiplied.

    Returns
    -------
    images_noisy: Tensor
      The images with added noise.
    """
    noise = tf.random.normal(shape=tf.shape(images), mean=mean, stddev=std)
    images_noisy = images + noise*noise_factor
    images_noisy = tf.clip_by_value(images_noisy, 0., 1.)
    return images_noisy


def pre_process(original_dataset, batch_size=10, mean=0.0, std=0.5, noise_factor=1):
    """
    Preprocesses the original dataset by normalizing, adding noise, batching, 
    and zipping the noisy images with the original images.

    Parameters
    ----------
    original_dataset: tf.data.Dataset
      The original dataset to be preprocessed.
    batch_size: int, optional
      The batch size for batching the dataset.
    mean: float, optional
      The mean of the Gaussian noise to be added.
    std: float, optional
      The standard deviation of the Gaussian noise to be added.
    noise_factor: float, optional
      A factor by which the noise will be multiplied.

    Returns
    -------
    dataset_combined: tf.data.Dataset
      The preprocessed dataset with noisy and clean images combined.
    """
    # Apply normalization to the dataset
    dataset = original_dataset.map(normalize)

    # Create noisy datasets
    dataset_noisy = dataset.map(lambda x: add_noise(x, mean=mean, std=std, noise_factor=noise_factor))

    # Batch the datasets
    dataset = dataset.batch(batch_size)
    dataset_noisy = dataset_noisy.batch(batch_size)

    # Zip and combine noisy dataset with original dataset for Denoising Autoencoder
    dataset_combined = tf.data.Dataset.zip((dataset_noisy, dataset))

    return dataset_combined

def plot_loss_curve(history):
    """
    Plots the training and validation loss curves.

    Parameters
    ----------
    history: tf.keras.callbacks.History
      History object from training a model.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')  # Optional: Add x-axis label
    plt.ylabel('Loss')  # Optional: Add y-axis label
    plt.legend()
    plt.show()


def visualize_denoising(autoencoder, test_dataset_combined):
    """
    Visualizes the denoising capability of the autoencoder on a test dataset.
    Display original + noise, denoised, and original images.

    Parameters
    ----------
    autoencoder: DenoiseAutoEncoder
      The denoising autoencoder model.
    test_dataset_combined: tf.data.Dataset
      The combined dataset of original and noisy images.

    Returns
    -------
    None
    """
    # Take one batch from the test dataset
    noisy_images, original_images = next(iter(test_dataset_combined))
    
    # Run the noisy images through the autoencoder
    denoised_images = autoencoder.predict(noisy_images)

    # Define the number of example pairs to show, max 
    n = np.minimum(10, denoised_images.shape[0])  

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original + noise image
        ax = plt.subplot(3, n, i + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(noisy_images[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display denoised image
        bx = plt.subplot(3, n, i + n + 1)
        plt.title("denoised")
        plt.imshow(tf.squeeze(denoised_images[i]))
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)

        # Display original image
        cx = plt.subplot(3, n, i + 2*n + 1)
        plt.title("original")
        plt.imshow(tf.squeeze(original_images[i]))
        plt.gray()
        cx.get_xaxis().set_visible(False)
        cx.get_yaxis().set_visible(False)

    plt.show()


def denoise_image(img_path, model, bool_noise=True, mean=0.5, std=0.1, noise_factor=1):
    """
    First, converts image to RGB image and (32, 32) shape. Then apply noise to the image.
    Denoises an image from a specified path using the given model. Plots images.

    Parameters
    ----------
    img_path: str
      The path to the image to be denoised.
    model: tf.keras.Model
      The denoising model to use.
    bool_noise: bool, optional
      A flag to determine whether to add noise before denoising.
    mean: float, optional
      The mean of the Gaussian noise to be added.
    std: float, optional
      The standard deviation of the Gaussian noise to be added.
    noise_factor: float, optional
      A factor by which the noise will be multiplied.

    Returns
    -------
    img_normalized: np.array
      The normalized original image.
    img_denoised: np.array
      The denoised image.
    """
    img = Image.open(img_path).convert('RGB')  # Add convert('RGB') to remove 4th dimension
    img_resized = img.resize((32, 32))
    img_normalized = np.array(img_resized) / 255.0  
    if bool_noise:
        img_noisy = add_noise(img_normalized, mean=mean, std=std, noise_factor=noise_factor)  
    else:
        img_noisy = img_normalized 
    img_noisy = np.expand_dims(img_noisy, axis=0)  # add batch dimension

    img_denoised = model.predict(img_noisy)
    img_denoised = np.squeeze(img_denoised)  # remove batch dimension

    plt.figure(figsize=(20, 4))
    # Display original + noise image
    ax = plt.subplot(1, 3, 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(img_noisy))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display denoised image
    bx = plt.subplot(1, 3, 2)
    plt.title("denoised")
    plt.imshow(tf.squeeze(img_denoised))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)

    # Display original image
    cx = plt.subplot(1, 3, 3)
    plt.title("original")
    plt.imshow(tf.squeeze(img_normalized))
    plt.gray()
    cx.get_xaxis().set_visible(False)
    cx.get_yaxis().set_visible(False)

    plt.show()

    return img_normalized, img_denoised
    

def metric_mse(original_image, denoised_image):
    """
    Computes the Mean Squared Error (MSE) between the original and denoised image.
    Prints MSE.

    Parameters
    ----------
    original_image: np.array
      The original image.
    denoised_image: np.array
      The denoised image.

    Returns
    -------
    mse: float
      The Mean Squared Error between the original and denoised image.
    """
    original_image_flattened = original_image.flatten()
    denoised_image_flattened = denoised_image.flatten()
    
    # Compute MSE between original and denoised image
    mse = mean_squared_error(original_image_flattened, denoised_image_flattened)
    print("MSE: ", mse)
    return mse
    
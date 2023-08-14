import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torchvision import transforms


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    return ImageOps.expand(img, padding)


def video_to_images(
    video_file_path, image_folder_path, save_after_frame=None, save_before_frame=None
):
    os.makedirs(image_folder_path, exist_ok=True)
    vidcap = cv2.VideoCapture(video_file_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if save_before_frame or save_after_frame:
            if count >= save_after_frame and count <= save_before_frame:
                cv2.imwrite(
                    os.path.join(
                        image_folder_path, f"image_{count-save_after_frame}.png"
                    ),
                    image,
                )
        else:
            cv2.imwrite(
                os.path.join(image_folder_path, f"image_{count}.png"),
                image,
            )
        success, image = vidcap.read()
        count += 1
    print(f"Total Frames Read: {count}!")


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    return ImageOps.expand(img, padding)


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]



def create_numpy_array_file(
    image_sequences_folder_name,
    numpy_array_file_name,
    resize_image_size=None,
    padding_required=False,
    grayscale_required=False,
):
    numpy_arrays = []
    transformer = transforms.Grayscale()
    for image_file_name in os.listdir(image_sequences_folder_name):
        image_file_relative_path = os.path.join(
            image_sequences_folder_name, image_file_name
        )
        if (
            image_file_relative_path.endswith(".tif")
            or image_file_relative_path.endswith(".png")
            or image_file_relative_path.endswith(".jpg")
            or image_file_relative_path.endswith(".jpeg")
        ):
            image = Image.open(image_file_relative_path)
            if grayscale_required:
                image = transformer(image)
            if padding_required:
                image = np.asarray(
                    resize_with_padding(
                        image,
                        (resize_image_size, resize_image_size),
                    )
                )
            else:
                image = crop_center(
                    np.asarray(image), resize_image_size, resize_image_size
                )

            numpy_arrays.append(image)
    np.save(numpy_array_file_name, numpy_arrays)


def create_single_npy_file(numpy_files_folder_path, single_npy_file_path):
    all_numpy_arrays = []
    for numpy_file_name in os.listdir(numpy_files_folder_path):
        all_numpy_arrays.append(
            np.load(os.path.join(numpy_files_folder_path, numpy_file_name))
        )
    np.save(single_npy_file_path, np.vstack(np.array(all_numpy_arrays)))


def add_noise(inputs, noise_factor=0.3):
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0.0, 1.0)
    return noisy


def plot_examples(image_batch, rows, columns):
    n = image_batch.shape[0]
    fig = plt.figure(figsize=(8, 2))
    for i in range(n):
        img = image_batch[i]
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
        plt.axis("off")
    plt.show()


def train_sdae(
    model,
    device,
    dataloader,
    noise_factor=0.2,
):
    model.train()
    final_loss = []
    for image_batch in tqdm(dataloader):
        image_noisy = add_noise(image_batch, noise_factor)
        image_batch = image_batch.to(device)
        image_noisy = image_noisy.to(device)
        featurees, total_loss = model(image_noisy)
        final_loss.append(total_loss)
    return sum(final_loss) / len(final_loss)


def validate_sdae(
    model,
    device,
    loss_func,
    dataloader,
    noise_factor=0.2,
):
    model.eval()
    with torch.inference_mode():
        outputs = []
        labels = []
        for image_batch in tqdm(dataloader):
            image_noisy = add_noise(image_batch, noise_factor)
            image_noisy = image_noisy.to(device)
            encoded_data, reconstucted_images = model(image_noisy)
            outputs.append(reconstucted_images.cpu())
            labels.append(image_batch.cpu())
        outputs = torch.cat(outputs)
        labels = torch.cat(labels)
        val_loss = loss_func(outputs, labels)
    return val_loss.data


def test_sdae(
    model,
    device,
    dataloader,
    loss_fn,
    threshold=0.2,
    noise_factor=0.2,
):
    model.eval()
    with torch.inference_mode():
        for idx, image_batch in tqdm(enumerate(dataloader)):
            image_noisy = add_noise(image_batch, noise_factor)
            image_noisy = image_noisy.to(device)
            image_batch = image_batch.to(device)
            encoded_data, reconstucted_images = model(image_noisy)
            test_loss = loss_fn(reconstucted_images, image_batch)
            print(f"Reconstruction Error for Batch No. {idx} = {test_loss}")
            if test_loss > threshold:
                print("=" * 30)
                print("Abnormal Event Detected!\n")
                plot_examples(image_batch[:10], rows=2, columns=5)
                plot_examples(image_batch[10:], rows=2, columns=5)
                print("=" * 30)


def test_sdae_inference(
    model,
    device,
    dataloader,
    loss_fn,
    threshold=0.2,
    noise_factor=0.2,
):
    model.eval()
    reconstruction_errors = []
    with torch.inference_mode():
        for idx, image_batch in tqdm(enumerate(dataloader)):
            image_noisy = add_noise(image_batch, noise_factor)
            image_noisy = image_noisy.to(device)
            image_batch = image_batch.to(device)
            encoded_data, reconstucted_images = model(image_noisy)
            test_loss = loss_fn(reconstucted_images, image_batch)
            print(f"Reconstruction Error for Batch No. {idx} = {test_loss}")
            reconstruction_errors.append(test_loss.cpu().item())
    return reconstruction_errors

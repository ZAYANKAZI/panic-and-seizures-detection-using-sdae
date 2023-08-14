import os
import argparse
from utils import *
from sdae import *
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


LR = 1e-3
DEVICE = get_device()
BATCH_SIZE = 16
NUM_EPOCHS = 100
NOISE_FACTOR = 0.2
WEIGHT_DECAY = 1e-5
RESIZE_IMAGE_SIZE = 256
LOSS_FUNC = nn.MSELoss()
NORMAL_VIDEO_PATH = None
ABNORMAL_VIDEO_PATH = None
TRAIN_IMAGES_PATH = None
TEST_IMAGES_PATH = None
TRAINED_MODELS_PATH = None
DATA_FOLDER_PATH = None
NUMPY_FILES_FOLDER_PATH = None
MODEL_DATA_FILES_FOLDER_PATH = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDAE Arguments")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="total number of epochs to run"
    )
    parser.add_argument(
        "--noise-factor", type=float, default=0.2, help="noise factor for SDAE"
    )
    parser.add_argument("--normal-path", type=str, help="normal video path")
    parser.add_argument("--abnormal-path", type=str, help="abnormal video path")
    parser.add_argument(
        "--folder-name", type=str, help="folder name to store files as per scenerio"
    )
    parser.add_argument(
        "--show-test-logs", type=bool, help="this will perform testing on normal video"
    )
    opt = parser.parse_args()

    print(opt)

    scenario_folder_name = opt.folder_name

    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.epochs
    NOISE_FACTOR = opt.noise_factor
    NORMAL_VIDEO_PATH = opt.normal_path
    ABNORMAL_VIDEO_PATH = opt.abnormal_path
    SHOW_TEST_LOGS = opt.show_test_logs
    TRAIN_IMAGES_PATH = os.path.join(
        os.getcwd(), "image_data_folder", scenario_folder_name, "Train"
    )
    TEST_IMAGES_PATH = os.path.join(
        os.getcwd(), "image_data_folder", scenario_folder_name, "Test"
    )
    DATA_FOLDER_PATH = os.path.join(
        os.getcwd(), "image_data_folder", scenario_folder_name
    )
    TRAINED_MODELS_PATH = os.path.join(
        os.getcwd(), "trained_models", scenario_folder_name
    )
    NUMPY_FILES_FOLDER_PATH = os.path.join(
        os.getcwd(), "numpy_files", scenario_folder_name
    )
    MODEL_DATA_FILES_FOLDER_PATH = os.path.join(
        os.getcwd(), "processed_data_files", scenario_folder_name
    )

    if not os.path.exists(TRAIN_IMAGES_PATH):
        os.makedirs(TRAIN_IMAGES_PATH, exist_ok=True)
    if not os.path.exists(TEST_IMAGES_PATH):
        os.makedirs(TEST_IMAGES_PATH, exist_ok=True)
    if not os.path.exists(TRAINED_MODELS_PATH):
        os.makedirs(TRAINED_MODELS_PATH, exist_ok=True)
    if not os.path.exists(DATA_FOLDER_PATH):
        os.makedirs(DATA_FOLDER_PATH, exist_ok=True)
    if not os.path.exists(NUMPY_FILES_FOLDER_PATH):
        os.makedirs(NUMPY_FILES_FOLDER_PATH, exist_ok=True)
    if not os.path.exists(MODEL_DATA_FILES_FOLDER_PATH):
        os.makedirs(MODEL_DATA_FILES_FOLDER_PATH, exist_ok=True)
    os.makedirs(f"./summaries/{scenario_folder_name}/", exist_ok=True)

    print(f"Torch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")

    if len(os.listdir(TRAIN_IMAGES_PATH)) == 0:
        video_to_images(
            NORMAL_VIDEO_PATH,
            TRAIN_IMAGES_PATH,
            # save_after_frame=0,
            # save_before_frame=9000,
        )
        video_to_images(
            ABNORMAL_VIDEO_PATH,
            TEST_IMAGES_PATH,
            save_after_frame=0,
            save_before_frame=3000,
        )

    for folder_name in os.listdir(DATA_FOLDER_PATH):
        if folder_name in ["Train", "Test"]:
            try:
                video_image_sequences_folder_name = os.path.join(
                    DATA_FOLDER_PATH, folder_name
                )
                numpy_folder_name = os.path.join(NUMPY_FILES_FOLDER_PATH, folder_name)
                os.makedirs(numpy_folder_name, exist_ok=True)
                numpy_file_name = os.path.join(
                    NUMPY_FILES_FOLDER_PATH,
                    folder_name,
                    f"{folder_name.lower()}_numpy_version.npy",
                )
                create_numpy_array_file(
                    video_image_sequences_folder_name,
                    numpy_file_name,
                    resize_image_size=RESIZE_IMAGE_SIZE,
                    padding_required=False,
                    grayscale_required=True,
                )
            except Exception as e:
                print(f"error occured as:  {e}")

    for label in ["Train", "Test"]:
        create_single_npy_file(
            os.path.join(NUMPY_FILES_FOLDER_PATH, label),
            os.path.join(MODEL_DATA_FILES_FOLDER_PATH, label.lower() + "_data.npy"),
        )

    model_training_dataset = ModelDataset(
        os.path.join(MODEL_DATA_FILES_FOLDER_PATH, "train_data.npy"),
        RESIZE_IMAGE_SIZE,
    )

    model_testing_dataset = ModelDataset(
        os.path.join(MODEL_DATA_FILES_FOLDER_PATH, "test_data.npy"),
        RESIZE_IMAGE_SIZE,
    )

    train_dataset_size = int(0.8 * len(model_training_dataset))
    val_dataset_size = len(model_training_dataset) - train_dataset_size
    test_dataset_size = len(model_testing_dataset)
    print(
        f"Train Dataset Size: {train_dataset_size} | Validation Dataset Size: {val_dataset_size} | Test Dataset Size: {test_dataset_size}"
    )

    train_dataset = model_training_dataset[:train_dataset_size]
    val_dataset = model_training_dataset[train_dataset_size:]

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
    )

    test_dataloader = DataLoader(
        dataset=model_testing_dataset,
        batch_size=BATCH_SIZE,
    )

    torch.manual_seed(47)
    sdae_model = StackedDenoisingAutoencoder(1, LOSS_FUNC, LR, WEIGHT_DECAY)
    sdae_model.to(DEVICE)
    print(sdae_model)

    history_data = {"train_loss": [], "valid_loss": []}
    min_loss_so_far = 10e9

    train_losses, val_losses = [], []
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH: {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_sdae(
            model=sdae_model,
            device=DEVICE,
            dataloader=train_dataloader,
            noise_factor=NOISE_FACTOR,
        )
        valid_loss = validate_sdae(
            model=sdae_model,
            device=DEVICE,
            loss_func=LOSS_FUNC,
            dataloader=val_dataloader,
            noise_factor=NOISE_FACTOR,
        )
        history_data["train_loss"].append(train_loss)
        history_data["valid_loss"].append(valid_loss)
        if valid_loss < min_loss_so_far:
            min_loss_so_far = valid_loss
        print(f"train loss: {train_loss}")
        print(f"valid loss: {valid_loss}")
        print(f"min loss so far: {min_loss_so_far}")
        train_losses.append(train_loss)
        val_losses.append(valid_loss)

    torch.save(
        sdae_model,
        os.path.join(
            TRAINED_MODELS_PATH,
            "trained_sdae_model_" + f"_{BATCH_SIZE}_{NUM_EPOCHS}" + ".pth",
        ),
    )

    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig(os.path.join('summaries', scenario_folder_name, "train_graph.png"))
    plt.plot(val_losses)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig(os.path.join('summaries', scenario_folder_name, "train_validation_graph.png"))

    if SHOW_TEST_LOGS:
        test_sdae(
            model=sdae_model,
            device=DEVICE,
            dataloader=test_dataloader,
            loss_fn=LOSS_FUNC,
            threshold=min_loss_so_far + min_loss_so_far / 20,
            noise_factor=NOISE_FACTOR,
        )

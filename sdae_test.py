import os
import argparse
import torch
from utils import *
from sdae import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDAE Arguments")
    parser.add_argument("--model-path", type=str, help="trained model path")
    parser.add_argument("--batch-size", type=int, help="trained model path")
    parser.add_argument("--noise-factor", type=float, help="inference noise factor")
    parser.add_argument("--loss-threshold", type=float, help="reconstruction threshold")
    parser.add_argument("--video-file-path", type=str, help="video file path")
    # parser.add_argument(
    #     "--output-video-file-name", type=str, help="output video file name"
    # )
    opt = parser.parse_args()

    print(opt)

    model_path = opt.model_path
    batch_size = opt.batch_size
    loss_threshold = opt.loss_threshold
    noise_factor = opt.noise_factor
    video_file_path = opt.video_file_path
    # output_video_file_name = opt.output_video_file_name

    LOSS_FUNC = nn.MSELoss()
    RESIZE_IMAGE_SIZE = 256
    TEST_IMAGES_PATH = os.path.join(os.getcwd(), "temp_image_folder")

    device = get_device()
    sdae_model = torch.load(model_path, map_location=device)
    sdae_model.eval()
    print(sdae_model)

    vidcap = cv2.VideoCapture(video_file_path)
    success, image = vidcap.read()
    count = 0
    numpy_images = []
    reconstruction_losses = []
    abnormal_events = []
    with torch.inference_mode():
        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_numpy = image = crop_center(
                np.asarray(image, dtype=np.float32), RESIZE_IMAGE_SIZE, RESIZE_IMAGE_SIZE
            ) / 255.0
            image_numpy = np.expand_dims(image_numpy, axis=0)
            numpy_images.append(image_numpy)
            count += 1
            if count == batch_size:
                count = 0
                tensor_array = torch.from_numpy(np.array(numpy_images))
                image_noisy = add_noise(tensor_array, noise_factor)
                image_noisy = image_noisy.to(device)
                image_batch = tensor_array.to(device)
                encoded_data, reconstucted_images = sdae_model(image_noisy)
                test_loss = LOSS_FUNC(reconstucted_images, image_batch)
                reconstruction_losses.append(test_loss.cpu().item())
                print(f"Reconstruction Error with loss {test_loss}")
                if test_loss > loss_threshold:
                    print("=" * 30)
                    print("Abnormal Event Detected!\n")
                    abnormal_events.append('red')
                else:
                    abnormal_events.append('green')
                numpy_images = []
            success, image = vidcap.read()
    
    plt.plot(reconstruction_losses)
    plt.title(f'threshold {loss_threshold:.4f}')
    plt.savefig('inference.png')
    plt.show()

            

    """
     

    video_to_images(
        video_file_path,
        TEST_IMAGES_PATH,
        save_after_frame=0,
        save_before_frame=6000,
    )

    temp_numpy_path = os.path.join(os.getcwd(), "temp_numpy_folder")
    if not os.path.exists(temp_numpy_path):
        os.makedirs(temp_numpy_path, exist_ok=True)
    numpy_file_name = os.path.join(
        temp_numpy_path,
        "numpy_version.npy",
    )
    create_numpy_array_file(
        TEST_IMAGES_PATH,
        numpy_file_name,
        resize_image_size=RESIZE_IMAGE_SIZE,
        padding_required=False,
        grayscale_required=True,
    )

    model_testing_dataset = ModelDataset(
        numpy_file_name,
        RESIZE_IMAGE_SIZE,
    )

    test_dataloader = DataLoader(
        dataset=model_testing_dataset,
        batch_size=batch_size,
    )

    reconstruction_losses = test_sdae_inference(
        model=sdae_model,
        device=device,
        dataloader=test_dataloader,
        loss_fn=LOSS_FUNC,
        threshold=loss_threshold + loss_threshold / 20,
        noise_factor=noise_factor,
    )

    plt.plot(reconstruction_losses)
    plt.show()

    try:
        os.system(f"rm -rf {TEST_IMAGES_PATH}")
        os.system(f"rm -rf {temp_numpy_path}")
    except Exception as e:
        print(e)


    """

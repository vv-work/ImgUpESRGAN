import os
import sys
import glob
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import RRDBNet_arch as arch

def upscale_image(input_path, output_path, model_path, device):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(Variable(img_LR)).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
    # Set a threshold value to determine white and near-white pixels
    threshold_value = 240

    # Create a mask for white and near-white pixels
    white_mask = cv2.inRange(output, np.array([threshold_value, threshold_value, threshold_value]), np.array([255, 255, 255]))

    # Replace near-white pixels with absolute white
    output[white_mask == 255] = (255, 255, 255)

    # Apply a more aggressive median filter
    output = cv2.medianBlur(output, 5)
    

    cv2.imwrite(output_path, output)

input_folder = sys.argv[1]
output_folder = sys.argv[2]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model_path = "models/RRDB_ESRGAN_x4.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

for img_path in glob.glob(f"{input_folder}/*.*"):
    img_name = os.path.basename(img_path)
    output_path = os.path.join(output_folder, img_name)
    upscale_image(img_path, output_path, model_path, device)

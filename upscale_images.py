import os
import sys
import glob
from test import upscale_image

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

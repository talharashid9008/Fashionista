import torchvision.transforms as transforms
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'data')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu
from networks import U2NET


import torch.nn.functional as F
import torch
import os

from tqdm import tqdm
from PIL import Image
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


device = "cpu"
image_dir = "C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\model\\input_images"
result_dir = "C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\static\\segmentation_output"
checkpoint_path = os.path.join(
    "trained_checkpoint", "cloth_segm_u2net_latest.pth")
print(checkpoint_path)

do_palette = True


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
temp = 'C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\model\\trained_checkpoint\\cloth_segm_u2net_latest.pth'
net = load_checkpoint_mgpu(net, temp)
net = net.to(device)
net = net.eval()

def work():
    palette = get_palette(4)
    images_list = sorted(os.listdir(image_dir))
    pbar = tqdm(total=len(images_list))
    for image_name in images_list:
        img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
        image_tensor = transform_rgb(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()
        output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
        if do_palette:
            output_img.putpalette(palette)
        output_img.save(os.path.join(result_dir, image_name[:-3] + "png"))
        mask_img = Image.open(os.path.join(result_dir, image_name[:-3] + "png")).convert("RGB")
        mask = np.array(mask_img)
        inp = np.array(img)
        for s in range(4):
                r=palette[s*3+0]
                g=palette[s*3+1]
                b=palette[s*3+2]
                if(r==g==b==0): 
                    continue 
                new_array = np.zeros_like(inp)
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        r1 , g1 , b1 = mask[i,j]
                        r1=int(r1)
                        g1=int(g1)
                        b1=int(b1)
                        if(r1 == r and g1 == g and b1 == b):
                            new_array[i,j]=inp[i,j]
                result_image = Image.fromarray(new_array)
                if not( len(np.unique(new_array))==1 and np.unique(new_array)[0]==0):
                    result_image.save(os.path.join(result_dir, f"{image_name[:-4]}_{s}.png"))







        
# import torchvision.transforms as transforms
# import sys
# import os
# # sys.path.append('C:\\Users\\ahmer\\Desktop\\FYP1\\model\\data')
# # sys.path.append('C:\\Users\\ahmer\\Desktop\\FYP1\\model\\utils')

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'data')))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
# from data.base_dataset import Normalize_image
# from utils.saving_utils import load_checkpoint_mgpu
# from networks import U2NET


# import torch.nn.functional as F
# import torch
# import os

# from tqdm import tqdm
# from PIL import Image
# import numpy as np

# import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)


# device = "cpu"
# image_dir = "C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\model\\input_images"
# result_dir = "C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\static\\segmentation_output"
# checkpoint_path = os.path.join(
#     "trained_checkpoint", "cloth_segm_u2net_latest.pth")
# print(checkpoint_path)

# do_palette = True


# def get_palette(num_cls):
#     """Returns the color map for visualizing the segmentation mask.
#     Args:
#         num_cls: Number of classes
#     Returns:
#         The color map
#     """
#     n = num_cls
#     palette = [0] * (n * 3)
#     for j in range(0, n):
#         lab = j
#         palette[j * 3 + 0] = 0
#         palette[j * 3 + 1] = 0
#         palette[j * 3 + 2] = 0
#         i = 0
#         while lab:
#             palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
#             palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
#             palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
#             i += 1
#             lab >>= 3
#     return palette


# transforms_list = []
# transforms_list += [transforms.ToTensor()]
# transforms_list += [Normalize_image(0.5, 0.5)]
# transform_rgb = transforms.Compose(transforms_list)

# net = U2NET(in_ch=3, out_ch=4)
# # net = load_checkpoint_mgpu(net, checkpoint_path)
# temp = 'C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\model\\trained_checkpoint\\cloth_segm_u2net_latest.pth'
# net = load_checkpoint_mgpu(net, temp)
# net = net.to(device)
# net = net.eval()

# def work():
#     palette = get_palette(4)

#     images_list = sorted(os.listdir(image_dir))
#     pbar = tqdm(total=len(images_list))
#     for image_name in images_list:
#         img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
#         image_tensor = transform_rgb(img)
#         image_tensor = torch.unsqueeze(image_tensor, 0)

#         output_tensor = net(image_tensor.to(device))
#         output_tensor = F.log_softmax(output_tensor[0], dim=1)
#         output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
#         output_tensor = torch.squeeze(output_tensor, dim=0)
#         output_tensor = torch.squeeze(output_tensor, dim=0)
#         output_arr = output_tensor.cpu().numpy()

#         output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
#         if do_palette:
#             output_img.putpalette(palette)
#         output_img.save(os.path.join(result_dir, image_name[:-3] + "png"))

#         # Classify mask based on color palette
#         mask = np.array(output_img)
#         unique_colors = np.unique(mask)

#         for i, color in enumerate(unique_colors):
#             if color == 0:
#                 continue
#             color_indices = np.where(mask == color)
#             color_pixels = img.crop(
#                 (min(color_indices[1]), min(color_indices[0]), max(color_indices[1]), max(color_indices[0])))
#             color_pixels.save(os.path.join(result_dir, f"{image_name[:-4]}_{i}.png"))


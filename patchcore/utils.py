import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
##############################
import torch.nn.functional as F
##############################
import tqdm
import cv2
from torchvision import transforms

LOGGER = logging.getLogger(__name__)


transform_img = [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ]
transform_img = transforms.Compose(transform_img)

def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
    image_type='png'
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        # image = PIL.Image.open(image_path).convert("RGB")
        # # image = image_transform(image)
        # image = transform_img(image)
        image = cv2.imread(image_path)
        image = cv2.resize(image,(256,256))
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros([3,256,256])

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        hoi = heatmap_on_image(cv2heatmap(segmentation*255),image)
        # print(image.shape)
        # print(mask.shape)
        # print(segmentation.shape)
        # (3, 256, 256)
        # (3, 256, 256)
        # (256, 256)
        cv2.imwrite(savename.replace('.'+image_type,'_org.'+image_type),image)
        cv2.imwrite(savename.replace('.'+image_type,'_mask.'+image_type),mask.transpose(1, 2, 0)*255)
        cv2.imwrite(savename.replace('.'+image_type,'_segmentation.'+image_type),segmentation*255)
        cv2.imwrite(savename.replace('.'+image_type,'_hoi.'+image_type),hoi)
        # f, axes = plt.subplots(1, 2 + int(masks_provided))
        # axes[0].imshow(image.transpose(1, 2, 0))
        # axes[1].imshow(mask.transpose(1, 2, 0))
        # axes[2].imshow(segmentation)
        # f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        # f.tight_layout()
        # f.savefig(savename)
        # plt.close()

def cv2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.
    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
    predict=False,
    predict_2 = False
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    if predict:
        savename = os.path.join(results_path, "results_predict.csv")
    elif predict_2:
        savename = os.path.join(results_path, "results_predict2.csv")
    else:
        savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics

########################################
class CrossAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # 线性变换层
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        
        # 输出线性层
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        query: (batch_size, target_seq_len, embed_dim)
        key: (batch_size, source_seq_len, embed_dim)
        value: (batch_size, source_seq_len, embed_dim)
        mask: (batch_size, target_seq_len, source_seq_len) 可选
        """
        batch_size, target_seq_len, embed_dim = query.size()
        _, source_seq_len, _ = key.size()
        # 线性变换
        query = self.query(query).view(batch_size, target_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch_size, source_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch_size, source_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        # 加权求和
        output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous().view(batch_size, target_seq_len, embed_dim)
        # 输出线性变换
        output = self.out_proj(output)
        return output, attn_weights
#####################################

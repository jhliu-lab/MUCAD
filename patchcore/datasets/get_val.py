import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageEnhance
import shutil
import torch

def generate_anomaly(image_path, output_size=(256, 256)):
    """
    生成异常图像及掩码
    参数:
        image_path: 输入图像路径
        output_size: 输出图像尺寸
    返回:
        anomaly_img: 异常图像 (numpy数组)
        mask: 异常掩码 (numpy数组)
        anomaly_type: 生成的异常类型
    """
    # 读取图像并调整尺寸
    img = Image.open(image_path).convert('RGB')
    # img = img.resize(output_size)
    width, height = img.size
    
    # 创建全黑掩码（单通道）
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 随机选择异常类型
    # anomaly_type = random.choice([
    #     'cutpaste', 'cutpaste_rotated', 'noise', 
    #     'cutout', 'affine', 'color_jitter', 'random_shape'
    # ])
    anomaly_type = random.choice([
        'cutpaste', 
        #'cutpaste_rotated', 
        'noise', 
        #'cutout', 
        'affine', 
        #'color_jitter', 
        #'random_shape'
    ])
    
    if anomaly_type == 'cutpaste':
        # CutPaste: 剪切并粘贴区域
        w, h = random.randint(10, width//4), random.randint(10, height//4)
        x1, y1 = random.randint(0, width-w), random.randint(0, height-h)
        x2, y2 = random.randint(0, width-w), random.randint(0, height-h)
        
        # 剪切区域
        region = img.crop((x1, y1, x1+w, x1+h))
        # 粘贴到新位置
        img.paste(region, (x2, y2))
        # 更新掩码
        draw.rectangle([x2, y2, x2+w, y2+h], fill=255)

    elif anomaly_type == 'cutpaste_rotated':
        # 旋转版CutPaste
        w, h = random.randint(15, width//3), random.randint(15, height//3)
        x1, y1 = random.randint(0, width-w), random.randint(0, height-h)
        x2, y2 = random.randint(0, width-w), random.randint(0, height-h)
        
        region = img.crop((x1, y1, x1+w, x1+h))
        # 随机旋转
        region = region.rotate(random.randint(15, 345), expand=True)
        # 粘贴
        img.paste(region, (x2, y2))
        # 获取旋转后区域的实际边界
        bbox = region.getbbox()
        if bbox:
            w_rot, h_rot = bbox[2]-bbox[0], bbox[3]-bbox[1]
            draw.rectangle([x2, y2, x2+w_rot, y2+h_rot], fill=255)

    elif anomaly_type == 'noise':
        # 添加局部高斯噪声
        w, h = random.randint(10, width//2), random.randint(10, height//2)
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        
        # 转换为numpy操作
        img_np = np.array(img)
        region = img_np[y:y+h, x:x+w]
        # 添加高斯噪声
        noise = np.random.normal(0, 50, region.shape).astype(np.uint8)
        noisy_region = np.clip(region + noise, 0, 255)
        img_np[y:y+h, x:x+w] = noisy_region
        img = Image.fromarray(img_np)
        # 更新掩码
        draw.rectangle([x, y, x+w, y+h], fill=255)

    elif anomaly_type == 'cutout':
        # 矩形遮挡
        w, h = random.randint(15, width//3), random.randint(15, height//3)
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        
        # 使用随机颜色填充
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_rect = ImageDraw.Draw(img)
        draw_rect.rectangle([x, y, x+w, y+h], fill=color)
        # 更新掩码
        draw.rectangle([x, y, x+w, y+h], fill=255)

    elif anomaly_type == 'affine':
        # 局部仿射变换
        w, h = random.randint(20, width//2), random.randint(20, height//2)
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        
        # 提取区域并旋转
        region = img.crop((x, y, x+w, y+h))
        angle = random.randint(-45, 45)
        region = region.rotate(angle, resample=Image.BILINEAR)
        # 粘贴回原图
        img.paste(region, (x, y))
        # 更新掩码
        draw.rectangle([x, y, x+w, y+h], fill=255)

    elif anomaly_type == 'color_jitter':
        # 局部颜色变换
        w, h = random.randint(20, width//2), random.randint(20, height//2)
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        
        # 提取区域
        region = img.crop((x, y, x+w, y+h))
        # 随机调整亮度
        enhancer = ImageEnhance.Brightness(region)
        region = enhancer.enhance(random.uniform(0.3, 1.8))
        # 随机调整饱和度
        enhancer = ImageEnhance.Color(region)
        region = enhancer.enhance(random.uniform(0.2, 2.0))
        # 粘贴回原图
        img.paste(region, (x, y))
        # 更新掩码
        draw.rectangle([x, y, x+w, y+h], fill=255)

    elif anomaly_type == 'random_shape':
        # 随机形状异常（组合多个小矩形）
        num_rects = random.randint(3, 8)
        start_x, start_y = random.randint(0, width-20), random.randint(0, height-20)
        x, y = start_x, start_y
        
        for _ in range(num_rects):
            w, h = random.randint(5, width//6), random.randint(5, height//6)
            # 随机颜色填充
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw_rect = ImageDraw.Draw(img)
            draw_rect.rectangle([x, y, x+w, y+h], fill=color)
            # 更新掩码
            draw.rectangle([x, y, x+w, y+h], fill=255)
            
            # 随机移动到相邻位置
            direction = random.choice(['right', 'down', 'left', 'up'])
            if direction == 'right' and x + w + 5 < width:
                x += w + 2
            elif direction == 'down' and y + h + 5 < height:
                y += h + 2
            elif direction == 'left' and x - w - 5 > 0:
                x -= w + 2
            elif direction == 'up' and y - h - 5 > 0:
                y -= h + 2

    # 转换为numpy数组返回
    anomaly_img = np.array(img)
    mask = np.array(mask)
    
    return anomaly_img, mask, anomaly_type

def add_random_noise(image_path, noise_type='gaussian', noise_level=0.01, num_clusters=3, 
                     center_radius_ratio=0.5, device='cpu', min_axis=10, max_axis_ratio=0.3):
    """
    在图像中添加随机噪声，并生成噪声掩码。
    :param image_path: 输入图像路径
    :param noise_type: 噪声类型，可选值为 'gaussian', 'salt_and_pepper', 'uniform'
    :param noise_level: 噪声强度
    :param num_clusters: 椭圆区域的数量
    :param center_radius_ratio: 中心圆形区域的半径占图像最小边的比例
    :param device: 设备类型，可选值为 'cpu' 或 'cuda'
    :param min_axis: 椭圆最小轴长度（像素）
    :param max_axis_ratio: 椭圆最大轴长度相对于中心区域半径的比例
    :return: 带噪声的图像和噪声掩码
    """
    # 读取图像并转换为numpy数组
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image, dtype=np.float32) / 255.0  # 确保图像在 [0, 1] 范围内
    H, W, C = image_np.shape

    # 创建中心圆形区域掩码
    center_x, center_y = W // 2, H // 2
    min_dim = min(W, H)
    center_radius = int(min_dim * center_radius_ratio)
    center_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(center_mask, (center_x, center_y), center_radius, 1, -1)

    # 创建噪声掩码（初始为全黑）
    noise_mask = np.zeros((H, W), dtype=np.uint8)

    # 在中心圆形区域内生成椭圆噪声区域
    max_axis = max(int(center_radius * max_axis_ratio), 100)
    for _ in range(num_clusters):
        # 在中心圆形区域内随机生成椭圆中心点
        while True:
            rand_x = np.random.randint(center_x - center_radius, center_x + center_radius)
            rand_y = np.random.randint(center_y - center_radius, center_y + center_radius)
            # 确保点在圆形区域内
            if (rand_x - center_x)**2 + (rand_y - center_y)**2 <= center_radius**2:
                break
        
        # 随机生成椭圆参数
        a = np.random.randint(min_axis, max_axis)  # 长轴
        b = np.random.randint(min_axis, max_axis)  # 短轴
        angle = np.random.uniform(0, 180)  # 旋转角度（度）
        
        # 在噪声掩码上绘制椭圆（白色）
        cv2.ellipse(noise_mask, (rand_x, rand_y), (a, b), angle, 0, 360, 255, -1)

    # 确保椭圆在中心圆形区域内
    noise_mask = noise_mask & center_mask

    # 将噪声掩码转换为浮点型用于后续计算
    noise_mask_float = noise_mask.astype(np.float32) / 255.0
    noise_mask_3d = noise_mask_float[:, :, np.newaxis]

    # 生成噪声
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, (H, W, C))
    elif noise_type == 'salt_and_pepper':
        noise = np.zeros((H, W, C))
        salt_pepper_mask = np.random.rand(H, W, C) < noise_level
        salt_mask = salt_pepper_mask & (np.random.rand(H, W, C) < 0.5)
        pepper_mask = salt_pepper_mask & ~salt_mask
        noise[salt_mask] = 1.0
        noise[pepper_mask] = -1.0
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level, noise_level, (H, W, C))
    else:
        raise ValueError("不支持的噪声类型")

    # 应用噪声掩码并添加到原图
    noisy_image = image_np + noise * noise_mask_3d
    noisy_image = np.clip(noisy_image, 0, 1) * 255.0

    return noisy_image.astype(np.uint8), noise_mask * 255.0

def clear_directory(directory):
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误：目录 {directory} 不存在")
        return

    # 遍历目录下的所有文件和子目录
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        # 如果是文件，直接删除
        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
                print(f"已删除文件：{item_path}")
            except Exception as e:
                print(f"删除文件 {item_path} 时出错：{e}")
        # 如果是子目录，跳过（不删除子目录）
        elif os.path.isdir(item_path):
            print(f"跳过子目录：{item_path}")

def get_normal(image_idx, input_path, output_path):
    for i in image_idx:
      source = input_path + "0" + str(i)+".png"
      try:
        shutil.copy2(source, output_path + "0" + str(i)+".png")
      except Exception as e:
          print(e)
    
def process_folder(input_folder, output_img_folder, output_img_good_folder, output_mask_folder, anomaly_ratio=0.3):
    """
    处理文件夹中的所有图片
    参数:
        input_folder: 输入图片文件夹路径
        output_img_folder: 输出图像保存路径
        output_mask_folder: 输出掩码保存路径
        anomaly_ratio: 生成异常图片的比例
    """
    # 创建输出文件夹
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_img_good_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_images = len(image_files)
    
    # 计算需要生成异常的数量
    num_anomaly = min(5,int(total_images * anomaly_ratio))
    num_normal = total_images - num_anomaly
    
    # 随机选择哪些图片生成异常
    indices = list(range(total_images))
    random.shuffle(indices)
    anomaly_indices = set(indices[:num_anomaly])
    print(f"处理文件夹: {input_folder}")
    print(f"总图片数: {total_images}, 异常图片: {num_anomaly}, 正常图片: {num_normal}")
    # 处理每张图片
    normal_num = 5
    for i, filename in enumerate(image_files):
        input_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        
        # 决定是否生成异常
        if i in anomaly_indices:
            # 生成异常图片和掩码
            #img, mask, anomaly_type = generate_anomaly(input_path)
            img, mask = add_random_noise(image_path=input_path,
                                         noise_type='gaussian', 
                                         noise_level=255*0.1, 
                                         num_clusters=3,
                                         center_radius_ratio=0.35, 
                                         device='cpu', 
                                         min_axis=10, 
                                         max_axis_ratio=0.5)
            
            # 保存异常图片
            img_output_path = os.path.join(output_img_folder, f"{base_name}.png")
            cv2.imwrite(img_output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # 保存掩码
            mask_output_path = os.path.join(output_mask_folder, f"{base_name}_mask.png")
            cv2.imwrite(mask_output_path, mask)
            
            #print(f"生成异常: {filename} -> 类型: {anomaly_type}")
        elif normal_num > 0:
            try:
                shutil.copy2(input_path, output_img_good_folder + filename)
            except Exception as e:
                print(e)
            normal_num -= 1

if __name__ == "__main__":
    names = ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]
    for i in names:
      # 配置路径
      input_folder = f"/home/ljh/datasets/mvtec2d/{i}/train/good/"  # 替换为您的输入图片文件夹
      output_img_folder = f"/home/ljh/datasets/mvtec2d/{i}/val/bad/"  # 替换为异常图片保存路径
      output_img_good_folder = f"/home/ljh/datasets/mvtec2d/{i}/val/good/"
      output_mask_folder = f"/home/ljh/datasets/mvtec2d/{i}/ground_truth/bad/"    # 替换为掩码图片保存路径
      os.makedirs(output_img_folder, exist_ok=True)
      os.makedirs(output_img_good_folder, exist_ok=True)
      os.makedirs(output_mask_folder, exist_ok=True)
      clear_directory(output_img_folder)
      clear_directory(output_mask_folder)
      clear_directory(output_img_good_folder)
      # 处理文件夹中的所有图片（80%生成异常，20%保持正常）
      process_folder(input_folder, output_img_folder, output_img_good_folder, output_mask_folder, anomaly_ratio=0.3)
      
      print("处理完成！")
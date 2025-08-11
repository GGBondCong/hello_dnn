import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np

class InvertIfBright(transforms.RandomApply):
    """
    如果图像的平均明度超过某个阈值，则反转其颜色。
    """
    def __init__(self, brightness_threshold=150):
        # 继承自 RandomApply，但我们在这里只是利用它的结构
        # 实际逻辑在 transform 中实现
        super().__init__(transforms=None, p=1.0)
        self.brightness_threshold = brightness_threshold
        self.invert_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 1 - x), transforms.ToPILImage()])

    def __call__(self, img):
        """
        在图像明度超过阈值时反转颜色。
        
        Args:
            img (PIL Image or Tensor): 输入图像。

        Returns:
            PIL Image or Tensor: 处理后的图像。
        """
        # 确保输入是 PIL Image
        if not isinstance(img, Image.Image):
            raise TypeError('Input must be a PIL.Image object.')

        # 1. 计算图像的平均明度
        # 将图像转换为灰度图可以方便地计算明度
        grayscale_img = img.convert('L')
        average_brightness = np.array(grayscale_img).mean()

        # 2. 判断明度是否超过阈值
        if average_brightness > self.brightness_threshold:
            # print(f"图像明度 {average_brightness:.2f} 超过阈值 {self.brightness_threshold}，进行反转。")
            # 使用 PIL.ImageOps.invert 进行颜色反转
            return ImageOps.invert(img)
        else:
            # print(f"图像明度 {average_brightness:.2f} 未超过阈值 {self.brightness_threshold}，不反转。")
            return img
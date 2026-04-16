# backend/leaf_detector.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2


class LeafDetector:
    """玉米叶片精确检测器"""

    def __init__(self, use_deep_features=False, model_path=None):
        """
        Args:
            use_deep_features: 是否使用深度学习特征（更准确但更慢）
            model_path: 预训练的CNN特征提取器路径
        """
        self.color_thresholds = {
            'green_mean': (50, 150),
            'green_ratio': 0.15,
            'saturation': (30, 200)
        }

        # 玉米叶片的典型特征参数
        self.corn_leaf_features = {
            'aspect_ratio_range': (3.0, 12.0),  # 长宽比范围（玉米叶片细长）
            'solidity_range': (0.85, 0.98),  # 坚实度（叶片通常较完整）
            'hu_moments_range': (0.1, 0.5),  # Hu矩范围
            'hough_line_threshold': 50  # 霍夫变换检测直线的阈值
        }

        self.use_deep_features = use_deep_features

        if use_deep_features and model_path:
            # 可选：加载一个轻量级CNN作为特征提取器
            self.feature_extractor = self._load_feature_extractor(model_path)

    def _load_feature_extractor(self, model_path):
        """加载预训练的特征提取器（可选）"""
        import torchvision.models as models
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model

    def detect_corn_leaf(self, image):
        """
        综合判断是否为玉米叶片

        Returns:
            (is_corn_leaf, confidence, details)
        """
        # 确保图像是 RGB 格式
        if isinstance(image, np.ndarray):
            img_array = image
        else:
            img_array = np.array(image)

        # 1. 颜色筛选（快速过滤）
        color_check, color_info = self._color_filter(img_array)
        if not color_check:
            return False, 0.0, {'reason': 'color_filter_failed', **color_info}

        # 2. 形态学分析（提取轮廓）
        contour_result = self._morphological_analysis(img_array)
        if contour_result is None:
            return False, 0.0, {'reason': 'no_valid_contour', 'color_info': color_info}

        contours, contour_info = contour_result

        # 3. 特征评分
        feature_scores = []

        # 长宽比评分
        aspect_score = self._score_aspect_ratio(contour_info['aspect_ratio'])
        feature_scores.append(aspect_score)

        # 坚实度评分
        solidity_score = self._score_solidity(contour_info['solidity'])
        feature_scores.append(solidity_score)

        # 纹理评分（叶片有平行叶脉）
        texture_score = self._analyze_texture(img_array, contour_info['mask'])
        feature_scores.append(texture_score)

        # 平行边缘检测（玉米叶片的重要特征）
        edge_score = self._detect_parallel_edges(img_array, contour_info['mask'])
        feature_scores.append(edge_score)

        # 综合置信度
        confidence = np.mean(feature_scores)
        is_corn_leaf = confidence > 0.6  # 阈值可调

        details = {
            'color_info': color_info,
            'contour_info': contour_info,
            'aspect_ratio_score': aspect_score,
            'solidity_score': solidity_score,
            'texture_score': texture_score,
            'edge_score': edge_score,
            'confidence': confidence,
            'is_corn_leaf': is_corn_leaf
        }

        return is_corn_leaf, confidence, details

    def _color_filter(self, img_array):
        """颜色筛选"""
        # 转换为HSV色彩空间
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # 绿色范围
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(img_hsv, lower_green, upper_green)

        # 计算绿色比例 - 修复：使用浮点数除法
        green_ratio = float(np.sum(green_mask > 0)) / float(green_mask.size)

        # 饱和度分析
        saturation = img_hsv[:, :, 1]
        sat_mean = float(np.mean(saturation))

        # 判断是否为叶片 - 修复：使用明确的布尔值
        is_green = (green_ratio > 0.15) and (sat_mean > 30)

        return is_green, {
            'green_ratio': green_ratio,
            'saturation_mean': sat_mean
        }

    def _morphological_analysis(self, img_array):
        """形态学分析，提取叶片轮廓"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # 高斯模糊去噪
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 边缘检测
            edges = cv2.Canny(blurred, 50, 150)

            # 形态学闭运算填充小孔
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 寻找轮廓
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # 选择面积最大的轮廓（假设为主要物体）
            largest_contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(largest_contour))

            # 如果面积太小，认为不是有效叶片
            if area < 100:
                return None

            # 最小外接矩形
            rect = cv2.minAreaRect(largest_contour)
            width, height = rect[1]

            # 防止除零错误
            if width == 0 or height == 0:
                return None

            if width > height:
                aspect_ratio = float(width) / float(height)
            else:
                aspect_ratio = float(height) / float(width)

            # 凸包（计算坚实度）
            hull = cv2.convexHull(largest_contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 0:
                solidity = area / hull_area
            else:
                solidity = 0.0

            # 创建掩码
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)

            return largest_contour, {
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'mask': mask,
                'contour': largest_contour
            }
        except Exception as e:
            print(f"形态学分析错误: {e}")
            return None

    def _score_aspect_ratio(self, aspect_ratio):
        """长宽比评分（玉米叶片通常细长）"""
        min_ratio, max_ratio = self.corn_leaf_features['aspect_ratio_range']

        if aspect_ratio < min_ratio:
            return max(0.0, float(aspect_ratio) / float(min_ratio) * 0.5)
        elif aspect_ratio > max_ratio:
            return max(0.0, 1.0 - (float(aspect_ratio) - float(max_ratio)) / float(max_ratio))
        else:
            # 在理想范围内，线性映射到0.6-1.0
            return 0.6 + (float(aspect_ratio) - float(min_ratio)) / (float(max_ratio) - float(min_ratio)) * 0.4

    def _score_solidity(self, solidity):
        """坚实度评分（叶片边缘完整）"""
        min_solid, max_solid = self.corn_leaf_features['solidity_range']

        if solidity < min_solid:
            return float(solidity) / float(min_solid) * 0.5
        elif solidity > max_solid:
            return 1.0
        else:
            return 0.5 + (float(solidity) - float(min_solid)) / (float(max_solid) - float(min_solid)) * 0.5

    def _analyze_texture(self, img_array, mask):
        """纹理分析：使用方差和梯度代替LBP"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # 只分析叶片区域
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

            # 计算梯度（边缘强度）
            grad_x = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # 叶片区域内的像素
            leaf_pixels = masked_gray[mask > 0]
            grad_pixels = gradient_magnitude[mask > 0]

            if len(leaf_pixels) == 0:
                return 0.3

            # 纹理均匀性（方差越小越均匀）
            texture_variance = float(np.var(leaf_pixels))
            mean_gradient = float(np.mean(grad_pixels))

            # 玉米叶片通常纹理均匀且有方向性梯度
            if texture_variance < 2000 and mean_gradient > 30:
                return 0.8
            elif texture_variance < 4000 and mean_gradient > 20:
                return 0.6
            else:
                return 0.4
        except Exception as e:
            print(f"纹理分析错误: {e}")
            return 0.4

    def _detect_parallel_edges(self, img_array, mask):
        """检测平行边缘（玉米叶片的重要特征）"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

            # 霍夫变换检测直线
            edges = cv2.Canny(masked_gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

            if lines is None or len(lines) < 2:
                return 0.2

            # 计算直线角度
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

            # 检查是否有相近方向的直线（平行线）
            angles = np.array(angles)
            parallel_count = 0
            for i in range(len(angles)):
                for j in range(i + 1, len(angles)):
                    angle_diff = abs(angles[i] - angles[j])
                    if angle_diff < 10 or abs(angle_diff - 180) < 10:  # 平行或共线
                        parallel_count += 1

            # 评分
            if parallel_count > 10:
                return 0.9
            elif parallel_count > 5:
                return 0.7
            elif parallel_count > 2:
                return 0.5
            else:
                return 0.3
        except Exception as e:
            print(f"平行边缘检测错误: {e}")
            return 0.3

    # 保持向后兼容的接口
    def is_leaf(self, image):
        """向后兼容的接口，现在返回更精确的结果"""
        is_corn, confidence, details = self.detect_corn_leaf(image)

        # 构建兼容的返回格式
        leaf_info = {
            'green_ratio': details['color_info']['green_ratio'],
            'saturation_mean': details['color_info']['saturation_mean'],
            'is_leaf': is_corn,
            'confidence': confidence,
        }

        # 添加可能存在的字段
        if 'aspect_ratio' in details.get('contour_info', {}):
            leaf_info['aspect_ratio'] = details['contour_info']['aspect_ratio']
        if 'solidity' in details.get('contour_info', {}):
            leaf_info['solidity'] = details['contour_info']['solidity']

        return is_corn, leaf_info
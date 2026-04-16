import torch
import torch.nn as nn
from torchvision import transforms, models
import json
import os
import numpy as np
from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image

class MobileNet_ResNet_UltraLight(nn.Module):
    """
    完全匹配训练权重的模型结构
    """

    def __init__(self, num_classes=7):
        super().__init__()

        print("\n📥 加载超轻量版本...")

        # MobileNetV3-Small
        print("  加载MobileNetV3-Small...")
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # 保留完整的features
        self.mobilenet_features = mobilenet.features

        # MobileNet输出降维
        self.mobilenet_reduce = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # ResNet50
        print("  加载完整ResNet50...")
        resnet_full = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # ResNet的各个stage
        self.resnet_stage1 = nn.Sequential(
            resnet_full.conv1,
            resnet_full.bn1,
            resnet_full.relu,
            resnet_full.maxpool
        )
        self.resnet_stage2 = resnet_full.layer1  # 256通道
        self.resnet_stage3 = resnet_full.layer2  # 512通道
        self.resnet_stage4 = resnet_full.layer3  # 1024通道

        # ResNet输出降维
        self.resnet_reduce = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 全局池化 - 用于后续处理
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 融合层 - 完全匹配权重文件的结构
        self.fusion = nn.Sequential(
            # fusion.0: Linear层 (768 -> 384)
            nn.Linear(768, 384),
            # fusion.1: BatchNorm1d
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            # fusion.4: Linear层 (384 -> 192)
            nn.Linear(384, 192),
            # fusion.5: BatchNorm1d
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            # fusion.8: 最后的分类层 (192 -> num_classes)
            nn.Linear(192, num_classes)
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  总参数量: {total_params / 1e6:.2f}M")
        print(f"  模型大小: {total_params * 4 / (1024 * 1024):.2f} MB")

    def forward(self, x):
        # MobileNet分支
        x_mob = self.mobilenet_features(x)
        x_mob = self.global_pool(x_mob)
        x_mob = x_mob.view(x_mob.size(0), -1)  # [B, 576]
        x_mob = self.mobilenet_reduce[0](x_mob.unsqueeze(-1).unsqueeze(-1))  # Conv2d需要4D
        x_mob = x_mob.squeeze(-1).squeeze(-1)  # [B, 256]

        # ResNet分支
        x_res = self.resnet_stage1(x)
        x_res = self.resnet_stage2(x_res)
        x_res = self.resnet_stage3(x_res)
        x_res = self.resnet_stage4(x_res)
        x_res = self.global_pool(x_res)
        x_res = x_res.view(x_res.size(0), -1)  # [B, 1024]
        x_res = self.resnet_reduce[0](x_res.unsqueeze(-1).unsqueeze(-1))
        x_res = x_res.squeeze(-1).squeeze(-1)  # [B, 512]

        # 融合
        x = torch.cat([x_res, x_mob], dim=1)  # [B, 768]
        x = self.fusion(x)  # [B, num_classes]

        return x


class PestDiseaseClassifier:
    def __init__(self, model_path='models/best_model.pth', config_path='models/config.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 类别名称
        self.class_names = [
            'common_rust', 'fall_army_worm', 'healthy', 'leaf_blight',
            'leaf_spot', 'oxya', 'streak_virus'
        ]

        # 中文显示名称
        self.class_names_cn = {
            'common_rust': '普通锈病', 'fall_army_worm': '草地贪夜蛾', 'healthy': '健康',
            'leaf_blight': '叶枯病', 'leaf_spot': '叶斑病', 'oxya': '玉米蝗', 'streak_virus': '条纹病毒病'
        }

        # 类别颜色映射
        self.class_colors = {
            'common_rust': '#FF6B6B', 'fall_army_worm': '#4ECDC4', 'healthy': '#95E1D3',
            'leaf_blight': '#FFA07A', 'leaf_spot': '#C06C84', 'oxya': '#6C5B7B', 'streak_virus': '#F8B195'
        }

        # 初始化模型
        print("初始化模型结构...")
        self.model = MobileNet_ResNet_UltraLight(num_classes=len(self.class_names))

        # 加载训练好的权重
        if os.path.exists(model_path):
            print(f"加载模型权重: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)

                # 权重文件直接就是state_dict
                state_dict = checkpoint
                print(f"state_dict 包含 {len(state_dict)} 个键")

                # 获取模型期望的state_dict
                model_state_dict = self.model.state_dict()

                # 创建映射字典
                mapping = {
                    # MobileNet部分
                    'mobilenet_features.': 'mobilenet.features.',
                    # MobileNet降维
                    'mobilenet_reduce.0.weight': 'mobilenet_reduce.0.weight',
                    'mobilenet_reduce.0.bias': 'mobilenet_reduce.0.bias',
                    'mobilenet_reduce.1.weight': 'mobilenet_reduce.1.weight',
                    'mobilenet_reduce.1.bias': 'mobilenet_reduce.1.bias',
                    'mobilenet_reduce.1.running_mean': 'mobilenet_reduce.1.running_mean',
                    'mobilenet_reduce.1.running_var': 'mobilenet_reduce.1.running_var',
                    'mobilenet_reduce.1.num_batches_tracked': 'mobilenet_reduce.1.num_batches_tracked',

                    # ResNet部分
                    'resnet_stage1.0.weight': 'resnet_stage1.0.weight',
                    'resnet_stage1.1.weight': 'resnet_stage1.1.weight',
                    'resnet_stage1.1.bias': 'resnet_stage1.1.bias',
                    # ... 其他ResNet层类似

                    # 融合层
                    'fusion.0.weight': 'fusion.0.weight',
                    'fusion.0.bias': 'fusion.0.bias',
                    'fusion.1.weight': 'fusion.1.weight',
                    'fusion.1.bias': 'fusion.1.bias',
                    'fusion.1.running_mean': 'fusion.1.running_mean',
                    'fusion.1.running_var': 'fusion.1.running_var',
                    'fusion.3.weight': 'fusion.4.weight',  # 注意索引偏移
                    'fusion.3.bias': 'fusion.4.bias',
                    'fusion.4.weight': 'fusion.5.weight',
                    'fusion.4.bias': 'fusion.5.bias',
                    'fusion.4.running_mean': 'fusion.5.running_mean',
                    'fusion.4.running_var': 'fusion.5.running_var',
                    'fusion.6.weight': 'fusion.8.weight',  # 最后的分类层
                    'fusion.6.bias': 'fusion.8.bias',
                }

                # 新的状态字典
                new_state_dict = {}

                # 使用映射加载权重
                for model_key in model_state_dict.keys():
                    # 跳过num_batches_tracked
                    if 'num_batches_tracked' in model_key:
                        continue
                    if model_key in state_dict:
                        if model_state_dict[model_key].shape == state_dict[model_key].shape:
                            new_state_dict[model_key] = state_dict[model_key]
                            print(f"✓ 直接匹配: {model_key}")
                            continue

                    # 尝试使用映射
                    mapped = False
                    for m_key, w_key in mapping.items():
                        if m_key in model_key:
                            w_key_full = model_key.replace(m_key, w_key)
                            if w_key_full in state_dict:
                                if model_state_dict[model_key].shape == state_dict[w_key_full].shape:
                                    new_state_dict[model_key] = state_dict[w_key_full]
                                    print(f"✓ 映射匹配: {model_key} <- {w_key_full}")
                                    mapped = True
                                    break

                    if not mapped:
                        print(f"⚠️ 未找到匹配: {model_key}")

                # 加载权重
                self.model.load_state_dict(new_state_dict, strict=False)
                print(f"\n✅ 模型权重加载成功，加载了 {len(new_state_dict)}/{len(model_state_dict)} 个参数")

                # 检查最后一层是否加载
                last_layer = 'fusion.6.weight'
                if last_layer in new_state_dict:
                    print(f"✅ 最后一层已加载: {last_layer}")
                    print(
                        f"   权重范围: [{new_state_dict[last_layer].min():.3f}, {new_state_dict[last_layer].max():.3f}]")
                else:
                    print(f"❌ 最后一层未加载")

            except Exception as e:
                print(f"❌ 模型权重加载失败: {e}")
                raise e
        else:
            print(f"❌ 警告: 模型文件不存在 {model_path}")

        self.model.to(self.device)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 添加置信度阈值
        self.min_confidence_threshold = 0.65  # 最低置信度阈值
        self.valid_leaf_threshold = 0.3  # 叶片检测阈值
        print(f"✅ 分类器初始化完成，支持 {len(self.class_names)} 个类别")

    def is_valid_leaf_image(self, image):
        """
        检测图片是否为有效的玉米叶片图片
        """
        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]

            # 1. 检查图像尺寸
            if h < 100 or w < 100:
                return False, "图片尺寸太小，请上传清晰的叶片照片", 0.0

            # 2. 检查亮度
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            if brightness < 30:
                return False, "图片太暗，请确保光线充足", 0.1
            if brightness > 230:
                return False, "图片太亮或过曝，请避免强光直射", 0.1

            # 3. 检查清晰度
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 30:
                return False, "图片模糊不清，请保持手机稳定拍摄", 0.2

            # 4. 检测绿色比例
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lower_green = np.array([30, 30, 30])
            upper_green = np.array([90, 255, 255])
            green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
            green_ratio = np.sum(green_mask > 0) / (h * w)

            if green_ratio < 0.1:
                return False, f"未检测到叶片（绿色比例仅{green_ratio:.1%}），请拍摄玉米叶片", green_ratio

            # 5. 检查颜色多样性
            color_std = np.std(img_array, axis=(0, 1)).mean()
            if color_std < 20:
                return False, "图片颜色过于单一，可能是截图或非自然图片", color_std / 255

            # 计算质量分数
            quality_score = (
                    min(brightness / 150, 1.0) * 0.2 +
                    min(laplacian_var / 200, 1.0) * 0.3 +
                    min(green_ratio / 0.5, 1.0) * 0.5
            )

            return True, "有效叶片图片", quality_score

        except Exception as e:
            print(f"图片检测出错: {e}")
            return False, f"图片处理出错: {str(e)}", 0.0

    def predict_with_validation(self, image):
        """
        带有效性检测的预测（唯一版本，已删除重复）
        """
        # 先进行图片有效性检测
        is_valid, reason, quality_score = self.is_valid_leaf_image(image)

        if not is_valid:
            return {
                'class': 'invalid',
                'class_cn': '无效图片',
                'confidence': 0.0,
                'color': '#808080',
                'warning': reason,
                'quality_score': quality_score,
                'top_predictions': [],
                'all_probabilities': [],
                'valid': False,
                'suggestion': '请拍摄清晰的玉米叶片照片'
            }

        # 进行正常预测
        result = self.predict(image)

        # 根据图片质量调整置信度
        adjusted_confidence = result['confidence'] * quality_score

        # 如果调整后的置信度低于阈值，标记为不确定
        if adjusted_confidence < self.min_confidence_threshold:
            result['class'] = 'uncertain'
            result['class_cn'] = '识别不确定'
            result['confidence'] = adjusted_confidence
            result['warning'] = f'图片质量评分较低({quality_score:.1%})，识别结果可能不准确，请重新拍摄更清晰的叶片'
            result['valid'] = False
        else:
            result['valid'] = True
            result['quality_score'] = quality_score
            result['warning'] = None

        return result

    def predict(self, image):
        """
        预测图像类别
        """
        # 预处理图像
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        pred_class = self.class_names[predicted.item()]
        confidence_score = confidence.item()

        # 获取所有类别的概率
        all_probs = probabilities.cpu().numpy()[0]

        # 按概率排序
        sorted_indices = np.argsort(all_probs)[::-1]

        # 打印调试信息
        print(f"\n预测结果:")
        print(f"  类别: {pred_class}")
        print(f"  置信度: {confidence_score:.4f}")
        print("  所有概率:")
        for i, prob in enumerate(all_probs):
            if prob > 0.01:  # 只打印大于1%的概率
                print(f"    {self.class_names[i]}: {prob:.4f}")

        return {
            'class': pred_class,
            'class_cn': self.class_names_cn[pred_class],
            'confidence': float(confidence_score),
            'color': self.class_colors.get(pred_class, '#808080'),
            'top_predictions': [
                {
                    'class': self.class_names[i],
                    'class_cn': self.class_names_cn[self.class_names[i]],
                    'probability': float(all_probs[i]),
                    'color': self.class_colors.get(self.class_names[i], '#808080')
                }
                for i in sorted_indices[:5]
            ],
            'all_probabilities': [
                {
                    'class': self.class_names[i],
                    'class_cn': self.class_names_cn[self.class_names[i]],
                    'probability': float(all_probs[i]),
                    'color': self.class_colors.get(self.class_names[i], '#808080')
                }
                for i in range(len(self.class_names))
            ]
        }

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_name': 'MobileNet_ResNet_UltraLight',
            'num_classes': len(self.class_names),
            'total_params': total_params,
            'total_params_m': total_params / 1e6,
            'trainable_params': trainable_params,
            'trainable_params_m': trainable_params / 1e6,
            'device': str(self.device),
            'classes': self.class_names,
            'classes_cn': self.class_names_cn
        }


# 单例模式
classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        classifier = PestDiseaseClassifier()
    return classifier
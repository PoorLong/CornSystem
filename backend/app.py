from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import time
import logging
from model import get_classifier
from leaf_detector import LeafDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# 创建叶片检测器实例
leaf_detector = LeafDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': '服务运行正常',
        'timestamp': time.time()
    })

# @app.route('/predict', methods=['POST'])
# def predict():
#     start_time = time.time()
#     try:
#         classifier = get_classifier()
#
#         # 获取图片
#         if 'image' in request.files:
#             file = request.files['image']
#             if not file or not allowed_file(file.filename):
#                 return jsonify({'success': False, 'error': '无效图片'}), 400
#             img_bytes = file.read()
#             image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
#         elif request.is_json:
#             data = request.get_json()
#             if 'image_base64' in data:
#                 img_data = base64.b64decode(data['image_base64'].split(',')[-1])
#                 image = Image.open(io.BytesIO(img_data)).convert('RGB')
#             else:
#                 return jsonify({'success': False, 'error': '未提供图片'}), 400
#         else:
#             return jsonify({'success': False, 'error': '不支持的请求格式'}), 400
#
#         # 叶片检测
#         is_corn_leaf, leaf_conf, leaf_details = leaf_detector.detect_corn_leaf(image)
#         if not is_corn_leaf:
#             return jsonify({
#                 'success': True,
#                 'is_corn_leaf': False,
#                 'leaf_confidence': leaf_conf,
#                 'message': '请上传玉米叶片照片',
#                 'details': leaf_details
#             })
#
#         # 病害识别
#         result = classifier.predict_with_validation(image)
#         process_time = time.time() - start_time
#
#         return jsonify({
#             'success': True,
#             'is_corn_leaf': True,
#             'leaf_confidence': leaf_conf,
#             'result': result,
#             'process_time': process_time
#         })
#
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        classifier = get_classifier()

        # 获取图片
        if 'image' in request.files:
            file = request.files['image']
            if not file or not allowed_file(file.filename):
                return jsonify({'success': False, 'error': '无效图片'}), 400
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        elif request.is_json:
            data = request.get_json()
            if 'image_base64' in data:
                img_data = base64.b64decode(data['image_base64'].split(',')[-1])
                image = Image.open(io.BytesIO(img_data)).convert('RGB')
            else:
                return jsonify({'success': False, 'error': '未提供图片'}), 400
        else:
            return jsonify({'success': False, 'error': '不支持的请求格式'}), 400

        # ========== 临时禁用叶片检测 ==========
        # is_corn_leaf, leaf_conf, leaf_details = leaf_detector.detect_corn_leaf(image)
        # if not is_corn_leaf:
        #     return jsonify({
        #         'success': True,
        #         'is_corn_leaf': False,
        #         'leaf_confidence': leaf_conf,
        #         'message': '请上传玉米叶片照片',
        #         'details': leaf_details
        #     })
        # 直接通过：
        is_corn_leaf = True
        leaf_conf = 1.0
        leaf_details = {}

        # 病害识别
        result = classifier.predict_with_validation(image)
        process_time = time.time() - start_time

        return jsonify({
            'success': True,
            'is_corn_leaf': is_corn_leaf,
            'leaf_confidence': leaf_conf,
            'result': result,
            'process_time': process_time
        })

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"预测失败: {error_detail}")
        return jsonify({'success': False, 'error': str(e), 'detail': error_detail}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    start_time = time.time()
    try:
        if 'images' not in request.files:
            return jsonify({'success': False, 'error': '未提供图片'}), 400

        files = request.files.getlist('images')
        classifier = get_classifier()

        logger.info(f"批量处理 {len(files)} 张图片")

        results = []
        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                img_bytes = file.read()
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                result = classifier.predict(image)
                results.append({
                    'filename': file.filename,
                    'result': result,
                    'index': i
                })
                logger.info(f"  [{i + 1}/{len(files)}] {file.filename} -> {result['class']}")

        process_time = time.time() - start_time

        return jsonify({
            'success': True,
            'results': results,
            'total': len(results),
            'process_time': process_time
        })

    except Exception as e:
        logger.error(f"批量预测失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def get_model_info():
    try:
        classifier = get_classifier()
        info = classifier.get_model_info()
        return jsonify({
            'success': True,
            'info': info
        })
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"启动病虫害识别后端服务，端口: {port}")
    logger.info(f"模型文件: models/best_model.pth")
    logger.info(f"配置文件: models/config.json")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
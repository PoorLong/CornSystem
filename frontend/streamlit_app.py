import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from datetime import datetime
import os

# 后端API地址（从环境变量读取，Railway部署时设置）
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000").rstrip('/')

# 页面配置
st.set_page_config(
    page_title="玉米叶片病虫害识别系统",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ED6A5E;
        font-weight: bold;
    }
    .confidence-low {
        color: #F4A261;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
def init_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'model_info' not in st.session_state:
        st.session_state.model_info = None
    if 'backend_connected' not in st.session_state:
        st.session_state.backend_connected = False

def main():
    init_session_state()

    # 侧边栏
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>🌽<br>病虫害识别系统</h1>", unsafe_allow_html=True)
        st.markdown("---")
        check_backend_connection()
        st.markdown("---")
        page = st.radio(
            "导航菜单",
            ["📤 单张识别", "📁 批量识别", "📊 识别历史", "ℹ️ 系统信息"],
            index=0
        )
        st.markdown("---")
        with st.expander("📋 支持的病虫害类型", expanded=False):
            if st.session_state.model_info and st.session_state.model_info.get('success'):
                classes_cn = st.session_state.model_info['info']['classes_cn']
                for class_name, class_cn in classes_cn.items():
                    st.caption(f"• {class_cn}")
            else:
                st.caption("• 普通锈病")
                st.caption("• 草地贪夜蛾")
                st.caption("• 健康")
                st.caption("• 叶枯病")
                st.caption("• 叶斑病")
                st.caption("• 玉米蝗")
                st.caption("• 条纹病毒病")

    if page == "📤 单张识别":
        show_single_prediction()
    elif page == "📁 批量识别":
        show_batch_prediction()
    elif page == "📊 识别历史":
        show_history()
    elif page == "ℹ️ 系统信息":
        show_system_info()

def check_backend_connection():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if response.status_code == 200:
            st.session_state.backend_connected = True
            st.success("✅ 后端服务已连接")
            if st.session_state.model_info is None:
                try:
                    info_response = requests.get(f"{BACKEND_URL}/info", timeout=3)
                    if info_response.status_code == 200:
                        st.session_state.model_info = info_response.json()
                except Exception as e:
                    st.caption(f"获取模型信息失败: {str(e)}")
        else:
            st.session_state.backend_connected = False
            st.error("❌ 后端服务异常")
    except requests.exceptions.ConnectionError:
        st.session_state.backend_connected = False
        st.error("❌ 后端服务未连接")
        st.caption("请检查后端服务是否运行")
    except Exception as e:
        st.session_state.backend_connected = False
        st.error(f"❌ 连接错误: {str(e)}")

def show_single_prediction():
    st.markdown("<h1 class='main-header'>📤 单张图片识别</h1>", unsafe_allow_html=True)
    st.info("""
    📌 **拍摄要求：**
    - 请拍摄**真实的玉米叶片**照片
    - 确保光线充足、画面清晰
    - 让叶片充满画面
    - ❌ 不要上传截图、白纸、或其他物体
    """)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 上传图片")
        uploaded_file = st.file_uploader(
            "选择一张农作物图片",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="请上传清晰的玉米叶片照片"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", use_column_width=True)

            # 简单图片质量提示
            img_array = np.array(image)
            if img_array.shape[0] < 100 or img_array.shape[1] < 100:
                st.warning("⚠️ 图片尺寸过小，可能影响识别效果")
            gray = np.mean(img_array, axis=2)
            brightness = np.mean(gray)
            if brightness < 50:
                st.warning("⚠️ 图片偏暗")
            elif brightness > 220:
                st.warning("⚠️ 图片偏亮，可能过曝")

            if st.button("🔍 开始识别", type="primary", use_container_width=True):
                if not st.session_state.backend_connected:
                    st.error("后端服务未连接，请稍后再试")
                else:
                    with st.spinner("🔄 识别中..."):
                        files = {'image': (uploaded_file.name, uploaded_file.getvalue(), 'image/jpeg')}
                        try:
                            response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=30)
                            if response.status_code == 200:
                                data = response.json()
                                if data.get('success'):
                                    # 处理叶片检测结果
                                    if not data.get('is_corn_leaf'):
                                        st.error(f"❌ 未检测到玉米叶片 (置信度: {data['leaf_confidence']:.2%})")
                                        st.info("请拍摄真实的玉米叶片照片")
                                        if 'details' in data:
                                            with st.expander("检测详情"):
                                                st.json(data['details'])
                                    else:
                                        # 叶片有效，显示病害识别结果
                                        result = data['result']
                                        # 保存历史记录
                                        st.session_state.history.append({
                                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            'filename': uploaded_file.name,
                                            'result': result,
                                            'process_time': data.get('process_time', 0),
                                            'leaf_confidence': data['leaf_confidence']
                                        })
                                        with col2:
                                            st.subheader("📊 识别结果")
                                            if not result.get('valid', True):
                                                st.error(f"❌ {result.get('warning', '图片无效')}")
                                                st.info(result.get('suggestion', '请重新拍摄清晰的玉米叶片照片'))
                                                if 'quality_score' in result:
                                                    st.caption(f"图片质量评分: {result['quality_score']:.1%}")
                                            else:
                                                display_result(result)
                                                if result.get('warning'):
                                                    st.warning(f"⚠️ {result['warning']}")
                                                else:
                                                    st.success("✅ 识别完成")
                                else:
                                    st.error(f"识别失败: {data.get('error', '未知错误')}")
                            else:
                                st.error(f"API请求失败: {response.status_code}")
                        except Exception as e:
                            st.error(f"发生错误: {str(e)}")

    with col2:
        st.subheader("📊 识别结果")
        if uploaded_file is None:
            st.info("👆 请先在左侧上传图片")
        else:
            st.info("点击识别按钮查看结果")

def display_result(result):
    confidence = result['confidence']
    if confidence > 0.8:
        conf_class = "confidence-high"
        conf_text = "高置信度"
    elif confidence > 0.5:
        conf_class = "confidence-medium"
        conf_text = "中等置信度"
    else:
        conf_class = "confidence-low"
        conf_text = "低置信度"

    st.markdown(f"""
    <div class='prediction-card'>
        <h3 style='color: {result.get("color", "#000000")};'>{result.get("class_cn", "未知")}</h3>
        <p>类别: {result.get("class", "unknown")}</p>
        <p class='{conf_class}'>置信度: {confidence:.2%} ({conf_text})</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence)

    st.subheader("📈 Top-5 预测")
    if 'top_predictions' in result:
        top = result['top_predictions']
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[p.get('class_cn', p.get('class', '')) for p in top],
            x=[p['probability'] for p in top],
            orientation='h',
            marker_color=[p.get('color', '#2E7D32') for p in top],
            text=[f"{p['probability']:.2%}" for p in top],
            textposition='outside'
        ))
        fig.update_layout(
            title="各类别概率分布",
            xaxis_title="概率",
            yaxis_title="类别",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig, use_container_width=True)

def show_batch_prediction():
    st.markdown("<h1 class='main-header'>📁 批量识别</h1>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "选择多张图片",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        accept_multiple_files=True,
        help="支持PNG、JPG、JPEG、BMP格式，可多选"
    )

    if uploaded_files:
        st.success(f"已选择 {len(uploaded_files)} 张图片")
        with st.expander("📸 图片预览", expanded=False):
            cols = st.columns(4)
            for i, file in enumerate(uploaded_files[:8]):
                with cols[i % 4]:
                    image = Image.open(file)
                    st.image(image, caption=file.name, use_column_width=True)
            if len(uploaded_files) > 8:
                st.caption(f"... 还有 {len(uploaded_files) - 8} 张未显示")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 开始批量识别", type="primary", use_container_width=True):
                if not st.session_state.backend_connected:
                    st.error("后端服务未连接，请稍后再试")
                else:
                    files = []
                    for file in uploaded_files:
                        files.append(('images', (file.name, file.getvalue(), 'image/jpeg')))
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    try:
                        status_text.text("正在上传并识别...")
                        response = requests.post(f"{BACKEND_URL}/batch_predict", files=files, timeout=120)
                        progress_bar.progress(50)
                        status_text.text("处理结果...")
                        if response.status_code == 200:
                            result = response.json()
                            if result.get('success'):
                                progress_bar.progress(100)
                                status_text.text("识别完成！")
                                st.success(f"✅ 识别完成！共 {result['total']} 张图片，耗时 {result.get('process_time', 0):.2f}秒")
                                for item in result['results']:
                                    st.session_state.history.append({
                                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'filename': item['filename'],
                                        'result': item['result']
                                    })
                                display_batch_results(result['results'])
                            else:
                                st.error(f"识别失败: {result.get('error', '未知错误')}")
                        else:
                            st.error(f"API请求失败: {response.status_code}")
                    except Exception as e:
                        st.error(f"发生错误: {str(e)}")

def display_batch_results(results):
    table_data = []
    for item in results:
        table_data.append({
            '文件名': item['filename'],
            '识别结果': item['result'].get('class_cn', '未知'),
            '置信度': f"{item['result'].get('confidence', 0):.2%}",
            '类别': item['result'].get('class', 'unknown')
        })
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)

    st.subheader("📊 识别结果统计")
    class_counts = {}
    for item in results:
        class_name = item['result'].get('class_cn', '未知')
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    if class_counts:
        fig = px.pie(values=list(class_counts.values()), names=list(class_counts.keys()), title="病虫害类型分布")
        st.plotly_chart(fig, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下载识别结果(CSV)",
        data=csv,
        file_name=f"识别结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def show_history():
    st.markdown("<h1 class='main-header'>📊 识别历史</h1>", unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("暂无识别历史记录")
        return
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        if st.button("🗑️ 清空历史", type="secondary"):
            st.session_state.history = []
            st.rerun()
    for i, record in enumerate(reversed(st.session_state.history[-20:])):
        with st.expander(f"[{record['time']}] {record['filename']} - {record['result'].get('class_cn', '未知')}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**时间**: {record['time']}")
                st.write(f"**文件名**: {record['filename']}")
                st.write(f"**识别结果**: {record['result'].get('class_cn', '未知')}")
                st.write(f"**置信度**: {record['result'].get('confidence', 0):.2%}")
                if 'leaf_confidence' in record:
                    st.write(f"**叶片置信度**: {record['leaf_confidence']:.2%}")
            with col2:
                if 'top_predictions' in record['result']:
                    df = pd.DataFrame(record['result']['top_predictions'][:3])
                    df = df.rename(columns={'class_cn': '类别', 'probability': '概率'})
                    st.dataframe(df[['类别', '概率']])

def show_system_info():
    st.markdown("<h1 class='main-header'>ℹ️ 系统信息</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📦 后端信息")
        if st.session_state.backend_connected:
            try:
                response = requests.get(f"{BACKEND_URL}/health", timeout=3)
                if response.status_code == 200:
                    health_data = response.json()
                    st.success("✅ 后端服务状态: 正常")
                    st.json(health_data)
                else:
                    st.error("❌ 后端服务状态: 异常")
            except:
                st.error("❌ 后端服务连接失败")
        else:
            st.error("❌ 后端服务未连接")
        st.subheader("🧠 模型信息")
        if st.session_state.model_info and st.session_state.model_info.get('success'):
            info = st.session_state.model_info['info']
            st.write(f"**模型名称**: {info.get('model_name', '未知')}")
            st.write(f"**设备**: {info.get('device', 'cpu')}")
            st.write(f"**类别数量**: {info.get('num_classes', 0)}")
            st.write(f"**总参数量**: {info.get('total_params_m', 0):.2f}M")
            st.write(f"**可训练参数量**: {info.get('trainable_params_m', 0):.2f}M")
            with st.expander("📋 类别详情"):
                classes_cn = info.get('classes_cn', {})
                classes_df = pd.DataFrame({
                    '中文名称': list(classes_cn.values()),
                    '英文名称': list(classes_cn.keys())
                })
                st.dataframe(classes_df, use_container_width=True)
        else:
            st.info("未获取到模型信息，请确保后端服务已启动")
    with col2:
        st.subheader("📁 数据集信息")
        classes_data = {
            '类别': ['普通锈病', '草地贪夜蛾', '健康', '叶枯病', '叶斑病', '玉米蝗', '条纹病毒病'],
            '英文名': ['common_rust', 'fall_army_worm', 'healthy', 'leaf_blight', 'leaf_spot', 'oxya', 'streak_virus']
        }
        st.dataframe(pd.DataFrame(classes_data), use_container_width=True)
        st.subheader("⚙️ 系统配置")
        config_data = {
            '配置项': ['前端框架', '后端框架', '深度学习框架', 'API地址'],
            '值': ['Streamlit', 'Flask', 'PyTorch', BACKEND_URL]
        }
        st.dataframe(pd.DataFrame(config_data), use_container_width=True)
        with st.expander("📖 使用说明"):
            st.markdown("""
            **单张识别:** 上传图片 → 点击识别 → 查看结果
            **批量识别:** 选择多张图片 → 批量识别 → 查看统计并下载CSV
            **识别历史:** 自动保存最近的识别记录
            """)

if __name__ == "__main__":
    main()
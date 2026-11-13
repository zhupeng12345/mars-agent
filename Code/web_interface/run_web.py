#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MARS-Agent 火眼检测系统 - Streamlit Web界面启动脚本
运行此脚本启动Web应用程序
"""

import os
import sys

def print_banner():
    print("=" * 60)
    print("MARS-Agent 火眼检测系统 v1.0 - Web版")
    print("基于 YOLOv10 + DC-MMGAT 的铝电解槽过热度检测")
    print("=" * 60)
    print("\n正在启动Web服务...")
    print("📌 服务启动后将自动打开浏览器")
    print("📌 如未自动打开，请手动访问: http://localhost:8501")
    print("📌 按 Ctrl+C 停止服务\n")

if __name__ == "__main__":
    print_banner()

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 通过当前 Python 解释器启动，确保网页与 LangGraph/视觉依赖使用同一环境。
    streamlit_app = os.path.join(current_dir, "streamlit_app.py")
    os.execv(sys.executable, [
        sys.executable, "-m", "streamlit", "run", streamlit_app,
        "--server.port", "8501", "--server.headless", "false",
    ])

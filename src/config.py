"""
Module Config: Quản lý cấu hình và API Key bảo mật cho hệ thống.
"""

import os
from typing import Optional


def get_gemini_api_key(user_input_key: Optional[str] = None) -> Optional[str]:
    """
    Lấy Gemini API Key theo thứ tự ưu tiên:
    1. Khóa do người dùng nhập trực tiếp trên Sidebar (nếu có)
    2. Khóa lưu trong Streamlit secrets (st.secrets["GEMINI_API_KEY"] hoặc st.secrets.get("gemini_api_key"))
    3. Khóa lưu trong biến môi trường hệ thống (os.getenv("GEMINI_API_KEY"))
    """
    # 1. Người dùng nhập trực tiếp trên UI
    if user_input_key and user_input_key.strip():
        return user_input_key.strip()

    # 2. Đọc từ Streamlit secrets (hỗ trợ Streamlit Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                key = str(st.secrets["GEMINI_API_KEY"]).strip()
                if key:
                    return key
            if "gemini_api_key" in st.secrets:
                key = str(st.secrets["gemini_api_key"]).strip()
                if key:
                    return key
    except Exception:
        pass

    # 3. Đọc từ biến môi trường OS
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    return None

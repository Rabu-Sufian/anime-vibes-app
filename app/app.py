# app/app.py

# ⛔ DO NOT run any Streamlit command before this
import sys
from pathlib import Path
# Ensure project root is in Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# ✅ MUST come before importing streamlit and any other Streamlit calls
import streamlit as st
st.set_page_config(page_title="Anime Vibes", layout="wide")

from PIL import Image
from views import dialogue_analyzer, recommender, explorer


# Load and display banner
banner_path = ROOT_DIR / "assets" / "banner.png"
if banner_path.exists():
    banner = Image.open(banner_path)
    banner = banner.resize((banner.width, 200))
    st.image(banner, use_container_width=True)

# Optional CSS injection
def local_css(file_path):
    css_path = ROOT_DIR / file_path
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# local_css("assets/style.css")

# Sidebar
with st.sidebar:
    st.markdown("### 🌸 Navigation")
    app_mode = st.radio("Go to", ["🏠 Home", "💬 Analyze", "🎯 Recommend", "📚 Explore"])

# Routing
if app_mode == "🏠 Home":
    st.title("🌸 Anime Vibes")
    st.subheader("Welcome to your cozy anime dialogue analyzer ✨")
    st.markdown("""
        Use the sidebar to:
        - 💬 Upload and explore subtitle moods
        - 🎯 Get personalized anime recommendations
        - 📚 Explore the full subtitle dataset
    """)
    st.markdown("More features coming soon... Stay comfy! ☕")

elif app_mode == "💬 Analyze":
    dialogue_analyzer.show()

elif app_mode == "🎯 Recommend":
    recommender.show()

elif app_mode == "📚 Explore":
    st.title("Coming Soon ...")

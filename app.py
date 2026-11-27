import json
import os
import pathlib
import streamlit as st
import streamlit.components.v1 as components

ROOT = pathlib.Path(__file__).parent
ART = pathlib.Path(os.environ.get("WAFER_ARTIFACTS", ROOT / "artifacts"))
TOKEN = "/*__WS_DATA__*/null"

st.set_page_config(page_title="WaferSense AI", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stDecoration"], footer { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stAppViewContainer"], .stApp { background: #050811 !important; }
iframe { border: 0 !important; display: block; width: 100% !important; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading WaferSense AI...")
def site_html():
    tpl = (ROOT / "site_template.html").read_text(encoding="utf-8")
    data = None
    web = ART / "webdata.json"
    if web.exists():
        data = web.read_text(encoding="utf-8")
    elif (ART / "model.pt").exists():
        try:
            from export_web import build_webdata  
            data = json.dumps(build_webdata(str(ART)), separators=(",", ":"))
        except ImportError:
            st.error("artifacts/webdata.json is missing. Run `python export_web.py` once on your computer "
                     "(needs torch), then commit artifacts/webdata.json and redeploy.")
            st.stop()
    assert tpl.count(TOKEN) == 1
    return tpl.replace(TOKEN, data or "null")


html = site_html()
if hasattr(st, "iframe"):
    st.iframe(html, height=900)
else:
    components.html(html, height=900, scrolling=True)

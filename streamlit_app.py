# -*- coding: utf-8 -*-
"""
SM-01: Bilingual (English/Kiswahili) Financial FAQ Chatbot for Tanzania
Streamlit app with tier + language selection, scoped FAISS retrieval per
(user_tier, language), and Groq LLM generation grounded in answer_context.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub, then deploy via share.streamlit.io
"""

import os
import html
import re
from datetime import timedelta
import numpy as np
import pandas as pd
import faiss
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq
from groq import APIError, RateLimitError
from sentence_transformers import SentenceTransformer
from groq.types.chat import ChatCompletionMessageParam

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAFIKI | Financial FAQ Chatbot",
    page_icon=":material/forum:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Material icon shortcuts (Streamlit :material/name: syntax)
ICON_BRAND = ":material/forum:"
ASSISTANT_AVATAR_ICON = "auto_awesome"

VALID_THEMES = ("system", "light", "dark")
THEME_LABELS = {
    "English": {"system": "System", "light": "Light", "dark": "Dark"},
    "Swahili": {"system": "Mfumo", "light": "Mwanga", "dark": "Giza"},
}
THEME_ICONS = {
    "system": ":material/computer:",
    "light": ":material/wb_sunny:",
    "dark": ":material/partly_cloudy_night:",
}

# Muted slate-blue accent
ACCENT = "#557A93"
ACCENT_RGB = "85, 122, 147"
ACCENT_DARK = "#4A6B80"
ACCENT_LIGHT = "#6B93AD"

# Cool light-mode palette — soft blue tints from ACCENT (scoped via [data-theme="light"])
LIGHT_MAIN_BG = "#F4F8FB"
LIGHT_SIDEBAR_BG = "#E8F1F7"
LIGHT_SURFACE = "#FAFCFE"
LIGHT_BORDER = "#C5D6E3"
LIGHT_TEXT = "#2A3844"
LIGHT_TEXT_SOFT = "#4A6272"
LIGHT_CHAT_FILL = "#EDF4F9"
LIGHT_CHAT_BTN = "#E0EBF3"
DARK_CHAT_BTN = "#363A45"

load_dotenv()

# API key: works both locally (.env via os.environ) and on Streamlit Cloud
# (via st.secrets, set in the app's "Secrets" settings panel).
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError, AttributeError):
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY not found. Locally: create a .env file with "
        "GROQ_API_KEY=your_key. On Streamlit Cloud: add it under "
        "App settings → Secrets as GROQ_API_KEY = \"your_key\"."
    )
    st.stop()

DATA_PATH = os.environ.get("SM01_DATA_PATH", "faq_qns22_csv.csv")
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 3
# L2 distances for good matches with this model are typically ~9–32 (see retrieval tests).
NO_MATCH_THRESHOLD = 32

TIERS = ["Machinga", "Micro", "SME Owner", "Employee", "Corporate"]
LANGUAGES = ["English", "Swahili"]

TIER_DESCRIPTIONS = {
    "Machinga": {
        "English": "Street vendors & informal traders — simple, practical money advice.",
        "Swahili": "Wamachinga na wafanyabiashara wadogo wa mtaani — ushauri rahisi wa fedha.",
    },
    "Micro": {
        "English": "Micro-entrepreneurs running very small registered/unregistered businesses.",
        "Swahili": "Wajasiriamali wadogo wanaoendesha biashara ndogo sana.",
    },
    "SME Owner": {
        "English": "Small & medium business owners — registration, growth, and credit topics.",
        "Swahili": "Wamiliki wa biashara ndogo na za kati — usajili, ukuaji, na mikopo.",
    },
    "Employee": {
        "English": "Salaried employees — payroll, PAYE, NSSF, NHIF, and personal finance.",
        "Swahili": "Wafanyakazi wenye mishahara — PAYE, NSSF, NHIF, na fedha za kibinafsi.",
    },
    "Corporate": {
        "English": "Corporate finance teams — tax compliance, audit, governance, statements.",
        "Swahili": "Timu za fedha za makampuni — kodi, ukaguzi, utawala, na taarifa za fedha.",
    },
}

TIER_PERSONA = {
    "Machinga": "Use very simple, warm, everyday language. Avoid jargon. Keep sentences short.",
    "Micro": "Use simple, encouraging language suited to someone running a very small business.",
    "SME Owner": "Use clear, practical business language suited to a small/medium business owner.",
    "Employee": "Use clear, professional language suited to a salaried employee asking about payroll and benefits.",
    "Corporate": "Use precise, formal, technical financial/legal language suited to a corporate finance professional.",
}

NO_MATCH_MESSAGE = {
    "English": (
        "I don't have verified information on that in my current knowledge base. "
        "Please contact the relevant institution directly (e.g. your bank, TRA, NSSF, "
        "or NHIF) or consult a licensed financial advisor for accurate guidance."
    ),
    "Swahili": (
        "Sina taarifa thabiti kuhusu hilo kwenye hifadhi yangu ya sasa ya maarifa. "
        "Tafadhali wasiliana moja kwa moja na taasisi husika (mfano benki yako, TRA, NSSF, "
        "au NHIF) au mshauri wa fedha aliyesajiliwa kwa ushauri sahihi."
    ),
}

STARTER_QUESTIONS = {
    ("Machinga", "English"): [
        "How do I keep my M-Pesa safe?",
        "How do I get a loan?",
        "How can I save money daily?",
    ],
    ("Machinga", "Swahili"): [
        "Ninawezaje kulinda M-Pesa yangu?",
        "Ninawezaje kupata mkopo?",
        "Ninawezaje kuweka akiba kila siku?",
    ],
    ("Micro", "English"): [
        "How do I register my small business?",
        "What records should I keep?",
        "How can I get a small business loan?",
    ],
    ("Micro", "Swahili"): [
        "Ninawezaje kusajili biashara yangu ndogo?",
        "Ni rekodi gani ninapaswa kuweka?",
        "Ninawezaje kupata mkopo wa biashara ndogo?",
    ],
    ("SME Owner", "English"): [
        "How can I get a loan to expand my business?",
        "How do I get a business TIN number?",
        "Do I need to register for VAT?",
    ],
    ("SME Owner", "Swahili"): [
        "Ninawezaje kupata mkopo wa kukuza biashara?",
        "Ninawezaje kupata TIN ya biashara?",
        "Je, ninahitaji kusajili VAT?",
    ],
    ("Employee", "English"): [
        "What is PAYE?",
        "How is PAYE calculated on my salary?",
        "What is NSSF and why does it matter?",
    ],
    ("Employee", "Swahili"): [
        "PAYE ni nini?",
        "PAYE inahesabiwaje kwenye mshahara wangu?",
        "NSSF ni nini na kwa nini ni muhimu?",
    ],
    ("Corporate", "English"): [
        "How often must tax returns be filed?",
        "What is transfer pricing?",
        "What are the roles of an audit committee?",
    ],
    ("Corporate", "Swahili"): [
        "Marejesho ya kodi yanapaswa kuwasilishwa mara ngapi?",
        "Bei ya uhamisho ni nini?",
        "Ni majukumu gani ya kamati ya ukaguzi?",
    ],
}

HISTORY_TURNS = 10  # 5 Q&A pairs sent to the LLM for follow-ups
EMPTY_FADE_SECONDS = 0.22


def _session_choice(key: str, valid: tuple[str, ...], default: str) -> str:
    """Return a valid session-state choice; reset key when missing, None, or invalid."""
    val = st.session_state.get(key)
    if val not in valid:
        st.session_state[key] = default
        return default
    return val


def format_display(text: str) -> str:
    """UI only: replace dashes with commas or spaces. Does not affect retrieval or LLM."""
    if not isinstance(text, str):
        return str(text)
    text = text.replace(" — ", ", ")
    text = text.replace("—", ", ")
    text = text.replace(" – ", ", ")
    text = text.replace("–", ", ")
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    return text


def sidebar_section_heading(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="rafiki-sidebar-block-heading">
          <p class="rafiki-sidebar-block-title">{html.escape(title)}</p>
          <p class="rafiki-sidebar-block-desc">{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_theme_picker(language: str) -> None:
    """Square theme tiles: icon on top, label underneath, inside the button."""
    current = _session_choice("theme_mode", VALID_THEMES, "system")
    labels = THEME_LABELS[language]
    cols = st.columns(3, gap="small")
    for col, mode in zip(cols, VALID_THEMES):
        with col:
            is_sel = current == mode
            if st.button(
                labels[mode],
                icon=THEME_ICONS[mode],
                key=f"theme_tile_{mode}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                if mode != current:
                    st.session_state.theme_mode = mode
                    sync_theme_query_param()
                    st.rerun()


def render_sidebar_profile_card(tier: str, language: str) -> None:
    description = format_display(TIER_DESCRIPTIONS.get(tier, {}).get(language, ""))
    st.markdown(
        f"""
        <div class="rafiki-sidebar-profile-card">
          <div class="rafiki-sidebar-profile-top">
            <div class="rafiki-sidebar-profile-avatar">
              {material_icon_html("person", size_px=18)}
            </div>
            <div class="rafiki-sidebar-profile-meta">
              <div class="rafiki-sidebar-profile-tier">{html.escape(tier)}</div>
              <div class="rafiki-sidebar-profile-lang">{html.escape(language)}</div>
            </div>
          </div>
          <p class="rafiki-sidebar-profile-desc">{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def material_icon_html(name: str, *, size_px: int = 20, color: str = "var(--rafiki-accent)") -> str:
    """Inline Material Symbols icon for custom HTML bubbles."""
    return (
        f'<span style="font-family:\'Material Symbols Outlined\';'
        f" font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24;"
        f' font-size:{size_px}px; color:{color}; line-height:1;'
        f' user-select:none;">{name}</span>'
    )


def render_chat_bubble(role: str, content: str) -> None:
    """Render user bubbles on the right, assistant bubbles on the left."""
    body = html.escape(format_display(content)).replace("\n", "<br>")
    if role == "user":
        st.markdown(
            f"""
            <div class="rafiki-chat-row rafiki-chat-user"
                 style="display:flex; justify-content:flex-end; width:100%;
                        margin-bottom:12px; padding:0 8px; box-sizing:border-box;">
              <div class="rafiki-bubble rafiki-bubble-user"
                style="border-radius:18px 18px 4px 18px; padding:10px 14px;
                font-size:0.9rem; max-width:72%; line-height:1.45; word-wrap:break-word;">
                {body}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="rafiki-chat-row rafiki-chat-assistant"
                 style="display:flex; align-items:flex-start; gap:10px; width:100%;
                        margin-bottom:12px; padding:0 8px; box-sizing:border-box;">
              <div class="rafiki-bubble-avatar"
                style="width:32px; height:32px; border-radius:50%;
                display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                {material_icon_html(ASSISTANT_AVATAR_ICON, size_px=20)}
              </div>
              <div class="rafiki-bubble rafiki-bubble-assistant"
                style="border-radius:4px 18px 18px 18px; padding:10px 14px;
                font-size:0.9rem; max-width:72%; line-height:1.45; word-wrap:break-word;">
                {body}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _theme_palette(is_dark: bool) -> dict:
    accent_rgb = "107, 147, 173" if is_dark else ACCENT_RGB
    if is_dark:
        return {
            "accent": ACCENT_LIGHT,
            "app_bg": "#0E1117",
            "sidebar_bg": "#262730",
            "text": "#FAFAFA",
            "muted": "0.65",
            "widget_bg": "#262730",
            "widget_border": "#4A4A4A",
            "user_bg": ACCENT,
            "user_text": "#FFFFFF",
            "assist_bg": f"rgba({accent_rgb}, 0.14)",
            "avatar_bg": f"rgba({accent_rgb}, 0.22)",
            "assist_text": "#FAFAFA",
            "color_scheme": "dark",
            "chat_pill_bg": "#262730",
            "chat_text": "#FAFAFA",
            "chat_btn_bg": DARK_CHAT_BTN,
            "chat_placeholder": "rgba(250, 250, 250, 0.55)",
        }
    return {
        "accent": ACCENT,
        "app_bg": "#FFFFFF",
        "sidebar_bg": "#F5F7F9",
        "text": "#31333F",
        "muted": "0.6",
        "widget_bg": "#FFFFFF",
        "widget_border": "#D0D7DE",
        "user_bg": ACCENT,
        "user_text": "#FFFFFF",
        "assist_bg": f"rgba({accent_rgb}, 0.08)",
        "avatar_bg": f"rgba({accent_rgb}, 0.15)",
        "assist_text": "inherit",
        "color_scheme": "light",
        "chat_pill_bg": LIGHT_CHAT_FILL,
        "chat_text": ACCENT_DARK,
        "chat_btn_bg": LIGHT_CHAT_BTN,
        "chat_placeholder": ACCENT,
    }


def _theme_rules(p: dict) -> str:
    return f"""
    :root {{
        --rafiki-accent: {p["accent"]};
        --rafiki-user-bg: {p["user_bg"]};
        --rafiki-user-text: {p["user_text"]};
        --rafiki-assist-bg: {p["assist_bg"]};
        --rafiki-assist-text: {p["assist_text"]};
        --rafiki-avatar-bg: {p["avatar_bg"]};
    }}
    html, body, #root {{
        background-color: {p["app_bg"]} !important;
    }}
    .stApp {{
        --primary-color: {p["accent"]} !important;
        --background-color: {p["app_bg"]} !important;
        --secondary-background-color: {p["sidebar_bg"]} !important;
        background-color: {p["app_bg"]} !important;
        color: {p["text"]} !important;
        color-scheme: {p["color_scheme"]};
    }}
    section[data-testid="stAppViewContainer"] {{
        background-color: {p["app_bg"]} !important;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {p["sidebar_bg"]} !important;
    }}
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stExpandSidebarButton"],
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] *,
    [data-testid="collapsedControl"] button,
    [data-testid="collapsedControl"] span {{
        color: {p["accent"]} !important;
        -webkit-text-fill-color: {p["accent"]} !important;
        fill: {p["accent"]} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption {{
        color: {p["text"]} !important;
    }}
    a, a:visited {{ color: {p["accent"]} !important; }}
    .stButton > button[kind="primary"] {{
        background-color: {p["accent"]} !important;
        border-color: {p["accent"]} !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {ACCENT_DARK} !important;
        border-color: {ACCENT_DARK} !important;
    }}
    .stButton > button[kind="secondary"] {{
        background-color: {p["widget_bg"]} !important;
        color: {p["text"]} !important;
        border-color: {p["widget_border"]} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        border-color: {p["accent"]} !important;
        color: {p["accent"]} !important;
    }}
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{
        border-color: {p["accent"]} !important;
        background-color: transparent !important;
    }}
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child {{
        background-color: {p["accent"]} !important;
        border-color: {p["accent"]} !important;
    }}
    div[data-testid="stRadio"] label[data-baseweb="radio"] {{
        background-color: transparent !important;
    }}
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:not(:first-child),
    div[data-testid="stRadio"] label[data-baseweb="radio"] input:checked + div {{
        background-color: transparent !important;
        color: {p["text"]} !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] > div:not(:first-child) {{
        background: transparent !important;
        background-color: transparent !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {p["widget_bg"]} !important;
        color: {p["text"]} !important;
        border-color: {p["widget_border"]} !important;
    }}
    div[data-baseweb="select"] > div:focus-within {{
        border-color: {p["accent"]} !important;
        box-shadow: 0 0 0 1px {p["accent"]} !important;
    }}
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] > div,
    [data-testid="stChatInput"] {{
        background: {p["app_bg"]} !important;
        background-color: {p["app_bg"]} !important;
        border: none !important;
        box-shadow: none !important;
    }}
    [data-testid="stChatInput"] > div {{
        background: {p["chat_pill_bg"]} !important;
        background-color: {p["chat_pill_bg"]} !important;
        border: 1px solid {p["widget_border"]} !important;
        border-radius: 10px !important;
        padding: 0.35rem 0.45rem 0.35rem 0.85rem !important;
        align-items: center !important;
        box-shadow: none !important;
    }}
    [data-testid="stChatInput"]:focus-within > div {{
        border-color: {p["accent"]} !important;
        box-shadow: 0 0 0 1px {p["accent"]} !important;
    }}
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [data-baseweb="input"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"] {{
        background: transparent !important;
        background-color: transparent !important;
        color: {p["chat_text"]} !important;
        -webkit-text-fill-color: {p["chat_text"]} !important;
        caret-color: {p["accent"]} !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInputTextArea"]::placeholder {{
        color: {p["chat_placeholder"]} !important;
        -webkit-text-fill-color: {p["chat_placeholder"]} !important;
        opacity: 1 !important;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    [data-testid="stChatInputSubmitButton"] {{
        background: {p["chat_btn_bg"]} !important;
        background-color: {p["chat_btn_bg"]} !important;
        border: 1px solid {p["widget_border"]} !important;
        border-radius: 8px !important;
        color: {p["accent"]} !important;
        min-width: 2.25rem !important;
        min-height: 2.25rem !important;
    }}
    [data-testid="stChatInputSubmitButton"]:hover {{
        background-color: rgba({ACCENT_RGB if p["color_scheme"] == "light" else "107, 147, 173"}, 0.14) !important;
        border-color: {p["accent"]} !important;
        color: {p["accent"]} !important;
    }}
    [data-testid="stChatInputSubmitButton"] svg {{
        fill: {p["accent"]} !important;
        stroke: none !important;
    }}
    .stSpinner > div {{
        border-top-color: {p["accent"]} !important;
    }}
    .rafiki-bubble-user {{
        background: var(--rafiki-user-bg) !important;
        color: var(--rafiki-user-text) !important;
    }}
    .rafiki-bubble-assistant {{
        background: var(--rafiki-assist-bg) !important;
        color: var(--rafiki-assist-text) !important;
    }}
    .rafiki-bubble-avatar {{
        background: var(--rafiki-avatar-bg) !important;
    }}
    .rafiki-header-icon {{
        color: var(--rafiki-accent) !important;
    }}
    .rafiki-header-sub {{
        opacity: {p["muted"]};
    }}
    """


def build_app_css(theme_mode: str) -> str:
    """Theme-aware CSS for Streamlit widgets and custom RAFIKI elements."""
    shared = """
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
    header[data-testid="stHeader"] {
        background: transparent !important;
        border: none !important;
    }
    #MainMenu { visibility: hidden !important; }
    footer { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { visibility: hidden !important; }
    .stAppDeployButton { visibility: hidden !important; }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        z-index: 999999 !important;
    }
    .block-container { padding-top: 1rem !important; }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 384px !important;
    }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem !important; }
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
        height: auto !important;
        min-height: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] h4 { font-size: 1.3rem !important; }
    .rafiki-sidebar-tagline {
      font-size: 0.82rem;
      line-height: 1.4;
      opacity: 0.65;
      margin: 0.15rem 0 1.1rem 0;
    }
    .rafiki-sidebar-block-heading {
      margin: 0 0 0.45rem 0;
    }
    .rafiki-sidebar-block-title {
      font-size: 0.9rem;
      font-weight: 700;
      margin: 0;
      line-height: 1.3;
    }
    .rafiki-sidebar-block-desc {
      font-size: 0.78rem;
      opacity: 0.6;
      margin: 0.15rem 0 0 0;
      line-height: 1.35;
    }
    .rafiki-sidebar-section-gap {
      margin-bottom: 1.15rem;
    }
    .rafiki-sidebar-profile-card {
      background: rgba(85, 122, 147, 0.12);
      border-radius: 10px;
      padding: 1.05rem 1.1rem;
      margin: 0 0 1.25rem 0;
    }
    .rafiki-sidebar-profile-top {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .rafiki-sidebar-profile-avatar {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: rgba(85, 122, 147, 0.18);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .rafiki-sidebar-profile-tier {
      font-size: 0.92rem;
      font-weight: 600;
      line-height: 1.25;
    }
    .rafiki-sidebar-profile-lang {
      font-size: 0.78rem;
      opacity: 0.72;
      line-height: 1.25;
      margin-top: 0.1rem;
    }
    .rafiki-sidebar-profile-desc {
      font-size: 0.8rem;
      opacity: 0.75;
      line-height: 1.45;
      margin: 0.9rem 0 0 0;
    }
    section[data-testid="stSidebar"] [data-testid="stButtonGroup"] {
      width: 100%;
    }
    section[data-testid="stSidebar"] [data-testid="stButtonGroup"] > div {
      width: 100%;
      background: transparent !important;
      gap: 0.45rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButtonGroup"] button {
      border-radius: 999px !important;
      border: 1px solid rgba(128, 128, 128, 0.28) !important;
      color: inherit !important;
      opacity: 1;
      font-size: 0.82rem !important;
      font-weight: 500 !important;
      padding: 0.38rem 0.65rem !important;
      min-height: 2rem !important;
      box-shadow: none !important;
    }
    """
    light_rules = _theme_rules(_theme_palette(False))
    dark_rules = _theme_rules(_theme_palette(True))

    if theme_mode == "light":
        themed = light_rules
    elif theme_mode == "dark":
        themed = dark_rules
    else:
        themed = f"""
        @media (prefers-color-scheme: light) {{
            {light_rules}
        }}
        @media (prefers-color-scheme: dark) {{
            {dark_rules}
        }}
        """
    if theme_mode == "light":
        pill_block = _pill_css(is_dark=False)
        tile_block = _theme_tile_css(is_dark=False)
    elif theme_mode == "dark":
        pill_block = _pill_css(is_dark=True)
        tile_block = _theme_tile_css(is_dark=True)
    else:
        pill_block = (
            f"@media (prefers-color-scheme: light) {{ {_pill_css(is_dark=False)} }}"
            f"@media (prefers-color-scheme: dark) {{ {_pill_css(is_dark=True)} }}"
        )
        tile_block = (
            f"@media (prefers-color-scheme: light) {{ {_theme_tile_css(is_dark=False)} }}"
            f"@media (prefers-color-scheme: dark) {{ {_theme_tile_css(is_dark=True)} }}"
        )

    return shared + themed + _light_theme_overrides() + pill_block + tile_block


def _pill_css(is_dark: bool) -> str:
    """Segmented-control (button group) pill colors, forced per chosen theme.

    Streamlit's native segmented-control colors follow the OS color scheme
    (because both [theme] and [theme.dark] are defined in config.toml), so the
    buttons bleed system colors regardless of the in-app theme choice.  These
    rules pin the colors to the app's selected theme.  The caller gates which
    block is emitted, so no [data-theme] selector is needed.

    Selectors target both the kind attribute and aria state for robustness,
    and avoid depending on the widget container's test id.
    """
    base = 'section[data-testid="stSidebar"] button'
    unselected = f'{base}[kind="segmented_control"]'
    selected = (
        f'{base}[kind="segmented_controlActive"], '
        f'{base}[aria-checked="true"], '
        f'{base}[aria-pressed="true"]'
    )

    if is_dark:
        unsel_bg = "#262730"
        unsel_text = "#FAFAFA"
        unsel_border = "#4A4A4A"
        sel_bg = ACCENT
        sel_text = "#F7FBFF"
        sel_border = ACCENT
        hover_bg = "#363A45"
    else:
        unsel_bg = LIGHT_CHAT_FILL
        unsel_text = ACCENT
        unsel_border = LIGHT_BORDER
        sel_bg = ACCENT
        sel_text = "#F7FBFF"
        sel_border = ACCENT
        hover_bg = LIGHT_CHAT_BTN

    return f"""
    {unselected} {{
        background: {unsel_bg} !important;
        background-color: {unsel_bg} !important;
        border-color: {unsel_border} !important;
        color: {unsel_text} !important;
        opacity: 1 !important;
    }}
    {unselected} * {{
        color: {unsel_text} !important;
    }}
    {unselected}:hover {{
        background: {hover_bg} !important;
        background-color: {hover_bg} !important;
        border-color: {ACCENT} !important;
    }}
    {selected} {{
        background: {sel_bg} !important;
        background-color: {sel_bg} !important;
        border-color: {sel_border} !important;
        color: {sel_text} !important;
        -webkit-text-fill-color: {sel_text} !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }}
    {base}[kind="segmented_controlActive"] *,
    {base}[aria-checked="true"] *,
    {base}[aria-pressed="true"] * {{
        color: {sel_text} !important;
        -webkit-text-fill-color: {sel_text} !important;
    }}
    """


def _theme_tile_css(is_dark: bool) -> str:
    """Theme picker tiles — icon stacked above label inside real st.button."""
    btn = (
        'section[data-testid="stSidebar"] '
        '[data-testid="column"] .stButton > button:has([data-testid="stIconMaterial"])'
    )
    icon = f'{btn} [data-testid="stIconMaterial"]'
    label = f'{btn} p'
    label_box = f'{btn} [data-testid="stMarkdownContainer"]'

    if is_dark:
        unsel_bg = "#262730"
        unsel_text = ACCENT_LIGHT
        unsel_border = "#4A4A4A"
        sel_bg = ACCENT
        sel_text = "#F7FBFF"
        sel_border = ACCENT
        hover_bg = "#363A45"
    else:
        unsel_bg = LIGHT_CHAT_FILL
        unsel_text = ACCENT
        unsel_border = LIGHT_BORDER
        sel_bg = ACCENT
        sel_text = "#F7FBFF"
        sel_border = ACCENT
        hover_bg = LIGHT_CHAT_BTN

    return f"""
    {btn} {{
        min-height: 3.6rem !important;
        height: auto !important;
        border-radius: 0.7rem !important;
        padding: 0.45rem 0.15rem !important;
        box-shadow: none !important;
    }}
    {btn} > div {{
        width: 100% !important;
        min-width: 0 !important;
    }}
    {btn} > div > span {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.2rem !important;
        width: 100% !important;
        min-width: 0 !important;
        white-space: nowrap !important;
        overflow-wrap: normal !important;
        word-break: keep-all !important;
        font-size: 0.68rem !important;
    }}
    {icon} {{
        font-size: 1.35rem !important;
        line-height: 1 !important;
        margin: 0 !important;
        flex-shrink: 0 !important;
    }}
    {label_box} {{
        width: auto !important;
        max-width: none !important;
        white-space: nowrap !important;
        overflow-wrap: normal !important;
        word-break: keep-all !important;
        overflow: visible !important;
    }}
    {label} {{
        margin: 0 !important;
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        line-height: 1.1 !important;
        text-align: center !important;
        white-space: nowrap !important;
        overflow-wrap: normal !important;
        word-break: keep-all !important;
        overflow: visible !important;
        max-width: none !important;
    }}
    {btn}[kind="secondary"] {{
        background: {unsel_bg} !important;
        background-color: {unsel_bg} !important;
        border: 1px solid {unsel_border} !important;
        color: {unsel_text} !important;
    }}
    {btn}[kind="secondary"] * {{
        color: {unsel_text} !important;
        -webkit-text-fill-color: {unsel_text} !important;
    }}
    {btn}[kind="secondary"]:hover {{
        background: {hover_bg} !important;
        background-color: {hover_bg} !important;
        border-color: {ACCENT} !important;
    }}
    {btn}[kind="primary"] {{
        background: {sel_bg} !important;
        background-color: {sel_bg} !important;
        border: 1px solid {sel_border} !important;
        color: {sel_text} !important;
    }}
    {btn}[kind="primary"] * {{
        color: {sel_text} !important;
        -webkit-text-fill-color: {sel_text} !important;
    }}
    {btn}[kind="primary"]:hover {{
        background: {ACCENT_DARK} !important;
        background-color: {ACCENT_DARK} !important;
        border-color: {ACCENT_DARK} !important;
    }}
    """


def _light_theme_overrides() -> str:
    """Light-mode polish — scoped to [data-theme="light"] only; dark mode untouched."""
    return f"""
    [data-theme="light"] .stApp {{
        --text-color: {LIGHT_TEXT} !important;
        --background-color: {LIGHT_MAIN_BG} !important;
        --secondary-background-color: {LIGHT_SIDEBAR_BG} !important;
        background-color: {LIGHT_MAIN_BG} !important;
        color: {LIGHT_TEXT} !important;
    }}
    [data-theme="light"] section[data-testid="stAppViewContainer"] {{
        background-color: {LIGHT_MAIN_BG} !important;
    }}
    [data-theme="light"] section[data-testid="stMain"],
    [data-theme="light"] section[data-testid="stMain"] > div,
    [data-theme="light"] section[data-testid="stMain"] .block-container {{
        background-color: {LIGHT_MAIN_BG} !important;
        color: {LIGHT_TEXT} !important;
    }}
    [data-theme="light"] section[data-testid="stSidebar"] {{
        background-color: {LIGHT_SIDEBAR_BG} !important;
        border-right: 1px solid {LIGHT_BORDER} !important;
        color: {LIGHT_TEXT} !important;
    }}
    [data-theme="light"] section[data-testid="stSidebar"] h4,
    [data-theme="light"] section[data-testid="stSidebar"] h4 * {{
        color: {LIGHT_TEXT} !important;
    }}
    [data-theme="light"] section[data-testid="stSidebar"] label,
    [data-theme="light"] section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-theme="light"] section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-theme="light"] section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {{
        color: {LIGHT_TEXT} !important;
    }}
    [data-theme="light"] section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-theme="light"] section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    [data-theme="light"] section[data-testid="stSidebar"] .stCaption,
    [data-theme="light"] section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-theme="light"] section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    [data-theme="light"] section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    [data-theme="light"] section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
    [data-theme="light"] section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] em {{
        color: {LIGHT_TEXT_SOFT} !important;
        opacity: 1 !important;
    }}
    [data-theme="light"] section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-theme="light"] section[data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-theme="light"] section[data-testid="stMain"] [data-testid="stCaptionContainer"] * {{
        color: {LIGHT_TEXT_SOFT} !important;
        opacity: 1 !important;
    }}
    [data-theme="light"] .rafiki-header-sub {{
        color: {LIGHT_TEXT_SOFT} !important;
        opacity: 1 !important;
    }}
    [data-theme="light"] .rafiki-empty-hero p:first-of-type {{
        color: {LIGHT_TEXT} !important;
        opacity: 1 !important;
    }}
    [data-theme="light"] .rafiki-empty-hero p:last-of-type,
    [data-theme="light"] .rafiki-empty-disclaimer {{
        color: {LIGHT_TEXT_SOFT} !important;
        opacity: 1 !important;
    }}
    [data-theme="light"] .rafiki-sidebar-tagline,
    [data-theme="light"] .rafiki-sidebar-block-desc,
    [data-theme="light"] .rafiki-sidebar-profile-lang,
    [data-theme="light"] .rafiki-sidebar-profile-desc {{
        color: {LIGHT_TEXT_SOFT} !important;
        opacity: 1 !important;
    }}
    [data-theme="light"] .rafiki-sidebar-block-title,
    [data-theme="light"] .rafiki-sidebar-profile-tier {{
        color: {LIGHT_TEXT} !important;
    }}
    [data-theme="light"] .rafiki-sidebar-profile-card {{
        background: rgba({ACCENT_RGB}, 0.12) !important;
    }}
    [data-theme="light"] .rafiki-sidebar-profile-avatar {{
        background: rgba({ACCENT_RGB}, 0.18) !important;
    }}
    [data-theme="light"] div[data-baseweb="select"] > div {{
        background-color: {LIGHT_SURFACE} !important;
        color: {LIGHT_TEXT} !important;
        border-color: {LIGHT_BORDER} !important;
    }}
    [data-theme="light"] section[data-testid="stSidebar"] .stButton > button,
    [data-theme="light"] section[data-testid="stMain"] .stButton > button {{
        background-color: {LIGHT_SURFACE} !important;
        color: {LIGHT_TEXT} !important;
        border: 1px solid {LIGHT_BORDER} !important;
    }}
    [data-theme="light"] section[data-testid="stSidebar"] .stButton > button:hover,
    [data-theme="light"] section[data-testid="stMain"] .stButton > button:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
        background-color: rgba({ACCENT_RGB}, 0.08) !important;
    }}
    """


def sync_theme_query_param() -> None:
    st.query_params["theme"] = st.session_state.theme_mode


def inject_theme_attribute(theme_mode: str) -> None:
    """Set data-theme on .stApp so [data-theme="light"] CSS overrides apply."""
    components.html(
        f"""
        <script>
        (function () {{
            const docs = [];
            const seen = new Set();
            [window.top, window.parent, window].forEach((w) => {{
                try {{
                    const d = w.document;
                    if (d && !seen.has(d)) {{
                        seen.add(d);
                        docs.push(d);
                    }}
                }} catch (e) {{}}
            }});
            const pref = {theme_mode!r};
            function applyTheme() {{
                docs.forEach((doc) => {{
                    const app = doc.querySelector(".stApp");
                    if (!app) return;
                    if (pref === "light") {{
                        app.setAttribute("data-theme", "light");
                    }} else if (pref === "dark") {{
                        app.setAttribute("data-theme", "dark");
                    }} else {{
                        const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
                        app.setAttribute("data-theme", dark ? "dark" : "light");
                    }}
                }});
            }}
            applyTheme();
            if (pref === "system") {{
                window
                    .matchMedia("(prefers-color-scheme: dark)")
                    .addEventListener("change", applyTheme);
            }}
        }})();
        </script>
        """,
        height=0,
    )


def inject_ui_animations() -> None:
    """Inject keyframe CSS into the parent document (Streamlit strips some st.markdown styles)."""
    components.html(
        """
        <script>
        (function () {
            const doc = parent.document;
            if (doc.getElementById("rafiki-ui-animations")) return;
            const style = doc.createElement("style");
            style.id = "rafiki-ui-animations";
            style.textContent = `
                @keyframes rafikiEmptyFadeOut {
                    from { opacity: 1; transform: translateY(0); }
                    to { opacity: 0; transform: translateY(-8px); }
                }
                @keyframes rafikiChatFadeIn {
                    from { opacity: 0; transform: translateY(6px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .rafiki-fade-out {
                    animation: rafikiEmptyFadeOut 0.22s ease forwards !important;
                    pointer-events: none !important;
                    will-change: opacity, transform;
                }
            `;
            doc.head.appendChild(style);
        })();
        </script>
        """,
        height=0,
    )


def apply_chat_fade_in() -> None:
    """Fade in only the newest chat bubble (first entry after empty state)."""
    st.markdown(
        """
        <style>
        section[data-testid="stMain"] .rafiki-chat-row:last-of-type {
            animation: rafikiChatFadeIn 0.25s ease forwards !important;
            will-change: opacity, transform;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _complete_empty_fade() -> None:
    query = st.session_state.pop("_queued_query", None)
    st.session_state.fading_empty = False
    st.session_state._fade_armed = False
    if query:
        st.session_state.display_messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.awaiting_answer = query
        if not st.session_state.get("chat_has_started"):
            st.session_state.chat_enter = True
            st.session_state.chat_has_started = True


@st.fragment(run_every=timedelta(seconds=EMPTY_FADE_SECONDS))
def _fade_exit_timer() -> None:
    """Arm on first tick, complete fade and show chat on second tick (~EMPTY_FADE_SECONDS later)."""
    if not st.session_state.get("fading_empty"):
        return
    if not st.session_state.get("_fade_armed"):
        st.session_state._fade_armed = True
        return
    _complete_empty_fade()
    st.rerun()

# ---------------------------------------------------------------------------
# 2. LOAD DATA + BUILD INDEXES (cached so this runs once, not per interaction)
# ---------------------------------------------------------------------------

def _fix_mojibake(text: str) -> str:
    """Fix common UTF-8-as-Latin1 artifacts in the CSV."""
    if not isinstance(text, str):
        return text
    return (
        text.replace("â€”", "—")
        .replace("â€“", "–")
        .replace("â€™", "'")
        .replace("â€œ", '"')
        .replace("â€\u009d", '"')
    )


@st.cache_resource(show_spinner="Loading RAFIKI knowledge base...")
def load_indexes():
    df = pd.read_excel(DATA_PATH) if DATA_PATH.endswith((".xlsx", ".xls")) else pd.read_csv(DATA_PATH)

    required_cols = {"pair_id", "user_tier", "language", "topic", "source",
                      "keywords", "question", "answer", "answer_context"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for col in ("question", "answer", "answer_context", "keywords"):
        df[col] = df[col].astype(str).map(_fix_mojibake)

    df["embed_text"] = (
        df["question"] + " "
        + df["keywords"] + " "
        + df["answer"] + " "
        + df["answer_context"].str.slice(0, 400)
    )

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    indexes = {}
    for (tier, lang), group in df.groupby(["user_tier", "language"]):
        group = group.reset_index(drop=True)
        embeddings = embed_model.encode(group["embed_text"].tolist(), show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        idx = faiss.IndexFlatL2(embeddings.shape[1])
        idx.add(embeddings)
        indexes[(tier, lang)] = (idx, group)

    return embed_model, indexes


@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)


embed_model, indexes = load_indexes()
client = get_groq_client()


# ---------------------------------------------------------------------------
# 3. RETRIEVAL
# ---------------------------------------------------------------------------

def _search_index(query: str, tier: str, language: str, k: int):
    key = (tier, language)
    if key not in indexes:
        return []

    idx, group = indexes[key]
    q_emb = np.array(embed_model.encode([query])).astype("float32")
    distances, indices = idx.search(q_emb, min(k, len(group)))

    results = []
    for dist, i in zip(distances[0], indices[0]):
        if i == -1:
            continue
        row = group.iloc[i]
        results.append({
            "distance": float(dist),
            "question": row["question"],
            "answer_context": row["answer_context"],
            "topic": row["topic"],
            "source": row["source"],
        })
    return results


def retrieve(query: str, tier: str, language: str, k: int = TOP_K):
    results = _search_index(query, tier, language, k)
    if results and results[0]["distance"] <= NO_MATCH_THRESHOLD:
        return results

    other_lang = "Swahili" if language == "English" else "English"
    fallback = _search_index(query, tier, other_lang, k)
    if fallback and (not results or fallback[0]["distance"] < results[0]["distance"]):
        return fallback
    return results


# ---------------------------------------------------------------------------
# 4. PROMPT CONSTRUCTION + GENERATION
# ---------------------------------------------------------------------------

def build_system_prompt(tier: str, language: str) -> str:
    persona = TIER_PERSONA.get(tier, "")
    lang_rule = (
        "Respond ONLY in English, regardless of the context language."
        if language == "English"
        else "Respond ONLY in Swahili (Kiswahili), regardless of the context language."
    )
    return (
        "You are Rafiki, a bilingual financial FAQ assistant for the Tanzanian market.\n"
        f"User tier: {tier}. {persona}\n"
        f"{lang_rule}\n"
        "Answer ONLY using the provided context. You may add directly related "
        "supporting detail, but never introduce unrelated information.\n"
        "If the context does not cover the question, say you do not have verified information.\n"
        "Always mention the source institution naturally if it is relevant "
        "(e.g. 'According to CRDB...' / 'Kulingana na CRDB...').\n"
        "If the user message is unclear, politely ask them to clarify.\n"
        "Keep answers clear, accurate, and appropriately concise for the user's tier."
    )


def build_user_prompt(query: str, contexts: list) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']} | Topic: {c['topic']}]\n{c['answer_context']}"
        for c in contexts
    )
    return f"Context:\n{context_block}\n\nQuestion:\n{query}"


def generate_answer(query: str, tier: str, language: str, history: list):
    contexts = retrieve(query, tier, language, k=TOP_K)
    passed = bool(contexts and contexts[0]["distance"] <= NO_MATCH_THRESHOLD)

    if not passed:
        return NO_MATCH_MESSAGE[language]

    system_prompt = build_system_prompt(tier, language)
    user_prompt = build_user_prompt(query, contexts)

    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]
    for turn in history[-HISTORY_TURNS:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content or NO_MATCH_MESSAGE[language]
    except RateLimitError:
        return (
            "The service is busy right now. Please wait a moment and try again."
            if language == "English"
            else "Huduma ina shughuli nyingi sasa hivi. Tafadhali subiri kidogo na ujaribu tena."
        )
    except APIError:
        return (
            "Sorry, I couldn't reach the language model right now. Please try again shortly."
            if language == "English"
            else "Samahani, sikuweza kuunganisha na mfumo wa lugha sasa hivi. Tafadhali jaribu tena baadaye."
        )


# ---------------------------------------------------------------------------
# 5. STREAMLIT UI
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "last_tier" not in st.session_state:
    st.session_state.last_tier = TIERS[0]
if "last_language" not in st.session_state:
    st.session_state.last_language = LANGUAGES[0]
if st.session_state.get("language") not in LANGUAGES:
    st.session_state.language = LANGUAGES[0]
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "fading_empty" not in st.session_state:
    st.session_state.fading_empty = False
if "chat_has_started" not in st.session_state:
    st.session_state.chat_has_started = False
if "theme_mode" not in st.session_state:
    qp_theme = st.query_params.get("theme", "system")
    st.session_state.theme_mode = qp_theme if qp_theme in VALID_THEMES else "system"
else:
    qp_theme = st.query_params.get("theme")
    if qp_theme in VALID_THEMES and qp_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = qp_theme

st.markdown(
    f"<style>{build_app_css(st.session_state.theme_mode)}</style>",
    unsafe_allow_html=True,
)

inject_theme_attribute(st.session_state.theme_mode)

if not st.session_state.get("_ui_animations_injected"):
    inject_ui_animations()
    st.session_state._ui_animations_injected = True

# --- Sidebar ---
SIDEBAR_SECTIONS = {
    "English": {
        "profile_title": "Profile",
        "profile_desc": "Scoped to your profile.",
        "language_title": "Language",
        "language_desc": "RAFIKI replies in this language.",
        "theme_title": "Theme",
        "theme_desc": "Website appearance.",
    },
    "Swahili": {
        "profile_title": "Wasifu",
        "profile_desc": "Majibu yanategemea wasifu wako.",
        "language_title": "Lugha",
        "language_desc": "RAFIKI hujibu kwa lugha hii.",
        "theme_title": "Mandhari",
        "theme_desc": "Muonekano wa tovuti.",
    },
}

with st.sidebar:
    st.markdown(f"#### {ICON_BRAND} RAFIKI")
    st.markdown(
        '<p class="rafiki-sidebar-tagline">'
        "Bilingual Financial FAQ Assistant for Tanzania"
        "</p>",
        unsafe_allow_html=True,
    )

    if st.button("New chat", icon=":material/add:", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.display_messages = []
        st.session_state.pending_query = None
        st.session_state.fading_empty = False
        st.session_state._queued_query = None
        st.session_state._fade_armed = False
        st.session_state.awaiting_answer = None
        st.session_state.chat_has_started = False
        st.session_state.pop("chat_enter", None)
        st.rerun()

    _sidebar_lang = _session_choice("language", tuple(LANGUAGES), LANGUAGES[0])
    _sections = SIDEBAR_SECTIONS[_sidebar_lang]

    sidebar_section_heading(_sections["profile_title"], _sections["profile_desc"])
    tier = st.selectbox("I am a... / Mimi ni...", TIERS, key="tier", label_visibility="collapsed")
    profile_card_slot = st.empty()
    st.markdown('<div class="rafiki-sidebar-section-gap"></div>', unsafe_allow_html=True)

    sidebar_section_heading(_sections["language_title"], _sections["language_desc"])
    language = st.segmented_control(
        "Language / Lugha",
        options=LANGUAGES,
        default=LANGUAGES[0],
        key="language",
        label_visibility="collapsed",
        width="stretch",
    )
    language = _session_choice("language", tuple(LANGUAGES), LANGUAGES[0])
    st.markdown('<div class="rafiki-sidebar-section-gap"></div>', unsafe_allow_html=True)

    if tier != st.session_state.last_tier or language != st.session_state.last_language:
        st.session_state.messages = []
        st.session_state.display_messages = []
        st.session_state.pending_query = None
        st.session_state.fading_empty = False
        st.session_state._queued_query = None
        st.session_state._fade_armed = False
        st.session_state.awaiting_answer = None
        st.session_state.chat_has_started = False
        st.session_state.pop("chat_enter", None)
        st.session_state.last_tier = tier
        st.session_state.last_language = language
        st.rerun()

    _sections = SIDEBAR_SECTIONS[language]
    sidebar_section_heading(_sections["theme_title"], _sections["theme_desc"])
    render_theme_picker(language)
    st.markdown('<div class="rafiki-sidebar-section-gap"></div>', unsafe_allow_html=True)

    with profile_card_slot.container():
        render_sidebar_profile_card(tier, language)

# --- Main chat panel ---
lang_code = "EN" if language == "English" else "SW"
st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 1rem;">
        <div style="font-size: 2rem; font-weight: 700; line-height: 1.2; margin-bottom: 0.35rem;">
            <span class="rafiki-header-icon" style="font-family: 'Material Symbols Outlined';
                         font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
                         font-size: 2rem; vertical-align: middle;">forum</span>
            RAFIKI
        </div>
        <div class="rafiki-header-sub" style="font-size: 0.9rem; font-weight: 400;">
            Talking as {tier} · {lang_code}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if (
    st.session_state.pending_query
    and not st.session_state.fading_empty
    and not st.session_state.display_messages
):
    st.session_state.fading_empty = True
    st.session_state._queued_query = st.session_state.pending_query
    st.session_state.pending_query = None
    st.session_state._fade_armed = False

_do_chat_fade_in = st.session_state.pop("chat_enter", False)

for msg in st.session_state.display_messages:
    render_chat_bubble(msg["role"], msg["content"])

if _do_chat_fade_in and st.session_state.display_messages:
    apply_chat_fade_in()

disclaimer = {
    "English": (
        "General guidance only — not official financial, tax, or legal advice. "
        "Verify with your bank, TRA, NSSF, NHIF, or a licensed advisor."
    ),
    "Swahili": (
        "Ushauri wa jumla tu — si ushauri rasmi wa kifedha, kodi, au kisheria. "
        "Thibitisha na benki yako, TRA, NSSF, NHIF, au mshauri aliyesajiliwa."
    ),
}

show_empty_state = (
    not st.session_state.display_messages
    and not st.session_state.pending_query
    and not st.session_state.get("awaiting_answer")
    and not st.session_state.get("fading_empty")
)
show_fading_empty = st.session_state.get("fading_empty")

if show_empty_state or show_fading_empty:
    fade_class = " rafiki-fade-out" if show_fading_empty else ""
    hero_title = (
        "What would you like to know?"
        if language == "English"
        else "Ungependa kujua nini?"
    )
    hero_subtitle = (
        "Ask a financial question for your profile, or try one of these:"
        if language == "English"
        else "Uliza swali la kifedha kwa wasifu wako, au jaribu moja ya hizi:"
    )
    st.markdown(
        f"""
        <div class="rafiki-empty-hero{fade_class}"
             style="text-align:center; padding: 3rem 1rem 1.5rem;">
          <p style="font-size:2rem; font-weight:700; margin-bottom:0.5rem; opacity:0.85;">
            {hero_title}
          </p>
          <p style="font-size:0.9rem; opacity:0.45; margin-bottom:1.5rem;">
            {hero_subtitle}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if show_empty_state:
        starters = STARTER_QUESTIONS.get((tier, language), [])
        if starters:
            chip_cols = st.columns([1, 2, 2, 2, 1])
            for i, question in enumerate(starters):
                with chip_cols[i + 1]:
                    if st.button(
                        format_display(question),
                        key=f"starter_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_query = question
                        st.rerun()

    st.markdown(
        f'<p class="rafiki-empty-disclaimer{fade_class}" style="text-align: center; '
        f'font-size: 0.875rem; opacity: 0.6; margin-top: 1rem;">'
        f"{format_display(disclaimer[language])}</p>",
        unsafe_allow_html=True,
    )

    if show_fading_empty:
        _fade_exit_timer()

_chat_placeholder = (
    "Type your question here..."
    if language == "English"
    else "Andika swali lako hapa..."
)
user_query = st.chat_input(_chat_placeholder)

if user_query and not st.session_state.get("fading_empty"):
    last_msg = (
        st.session_state.display_messages[-1]
        if st.session_state.display_messages
        else None
    )
    is_new_turn = (
        not last_msg
        or last_msg["role"] != "user"
        or last_msg["content"] != user_query
    )
    if is_new_turn:
        if not st.session_state.display_messages:
            st.session_state.fading_empty = True
            st.session_state._queued_query = user_query
            st.session_state._fade_armed = False
            st.rerun()
        else:
            st.session_state.display_messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "user", "content": user_query})
            render_chat_bubble("user", user_query)
            st.session_state.awaiting_answer = user_query

if st.session_state.get("awaiting_answer"):
    user_query = st.session_state.awaiting_answer
    st.session_state.awaiting_answer = None

    spinner = "Thinking..." if language == "English" else "Inafikiri..."
    with st.spinner(spinner):
        answer = generate_answer(user_query, tier, language, st.session_state.messages)
    render_chat_bubble("assistant", answer)

    st.session_state.display_messages.append({"role": "assistant", "content": answer})
    st.session_state.messages.append({"role": "assistant", "content": answer})

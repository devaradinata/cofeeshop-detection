"""Complete CSS Design System and HTML cleaner matching 100% Google Stitch specifications."""
import re
import streamlit as st


def clean_html_str(html_str: str) -> str:
    """Sanitize HTML string to prevent any markdown parser from converting it to code blocks.
    
    1. Removes HTML comments <!-- ... -->
    2. Strips leading and trailing whitespace from each line (eliminating 4-space markdown indents).
    3. Joins all non-empty lines into a single continuous stream.
    """
    no_comments = re.sub(r"<!--.*?-->", "", html_str, flags=re.DOTALL)
    lines = [line.strip() for line in no_comments.splitlines()]
    return "".join(line for line in lines if line)


def render_html(html_str: str):
    """Render HTML safely without any markdown code block interpretation."""
    clean = clean_html_str(html_str)
    st.markdown(clean, unsafe_allow_html=True)


STITCH_GLOBAL_CSS = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap">

<style>
  /* Global Resets */
  html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #FFFFFF !important;
    color: #171717 !important;
    font-family: 'Geist', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
  }

  /* Clean Header without hiding sidebar controls */
  [data-testid="stHeader"] {
    background-color: transparent !important;
    z-index: 1000 !important;
  }
  
  /* Hide ONLY right-side Streamlit hamburger menu and rainbow bar, NEVER the sidebar toggle */
  #MainMenu,
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"],
  footer {
    display: none !important;
  }

  /* Make sure sidebar expand button is ALWAYS visible, prominent, and clickable at top-left */
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"],
  button[data-testid="stSidebarCollapseButton"],
  button[aria-label*="Expand sidebar"],
  button[aria-label*="open sidebar" i] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 12px !important;
    left: 12px !important;
    z-index: 9999999 !important;
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
    border: 1px solid #1E293B !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    cursor: pointer !important;
    padding: 6px 10px !important;
  }

  [data-testid="stSidebarCollapsedControl"] svg,
  [data-testid="collapsedControl"] svg {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
  }

  /* Hide the close button inside the sidebar to prevent accidental closure (Fixed sidebar like Stitch) */
  section[data-testid="stSidebar"][aria-expanded="true"] button[data-testid="stSidebarCollapseButton"],
  section[data-testid="stSidebar"][aria-expanded="true"] button[aria-label*="Collapse sidebar"] {
    display: none !important;
  }

  .block-container {
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1560px !important;
    background-color: #FFFFFF !important;
  }

  /* Sidebar styling matching Stitch w-60 */
  [data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
    box-shadow: none !important;
    width: 250px !important;
    min-width: 250px !important;
  }

  [data-testid="stSidebar"] > div:first-child {
    background-color: #FFFFFF !important;
    padding: 16px 14px !important;
  }

  [data-testid="stSidebarContent"] {
    padding: 0 !important;
  }

  /* Clean up radio button bullets completely in sidebar for 100% Stitch look */
  [data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 3px !important;
  }

  [data-testid="stSidebar"] [data-testid="stRadio"] label {
    background-color: transparent !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    border: none !important;
  }

  /* Hide the native radio circle completely */
  [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
  }

  /* Label text formatting inside sidebar radio */
  [data-testid="stSidebar"] [data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #4B5563 !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    letter-spacing: -0.01em;
  }

  /* Active / Selected nav item */
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background-color: #F3F4F6 !important;
  }

  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
    color: #111827 !important;
    font-weight: 600 !important;
  }

  /* Hover state */
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover:not(:has(input:checked)) {
    background-color: #F9FAFB !important;
  }

  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover:not(:has(input:checked)) [data-testid="stMarkdownContainer"] p {
    color: #111827 !important;
  }

  /* Hide default Streamlit elements that clutter UI */
  footer {
    display: none !important;
  }
  #MainMenu {
    display: none !important;
  }

  /* Custom Streamlit Primary Button matching Stitch (Black background, white text) */
  button[kind="primary"], .stButton > button[kind="primary"] {
    background-color: #171717 !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    border: none !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.15s ease !important;
  }

  button[kind="primary"]:hover {
    background-color: #262626 !important;
  }

  /* Secondary Buttons */
  button[kind="secondary"], .stButton > button[kind="secondary"], .stButton > button {
    background-color: #FFFFFF !important;
    color: #374151 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
  }

  button[kind="secondary"]:hover, .stButton > button:hover {
    background-color: #F9FAFB !important;
    border-color: #D1D5DB !important;
  }

  /* Selectbox styling */
  div[data-baseweb="select"] > div {
    border-color: #E5E7EB !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    min-height: 32px !important;
    height: 32px !important;
  }

  /* Stitch Design System Utility Classes */
  .stitch-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }

  @media (max-width: 1024px) {
    .stitch-kpi-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  .stitch-kpi-card {
    background-color: #FFFFFF;
    border: 1px solid rgba(229, 231, 235, 0.85);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    min-height: 105px;
  }

  .stitch-kpi-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 13px;
    font-weight: 500;
    color: #6B7280;
  }

  .stitch-kpi-icon {
    font-size: 18px;
    color: #9CA3AF;
  }

  .stitch-kpi-body {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-top: 10px;
  }

  .stitch-kpi-val {
    font-size: 24px;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: #111827;
  }

  .stitch-kpi-sub {
    font-size: 12px;
    color: #9CA3AF;
  }

  .stitch-kpi-delta-up {
    font-size: 12px;
    font-weight: 500;
    color: #16A34A;
    display: inline-flex;
    align-items: center;
    gap: 2px;
  }

  .stitch-table-card {
    background-color: #FFFFFF;
    border: 1px solid rgba(229, 231, 235, 0.85);
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
  }

  .stitch-table-header {
    padding: 12px 16px;
    border-bottom: 1px solid #F3F4F6;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .stitch-table-row {
    padding: 10px 16px;
    border-bottom: 1px solid #F3F4F6;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    transition: background-color 0.1s ease;
    cursor: pointer;
  }

  .stitch-table-row:hover {
    background-color: #F9FAFB;
  }

  .stitch-table-row:last-child {
    border-bottom: none;
  }

  /* Seamless Icon Buttons for Video Header and Video Player */
  .stitch-table-header button,
  .stitch-player-bar button,
  [data-testid="stHorizontalBlock"] button:has(p:contains("⧉")),
  [data-testid="stHorizontalBlock"] button:has(p:contains("📷")),
  [data-testid="stHorizontalBlock"] button:has(p:contains("⛶")),
  [data-testid="stHorizontalBlock"] button:has(p:contains("▶")),
  [data-testid="stHorizontalBlock"] button:has(p:contains("⏸")),
  [data-testid="stHorizontalBlock"] button:has(p:contains("🔊")),
  [data-testid="stHorizontalBlock"] button:has(p:contains("🔇")),
  button[key^="hdr_ic_"],
  button[key^="vid_ctrl_"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    color: #6B7280 !important;
    font-size: 16px !important;
    min-height: 28px !important;
    height: 28px !important;
    width: 28px !important;
    cursor: pointer !important;
    border-radius: 4px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.15s ease !important;
  }

  .stitch-table-header button:hover,
  .stitch-player-bar button:hover,
  button[key^="hdr_ic_"]:hover,
  button[key^="vid_ctrl_"]:hover {
    color: #111827 !important;
    background: #F3F4F6 !important;
  }

  /* Compact progress bar height in video player */
  [data-testid="stHorizontalBlock"] [data-testid="stProgress"] > div {
    height: 6px !important;
    border-radius: 9999px !important;
  }
</style>
"""


def apply_custom_styles():
    """Inject global CSS into the app and force-open the sidebar if closed."""
    import streamlit.components.v1 as components
    render_html(STITCH_GLOBAL_CSS)
    
    # Real JavaScript execution in client browser to force open sidebar
    components.html("""
    <script>
    function openSidebarNow() {
      try {
        const doc = window.parent.document;
        if (!doc) return;
        
        // Clear Streamlit local storage collapsed memory
        try {
          window.parent.localStorage.removeItem('stSidebarState');
          window.parent.sessionStorage.removeItem('stSidebarState');
        } catch(e) {}

        const sb = doc.querySelector('section[data-testid="stSidebar"]');
        const isClosed = !sb || sb.getAttribute('aria-expanded') === 'false' || sb.offsetWidth < 50;
        
        if (isClosed) {
          const btn = doc.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
                      doc.querySelector('[data-testid="collapsedControl"] button') ||
                      doc.querySelector('[data-testid="stSidebarCollapsedControl"]') ||
                      doc.querySelector('[data-testid="collapsedControl"]') ||
                      doc.querySelector('button[data-testid="stSidebarCollapseButton"]') ||
                      doc.querySelector('button[aria-label*="Expand sidebar"]');
          if (btn) {
            btn.click();
          }
        }
      } catch (err) {
        console.warn("Sidebar auto-open attempt:", err);
      }
    }
    openSidebarNow();
    setTimeout(openSidebarNow, 100);
    setTimeout(openSidebarNow, 400);
    setTimeout(openSidebarNow, 1000);
    </script>
    """, height=0, width=0)

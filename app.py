import streamlit as st
import pandas as pd
import requests
import time
import random
# ### NEW: IMPORTS FOR GOOGLE SHEETS ###
import gspread
from oauth2client.service_account import ServiceAccountCredentials
# --------------------------------------
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats, scoreboardv2, commonteamroster

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="The Prop Auditor",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .metric-card { background-color: #262730; padding: 15px; border-radius: 5px; border-left: 5px solid #4CAF50; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ### NEW: GOOGLE SHEETS CONNECTION FUNCTION ###
def connect_to_sheet():
    """Connects to the Google Sheet using Streamlit Secrets."""
    try:
        # Define the scope (what we are allowed to touch)
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        # Load credentials from the secrets file we updated
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        # Open the specific sheet
        sheet = client.open("Prop_Auditor_Ledger").sheet1
        return sheet
    except Exception as e:
        # Silent fail so the app doesn't crash if secrets aren't set up yet
        return None
# ----------------------------------------------

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧾 The Prop Auditor")
    st.markdown("*Financial Rigor for Sports Betting*")
    st.divider()
    
    # API KEY CHECK
    if "ODDS_API_KEY" in st.secrets:
        api_key = st.secrets["ODDS_API_KEY"]
        st.success("🔐 License Key Active")
    else:
        api_key = st.text_input("Odds API Key", type="password")

    st.divider()

    # ### NEW: SIDEBAR HISTORY (READ THE LEDGER) ###
    st.markdown("### 🏛️ The Vault")
    sheet = connect_to_sheet() # Connect to Google
    if sheet:
        try:
            records = sheet.get_all_records()
            if records:
                df_hist = pd.DataFrame(records)
                # Count Wins (You must manually update the 'Result' column in Sheets to 'WIN' to see this go up!)
                wins = len(df_hist[df_hist['Result'] == 'WIN'])
                total = len(df_hist)
                if total > 0:
                    win_pct = (wins / total) * 100
                    st.metric("All-Time Record", f"{wins}-{total-wins}", f"{win_pct:.1f}% Win Rate")
                else:
                    st.caption("Ledger is active but empty.")
        except:
            st.caption("Connecting to Ledger...")
    else:
        st.caption("⚠️ Ledger Disconnected")
    # ---------------------------------------------
    
    st.divider()
    st.markdown("### ⚙️ Audit Settings")
    min_edge = st.slider("Min Edge (Units)", 1.0, 10.0, 2.0, 0.5)
    show_all = st.checkbox("Show All Audits", value=False, help="Uncheck to hide 'Low Priority' plays")
    
    if not api_key:
        st.warning("⚠️ Please enter API Key to begin.")
        st.stop()

# --- FUNCTIONS (The Engine) ---

@st.cache_data(ttl=3600)
def get_nba_data():
    """Fetches and caches NBA stats for 1 hour."""
    try:
        # 1. Team Stats
        team_stats = leaguedashteamstats.LeagueDashTeamStats(season='2025-26', measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame').get_data_frames()[0]
        if 'PACE' not in team_stats.columns: team_stats.rename(columns={'Pace': 'PACE'}, inplace=True)
        if 'DEF_RATING' not in team_stats.columns: team_stats.rename(columns={'DefRtg': 'DEF_RATING'}, inplace=True)
        
        team_ctx = {row['TEAM_ID']: {'Name': row['TEAM_NAME'], 'Pace': row['PACE'], 'DefRtg': row['DEF_RATING']} for _, row in team_stats.iterrows()}
        lg_pace = team_stats['PACE'].mean()
        lg_def = team_stats['DEF_RATING'].mean()

        # 2. Player Stats
        base = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', measure_type_detailed_defense='Base', per_mode_detailed='PerGame').get_data_frames()[0]
        adv = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame').get_data_frames()[0]
        
        df = pd.merge(base[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'MIN', 'GP', 'PTS', 'REB', 'AST', 'STL', 'BLK']], 
                      adv[['PLAYER_ID', 'DEF_RATING', 'USG_PCT']], on='PLAYER_ID')
        
        # 3. L5 Stats
        l5 = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', last_n_games=5, per_mode_detailed='PerGame').get_data_frames()[0]
        l5 = l5[['PLAYER_ID', 'PTS', 'REB', 'AST']].rename(columns={'PTS': 'L5_PTS', 'REB': 'L5_REB', 'AST': 'L5_AST'})
        df = pd.merge(df, l5, on='PLAYER_ID', how='left')

        return df, team_ctx, lg_pace, lg_def
    except:
        return pd.DataFrame(), {}, 100, 112

def get_market_lines(api_key):
    """Fetches live odds from The Odds API."""
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds?regions=us&markets=player_points,player_rebounds,player_assists&oddsFormat=american&apiKey={api_key}"
    try:
        resp = requests.get(url).json()
        lines = {}
        if isinstance(resp, list):
            for game in resp:
                book = next((b for b in game.get('bookmakers', []) if b['key'] == 'draftkings'), None)
                if not book and game.get('bookmakers'): book = game['bookmakers'][0] # Fallback
                
                if book:
                    for m in book.get('markets', []):
                        m_key = 'PTS' if 'points' in m['key'] else 'REB' if 'rebounds' in m['key'] else 'AST'
                        for out in m.get('outcomes', []):
                            if out.get('point'):
                                if out['description'] not in lines: lines[out['description']] =

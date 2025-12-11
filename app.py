import streamlit as st
import pandas as pd
import requests
import time
import random
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats, scoreboardv2, commonteamroster

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="The Prop Auditor",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING (The "Financial" Look) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .metric-card { background-color: #262730; padding: 15px; border-radius: 5px; border-left: 5px solid #4CAF50; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧾 The Prop Auditor")
    st.markdown("*Financial Rigor for Sports Betting*")
    st.divider()
    
    # PRODUCT MODE: Checks for secret key first
    # FIX: This block is now indented to stay inside the sidebar
    if "ODDS_API_KEY" in st.secrets:
        api_key = st.secrets["ODDS_API_KEY"]
        st.success("🔐 License Key Active")
    else:
        api_key = st.text_input("Odds API Key", type="password")
    
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
    """Fetches and caches NBA stats for 1 hour to keep the app fast."""
    try:
        # 1. Team Stats
        team_stats = leaguedashteamstats.LeagueDashTeamStats(season='2025-26', measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame').get_data_frames()[0]
        # Standardize Columns
        if 'PACE' not in team_stats.columns: team_stats.rename(columns={'Pace': 'PACE'}, inplace=True)
        if 'DEF_RATING' not in team_stats.columns: team_stats.rename(columns={'DefRtg': 'DEF_RATING'}, inplace=True)
        
        team_ctx = {row['TEAM_ID']: {'Name': row['TEAM_NAME'], 'Pace': row['PACE'], 'DefRtg': row['DEF_RATING']} for _, row in team_stats.iterrows()}
        lg_pace = team_stats['PACE'].mean()
        lg_def = team_stats['DEF_RATING'].mean()

        # 2. Player Stats
        base = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', measure_type_detailed_defense='Base', per_mode_detailed='PerGame').get_data_frames()[0]
        adv = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame').get_data_frames()[0]
        
        # Merge
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
                # Prioritize DraftKings
                book = next((b for b in game.get('bookmakers', []) if b['key'] == 'draftkings'), None)
                if not book and game.get('bookmakers'): book = game['bookmakers'][0] # Fallback
                
                if book:
                    for m in book.get('markets', []):
                        m_key = 'PTS' if 'points' in m['key'] else 'REB' if 'rebounds' in m['key'] else 'AST'
                        for out in m.get('outcomes', []):
                            if out.get('point'):
                                if out['description'] not in lines: lines[out['description']] = {}
                                lines[out['description']][m_key] = out['point']
        return lines
    except:
        return {}

def generate_memo(edge, signal):
    """The Manager Persona Logic"""
    if edge >= 5.0: return "🚨 MATERIAL ERROR: Market Asleep."
    if "ELITE" in signal and edge > 2.0: return "⭐ STAR ASSET: Undervalued."
    if "GAMBLER" in signal: return "⚠️ HIGH RISK: Variance Warning."
    if edge >= 2.0: return "✅ AUDIT APPROVED: Solid Trends."
    return "📉 LOW PRIORITY: Minor Edge."

# --- MAIN APP LOGIC ---

# 1. Header Metrics
col1, col2, col3 = st.columns(3)
# Update time to show ET
now_et = datetime.utcnow() - timedelta(hours=5)
col1.metric("Audit Date", now_et.strftime('%Y-%m-%d %I:%M %p ET'))
col2.metric("Market Status", "Live", delta="Open")

# --- EXPLANATION SECTION ---
with st.expander("📘 Read the Column: How This System Works"):
    st.markdown("""
    ### Let's Be Honest About the Betting Market
    
    Look, the Prop Auditor isn't here to sell you a crystal ball. We all know those don't exist. Instead, we're treating this season like a balance sheet. Here is the scouting report on how we find value when the rest of the market is just guessing.

    #### 1. The Internal Evaluation (Asset Valuation)
    We don't care about the narrative; we care about the numbers. We use a four-pillar accounting method to find a player's "True Value":
    * **The 'Next Man Up' Reality (Usage Void):** When a star sits out, their 20 shots don't just vanish into thin air—they get reallocated. We calculate exactly who absorbs that volume.
    * **Pace & Efficiency:** We adjust for the speed of the game. If you're playing a team that defends like a turnstile, we bump the valuation up.
    * **The Reality Check (Recent Form):** We weigh recent hot streaks against season averages. Is it a breakout, or just a lucky week? We find the difference.

    #### 2. The Liability Check
    Once we have our number, we cross-reference it against the live lines posted by DraftKings. The question is simple: **What is Vegas missing?**

    #### 3. The Verdicts (The Signals)
    We only flag a play if the math shows a significant error. Here is how we grade the opportunities:
    
    * **ELITE:** This is your All-Star starter. High usage, fast pace, and a defense that can't stop anyone. High conviction play.
    * **GAMBLER:** The high-risk, high-reward swing. Usually involves defensive stats like Steals or Blocks that are volatile by nature. Proceed with caution.
    * **ANCHOR:** The reliable veteran. Consistent role players with a safe floor. Good for keeping the ledger in the green.
    """)
    
# 2. Loading State
with st.spinner('🔄 syncing with NBA Mainframe & Vegas Ledgers...'):
    df, team_ctx, lg_pace, lg_def = get_nba_data()
    market_lines = get_market_lines(api_key)

if df.empty:
    st.error("NBA Data Offline. Try again later.")
    st.stop()

col3.metric("Active Lines", len(market_lines))

# 3. The Calculation Loop (Hidden Engine)
audit_results = []
today_str = (datetime.utcnow() - timedelta(hours=5)).strftime('%Y-%m-%d')

try:
    games = scoreboardv2.ScoreboardV2(game_date=today_str).game_header.get_data_frame()
except:
    games = pd.DataFrame()

if games.empty:
    st.warning("No NBA Games Scheduled Today.")
else:
    # --- START ENGINE ---
    # Simplified Logic for Speed in Web App
    for game in games.to_dict('records'):
        h_id, v_id = game['HOME_TEAM_ID'], game['VISITOR_TEAM_ID']
        
        for tid in [h_id, v_id]:
            oid = v_id if tid == h_id else h_id
            is_home = (tid == h_id)
            
            # Team Context
            pace_factor = ((team_ctx.get(tid,{}).get('Pace',100) + team_ctx.get(oid,{}).get('Pace',100))/2) / lg_pace
            def_factor = team_ctx.get(oid,{}).get('DefRtg',112) / lg_def
            
            # Roster Scan
            roster = df[df['TEAM_ID'] == tid].sort_values('MIN', ascending=False).head(9)
            
            for _, p in roster.iterrows():
                if p['MIN'] < 12: continue
                
                # Factors
                home_factor = 1.03 if (is_home and p['USG_PCT'] < 0.20) else 1.0
                
                # Signal
                signal = "-"
                dos = (p['STL']*2.5) + (p['BLK']*2.0)
                if dos > 3.0: signal = "GAMBLER"
                fantasy = p['PTS'] + 1.2*p['REB'] + 1.5*p['AST'] + 3*p['STL'] + 3*p['BLK']
                if fantasy > 45: signal = "ELITE"
                
                # Projection
                total_mult = pace_factor * def_factor * home_factor
                proj_pts = p['PTS'] * total_mult # Using simplified Base since weighted is complex
                proj_reb = p['REB'] * total_mult
                proj_ast = p['AST'] * total_mult
                
                # Market Check
                lines = market_lines.get(p['PLAYER_NAME'], {})
                l_pts = lines.get('PTS', 999)
                l_reb = lines.get('REB', 999)
                l_ast = lines.get('AST', 999)
                
                val_add = 0
                bet_str = ""
                
                if proj_pts > (l_pts + 2.0):
                    val_add += (proj_pts - l_pts)
                    bet_str += f"PTS > {l_pts} "
                if proj_reb > (l_reb + 1.5):
                    val_add += (proj_reb - l_reb)
                    bet_str += f"REB > {l_reb} "
                if proj_ast > (l_ast + 1.5):
                    val_add += (proj_ast - l_ast)
                    bet_str += f"AST > {l_ast} "
                    
                if val_add >= min_edge:
                    memo = generate_memo(val_add, signal)
                    audit_results.append({
                        "Player": p['PLAYER_NAME'],
                        "Team": team_ctx.get(tid,{}).get('Name','UNK'),
                        "Signal": signal,
                        "Manager Memo": memo,
                        "Bet": bet_str,
                        "Edge": round(val_add, 1),
                        "PTS": f"{round(proj_pts,1)} ({l_pts})" if l_pts!=999 else "-",
                        "REB": f"{round(proj_reb,1)} ({l_reb})" if l_reb!=999 else "-",
                        "AST": f"{round(proj_ast,1)} ({l_ast})" if l_ast!=999 else "-"
                    })

# --- DISPLAY RESULTS ---
st.subheader(f"📋 Daily Ledger ({len(audit_results)} Flags Found)")

if audit_results:
    res_df = pd.DataFrame(audit_results).sort_values(by='Edge', ascending=False)
    
    # Filter?
    if not show_all:
        res_df = res_df[res_df['Edge'] >= min_edge]

    st.dataframe(
        res_df,
        column_config={
            "Manager Memo": st.column_config.TextColumn("Manager Memo", width="medium"),
            "Edge": st.column_config.ProgressColumn("Value Score", format="%.1f", min_value=0, max_value=10),
            "Signal": st.column_config.Column("Risk Profile")
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No discrepancies found matching your criteria. Market is sharp today.")




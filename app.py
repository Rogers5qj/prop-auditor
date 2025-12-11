import streamlit as st
import pandas as pd
import requests
import time
import random
import re 
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats, scoreboardv2, commonteamroster

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="The Prop Auditor", page_icon="🧾", layout="wide", initial_sidebar_state="expanded")

# --- STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .metric-card { background-color: #262730; padding: 15px; border-radius: 5px; border-left: 5px solid #4CAF50; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# --- GOOGLE SHEETS CONNECTION ---
def connect_to_sheet():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open("Prop_Auditor_Ledger").sheet1
    except: return None

# --- AUTO-GRADING ENGINE (SMART V8) ---
def grade_pending_bets(sheet):
    """Checks PENDING rows against actual stats and updates the sheet."""
    try:
        data = sheet.get_all_records()
        if not data: return "Sheet is empty."
        
        updates_made = 0
        stats_cache = {}
        log_msgs = [] # To show the user what happened
        
        # Loop through every row
        for i, row in enumerate(data):
            if row['Result'] == 'PENDING':
                date_str = str(row['Date']).strip()
                player = row['Player']
                bet_str = str(row['Bet'])
                
                # 1. ROBUST DATE PARSING
                d_obj = None
                for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%m-%d-%Y'):
                    try:
                        d_obj = datetime.strptime(date_str, fmt)
                        break
                    except: continue
                
                if not d_obj:
                    log_msgs.append(f"❌ Skipped {player}: Invalid Date '{date_str}'")
                    continue
                
                fmt_date = d_obj.strftime('%m/%d/%Y') # Format API expects
                cache_key = d_obj.strftime('%Y-%m-%d')
                
                # 2. Fetch Stats (Cache)
                if cache_key not in stats_cache:
                    try:
                        stats = leaguedashplayerstats.LeagueDashPlayerStats(
                            date_from_nullable=fmt_date, 
                            date_to_nullable=fmt_date, 
                            season='2025-26', 
                            per_mode_detailed='PerGame'
                        ).get_data_frames()[0]
                        stats_cache[cache_key] = stats
                        time.sleep(0.2)
                    except:
                        log_msgs.append(f"⚠️ API Error for {fmt_date}")
                        stats_cache[cache_key] = pd.DataFrame()
                
                # 3. Find Player
                daily_df = stats_cache[cache_key]
                if daily_df.empty: continue
                
                p_stats = daily_df[daily_df['PLAYER_NAME'] == player]
                if p_stats.empty:
                    log_msgs.append(f"⚠️ {player} not found in box score for {fmt_date}")
                    continue
                
                # 4. Get Actuals
                act_pts = float(p_stats.iloc[0]['PTS'])
                act_reb = float(p_stats.iloc[0]['REB'])
                act_ast = float(p_stats.iloc[0]['AST'])
                
                # 5. Check Bet (SMART REGEX)
                # Handles "PTS > 14.5" AND "Over 14.5" (Defaults 'Over' to PTS)
                conditions = re.findall(r'(PTS|REB|AST|Over|Points)\s*(?:>)?\s*([\d\.]+)', bet_str, re.IGNORECASE)
                
                if not conditions:
                    log_msgs.append(f"⚠️ Could not read bet format: '{bet_str}'")
                    continue

                won = True
                for cat, val in conditions:
                    target = float(val)
                    cat_clean = cat.upper()
                    
                    # LOGIC: If "Over", assume Points. 
                    if (cat_clean == 'PTS' or cat_clean == 'OVER' or cat_clean == 'POINTS') and act_pts <= target: won = False
                    elif cat_clean == 'REB' and act_reb <= target: won = False
                    elif cat_clean == 'AST' and act_ast <= target: won = False
                
                # 6. Update Sheet
                result_text = "WIN" if won else "LOSS"
                sheet.update_cell(i + 2, 6, result_text) 
                updates_made += 1
                log_msgs.append(f"✅ Graded {player}: {result_text}")
                
        return f"Graded {updates_made} bets.", log_msgs
    except Exception as e:
        return f"Grading Error: {e}", []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧾 The Prop Auditor")
    st.markdown("*Financial Rigor for Sports Betting*")
    st.divider()
    
    if "ODDS_API_KEY" in st.secrets:
        api_key = st.secrets["ODDS_API_KEY"]
        st.success("🔐 License Key Active")
    else:
        api_key = st.text_input("Odds API Key", type="password")

    st.divider()

    # --- VAULT (HISTORY) ---
    st.markdown("### 🏛️ The Vault")
    sheet = connect_to_sheet() 
    
    if sheet:
        # AUTO GRADE BUTTON
        if st.button("🔄 Auto-Grade Pending"):
            with st.spinner("Auditing past performance..."):
                status, logs = grade_pending_bets(sheet)
                
                # Show results
                if "Error" in status: 
                    st.error(status)
                elif int(re.search(r'\d+', status).group()) == 0:
                    st.warning(status)
                    # Show the Debug Logs so user knows WHY it failed
                    with st.expander("View Debug Logs"):
                        for log in logs:
                            st.write(log)
                else: 
                    st.success(status)
                    time.sleep(2)
                    st.rerun()

        try:
            records = sheet.get_all_records()
            if records:
                df_hist = pd.DataFrame(records)
                graded = df_hist[df_hist['Result'].isin(['WIN', 'LOSS'])]
                wins = len(graded[graded['Result'] == 'WIN'])
                total = len(graded)
                losses = total - wins
                if total > 0:
                    win_pct = (wins / total) * 100
                    st.metric("All-Time Record", f"{wins}-{losses}", f"{win_pct:.1f}% Win Rate")
                else:
                    st.metric("All-Time Record", "0-0", "Pending Results")
            else:
                st.caption("Ledger is active but empty.")
        except:
            st.caption("Connecting to Ledger...")
        else:
            st.caption("⚠️ Ledger Disconnected")
    
    st.divider()
    
    # --- NEW: DATE PICKER ---
    st.markdown("### 📅 Select Date")
    # Default to today
    default_date = datetime.utcnow() - timedelta(hours=5)
    selected_date = st.date_input("Audit Date", default_date)
    # ------------------------

    st.markdown("### ⚙️ Audit Settings")

    
    st.divider()
    st.markdown("### ⚙️ Audit Settings")
    min_edge = st.slider("Min Edge (Units)", 1.0, 10.0, 2.0, 0.5)
    show_all = st.checkbox("Show All Audits", value=False)
    
    if not api_key:
        st.warning("⚠️ Please enter API Key to begin.")
        st.stop()

# --- FUNCTIONS (ENGINE) ---
@st.cache_data(ttl=3600)
def get_nba_data():
    """Fetches NBA player stats and creates a Name->ID map."""
    try:
        # 1. Team Stats
        team_stats = leaguedashteamstats.LeagueDashTeamStats(season='2025-26', measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame').get_data_frames()[0]
        if 'PACE' not in team_stats.columns: team_stats.rename(columns={'Pace': 'PACE'}, inplace=True)
        if 'DEF_RATING' not in team_stats.columns: team_stats.rename(columns={'DefRtg': 'DEF_RATING'}, inplace=True)
        
        # Create Maps
        team_ctx = {row['TEAM_ID']: {'Name': row['TEAM_NAME'], 'Pace': row['PACE'], 'DefRtg': row['DEF_RATING']} for _, row in team_stats.iterrows()}
        # NEW: Name to ID Map (Crucial for linking Odds API to NBA API)
        name_to_id_map = {row['TEAM_NAME']: row['TEAM_ID'] for _, row in team_stats.iterrows()}
        # Manual Fix for Clippers (Common mismatch)
        name_to_id_map['LA Clippers'] = 1610612746
        name_to_id_map['Los Angeles Clippers'] = 1610612746
        
        lg_pace = team_stats['PACE'].mean(); lg_def = team_stats['DEF_RATING'].mean()

        # 2. Player Stats
    base = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', measure_type_detailed_defense='Base', per_mode_detailed='PerGame').get_data_frames()[0]
    adv = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame').get_data_frames()[0]
    df = pd.merge(base[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'MIN', 'GP', 'PTS', 'REB', 'AST', 'STL', 'BLK']], adv[['PLAYER_ID', 'DEF_RATING', 'USG_PCT']], on='PLAYER_ID')
        
        # 3. L5 Stats
        l5 = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', last_n_games=5, per_mode_detailed='PerGame').get_data_frames()[0]
        l5 = l5[['PLAYER_ID', 'PTS', 'REB', 'AST']].rename(columns={'PTS': 'L5_PTS', 'REB': 'L5_REB', 'AST': 'L5_AST'})
        df = pd.merge(df, l5, on='PLAYER_ID', how='left')
        
        return df, team_ctx, name_to_id_map, lg_pace, lg_def
    except: return pd.DataFrame(), {}, {}, 100, 112


        base = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', measure_type_detailed_defense='Base', per_mode_detailed='PerGame').get_data_frames()[0]
        adv = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame').get_data_frames()[0]
        df = pd.merge(base[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'MIN', 'GP', 'PTS', 'REB', 'AST', 'STL', 'BLK']], adv[['PLAYER_ID', 'DEF_RATING', 'USG_PCT']], on='PLAYER_ID')
        
        l5 = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', last_n_games=5, per_mode_detailed='PerGame').get_data_frames()[0]
        l5 = l5[['PLAYER_ID', 'PTS', 'REB', 'AST']].rename(columns={'PTS': 'L5_PTS', 'REB': 'L5_REB', 'AST': 'L5_AST'})
        df = pd.merge(df, l5, on='PLAYER_ID', how='left')
        return df, team_ctx, lg_pace, lg_def
    except: return pd.DataFrame(), {}, 100, 112

def get_market_data(api_key):
    """Fetches BOTH lines AND the schedule from the Odds API."""
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds?regions=us&markets=player_points,player_rebounds,player_assists&oddsFormat=american&apiKey={api_key}"
    try:
        resp = requests.get(url).json()
        lines = {}
        schedule = [] # List of games found in the market
        
        if isinstance(resp, list):
            for game in resp:
                # 1. Build Schedule from Odds API
                schedule.append({
                    'home_team': game['home_team'],
                    'away_team': game['away_team']
                })
                
                # 2. Extract Lines
                book = next((b for b in game.get('bookmakers', []) if b['key'] == 'draftkings'), None)
                if not book and game.get('bookmakers'): book = game['bookmakers'][0]
                if book:
                    for m in book.get('markets', []):
                        m_key = 'PTS' if 'points' in m['key'] else 'REB' if 'rebounds' in m['key'] else 'AST'
                        for out in m.get('outcomes', []):
                            if out.get('point'):
                                if out['description'] not in lines: lines[out['description']] = {}
                                lines[out['description']][m_key] = out['point']
        return lines, schedule
    except: return {}, []

def generate_memo(edge, signal):
    if edge >= 5.0: return "🚨 MATERIAL ERROR: Market Asleep."
    if "ELITE" in signal and edge > 2.0: return "⭐ STAR ASSET: Undervalued."
    if "GAMBLER" in signal: return "⚠️ HIGH RISK: Variance Warning."
    if edge >= 2.0: return "✅ AUDIT APPROVED: Solid Trends."
    return "📉 LOW PRIORITY: Minor Edge."

# --- MAIN APP ---
col1, col2, col3 = st.columns(3)
now_et = datetime.utcnow() - timedelta(hours=5)
col1.metric("Audit Date", now_et.strftime('%Y-%m-%d %I:%M %p ET'))
col2.metric("Market Status", "Live", delta="Open")

with st.expander("📘 Read the Column: How This System Works"):
    st.markdown("""
    ### Let's Be Honest About the Betting Market
    We treat this season like a balance sheet. Here is the scouting report:
    * **The 'Next Man Up' Reality (Usage Void):** When a star sits, we calculate exactly who absorbs the volume.
    * **Pace & Efficiency:** We adjust for the speed of the game and opponent DrTg.
    * **The Reality Check:** We weigh recent hot streaks against sustainable averages.
    """)

with st.spinner('🔄 syncing with NBA Mainframe & Vegas Ledgers...'):
    # Use Odds API for EVERYTHING (Bypasses NBA Schedule Block)
with st.spinner('🔄 syncing with Market Data...'):
    df, team_ctx, name_to_id, lg_pace, lg_def = get_nba_data()
    market_lines, market_schedule = get_market_data(api_key)

col3.metric("Active Lines", len(market_lines))

audit_results = []

# --- NEW ENGINE: LOOP THROUGH ODDS API SCHEDULE ---
if market_schedule and not df.empty:
    for game in market_schedule:
        # Convert Names to IDs
        h_name = game['home_team']
        v_name = game['away_team']
        
        # Safe Lookup using the map we built in Change 1
        h_id = name_to_id.get(h_name, 0)
        v_id = name_to_id.get(v_name, 0)
        
        if h_id == 0 or v_id == 0:
            continue # Skip if name mismatch
            
        for tid in [h_id, v_id]:
            oid = v_id if tid == h_id else h_id
            is_home = (tid == h_id)
            
            # (Rest of logic is identical to before)
            pace_factor = ((team_ctx.get(tid,{}).get('Pace',100) + team_ctx.get(oid,{}).get('Pace',100))/2) / lg_pace
            def_factor = team_ctx.get(oid,{}).get('DefRtg',112) / lg_def
            roster = df[df['TEAM_ID'] == tid].sort_values('MIN', ascending=False).head(9)
            
            for _, p in roster.iterrows():
                if p['MIN'] < 12: continue
                home_factor = 1.03 if (is_home and p['USG_PCT'] < 0.20) else 1.0
                signal = "-"
                dos = (p['STL']*2.5) + (p['BLK']*2.0)
                if dos > 3.0: signal = "GAMBLER"
                fantasy = p['PTS'] + 1.2*p['REB'] + 1.5*p['AST'] + 3*p['STL'] + 3*p['BLK']
                if fantasy > 45: signal = "ELITE"
                
                total_mult = pace_factor * def_factor * home_factor
                proj_pts = p['PTS'] * total_mult
                proj_reb = p['REB'] * total_mult
                proj_ast = p['AST'] * total_mult
                lines = market_lines.get(p['PLAYER_NAME'], {})
                l_pts = lines.get('PTS', 999); l_reb = lines.get('REB', 999); l_ast = lines.get('AST', 999)
                val_add = 0; bet_str = ""
                
                if proj_pts > (l_pts + 2.0): val_add += (proj_pts - l_pts); bet_str += f"PTS > {l_pts} "
                if proj_reb > (l_reb + 1.5): val_add += (proj_reb - l_reb); bet_str += f"REB > {l_reb} "
                if proj_ast > (l_ast + 1.5): val_add += (proj_ast - l_ast); bet_str += f"AST > {l_ast} "
                
                if val_add >= min_edge or show_all:
                    memo = generate_memo(val_add, signal)
                    audit_results.append({
                        "Date": today_str,
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

st.subheader(f"📋 Daily Ledger ({len(audit_results)} Flags Found)")

if audit_results:
    res_df = pd.DataFrame(audit_results).sort_values(by='Edge', ascending=False)
    if not show_all: res_df = res_df[res_df['Edge'] >= min_edge]
    st.dataframe(res_df.drop(columns=['Date']), column_config={
        "Manager Memo": st.column_config.TextColumn("Manager Memo", width="medium"),
        "Edge": st.column_config.ProgressColumn("Value Score", format="%.1f", min_value=0, max_value=10),
    }, use_container_width=True, hide_index=True)
    
    if st.button("💾 Commit to Ledger (Google Sheets)"):
        if sheet:
            try:
                for item in audit_results:
                    # SAFETY CHECK: Only save if the edge is actually good.
                    # This prevents "Show All" garbage from polluting your sheet.
                    if item['Edge'] >= min_edge: 
                        existing = sheet.findall(item['Player'])
                        sheet.append_row([item['Date'], item['Player'], item['Team'], item['Bet'], item['Edge'], "PENDING"])
                st.success("✅ Successfully updated the Master Ledger!"); st.balloons()
            except Exception as e: st.error(f"Error saving to sheet: {e}")
        else: st.error("Sheet connection not active. Check Secrets.")
else: st.info("No discrepancies found matching your criteria. Market is sharp today.")

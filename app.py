import streamlit as st
import pandas as pd
import requests
import time
import random
import re 
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta 
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats

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

# --- AUTO-GRADING ENGINE (UPDATED FOR UNDERS) ---
def grade_pending_bets(sheet):
    """Checks PENDING rows against actual stats."""
    try:
        data = sheet.get_all_records()
        if not data: return "Sheet is empty."
        
        updates_made = 0
        stats_cache = {}
        log_msgs = [] 
        
        for i, row in enumerate(data):
            if row['Result'] == 'PENDING':
                date_str = str(row['Date']).strip()
                player = row['Player']
                bet_str = str(row['Bet'])
                
                # Robust Date Parsing
                d_obj = None
                for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%m-%d-%Y'):
                    try:
                        d_obj = datetime.strptime(date_str, fmt)
                        break
                    except: continue
                
                if not d_obj:
                    log_msgs.append(f"❌ Skipped {player}: Invalid Date '{date_str}'")
                    continue
                
                fmt_date = d_obj.strftime('%m/%d/%Y')
                cache_key = d_obj.strftime('%Y-%m-%d')
                
                # Cache Stats
                if cache_key not in stats_cache:
                    try:
                        stats = leaguedashplayerstats.LeagueDashPlayerStats(
                            date_from_nullable=fmt_date, date_to_nullable=fmt_date, 
                            season='2025-26', per_mode_detailed='PerGame'
                        ).get_data_frames()[0]
                        stats_cache[cache_key] = stats
                        time.sleep(0.2)
                    except:
                        log_msgs.append(f"⚠️ API Error for {fmt_date}")
                        stats_cache[cache_key] = pd.DataFrame()
                
                # Check Player
                daily_df = stats_cache[cache_key]
                if daily_df.empty: continue
                
                p_stats = daily_df[daily_df['PLAYER_NAME'] == player]
                if p_stats.empty: continue
                
                act_pts = float(p_stats.iloc[0]['PTS'])
                act_reb = float(p_stats.iloc[0]['REB'])
                act_ast = float(p_stats.iloc[0]['AST'])
                
                # --- NEW GRADING LOGIC (Handles < and >) ---
                # Regex now captures the operator (> or <)
                conditions = re.findall(r'(PTS|REB|AST)\s*(>|<)\s*([\d\.]+)', bet_str, re.IGNORECASE)
                
                # Fallback for old bets without operator (Assume Over)
                if not conditions:
                    old_conds = re.findall(r'(PTS|REB|AST|Points)\s*(?:>)?\s*([\d\.]+)', bet_str, re.IGNORECASE)
                    conditions = [(c[0], '>', c[1]) for c in old_conds]

                won = True
                for cat, op, val in conditions:
                    target = float(val)
                    cat_clean = cat.upper()
                    
                    actual = 0.0
                    if cat_clean in ['PTS', 'POINTS']: actual = act_pts
                    elif cat_clean == 'REB': actual = act_reb
                    elif cat_clean == 'AST': actual = act_ast
                    
                    # Grade based on Operator
                    if op == '>' and actual <= target: won = False
                    elif op == '<' and actual >= target: won = False
                
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

# ... inside st.sidebar, before the Audit Settings ...
    
    # 1. Create the Parking Spot
    injury_spot = st.empty()  
    
    st.divider()
    st.markdown("### ⚙️ Audit Settings")
    
    st.divider()

    # Vault Section
    st.markdown("### 🏛️ The Vault")
    sheet = connect_to_sheet() 
    if sheet:
        if st.button("🔄 Auto-Grade Pending"):
            with st.spinner("Auditing past performance..."):
                status, logs = grade_pending_bets(sheet)
                if "Error" in status: st.error(status)
                else: 
                    st.success(status)
                    time.sleep(1)
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
            else: st.caption("Ledger is active but empty.")
        except: st.caption("Connecting to Ledger...")
    else: st.caption("⚠️ Ledger Disconnected")
    
    st.divider()
    st.markdown("### ⚙️ Audit Settings")
    min_edge = st.slider("Min Edge (Units)", 1.0, 10.0, 2.0, 0.5)
    show_all = st.checkbox("Show All Audits", value=False)
    
    if not api_key: st.warning("⚠️ Please enter API Key."); st.stop()

# --- FUNCTIONS (ENGINE) ---
# --- FUNCTIONS (ENGINE) ---
@st.cache_data(ttl=3600)
def get_nba_data():
    """Fetches Stats + Calculates Volatility (Consistency) & Shot Quality."""
    try:
        # 1. Team Stats (Advanced) - Gets PACE and DEF_RATING
        adv_stats = leaguedashteamstats.LeagueDashTeamStats(
            season='2025-26', 
            measure_type_detailed_defense='Advanced', 
            per_mode_detailed='PerGame'
        ).get_data_frames()[0]

        # 2. Team Stats (Four Factors) - Gets OPP_EFG_PCT (The missing piece!)
        four_factors = leaguedashteamstats.LeagueDashTeamStats(
            season='2025-26', 
            measure_type_detailed_defense='Four Factors', 
            per_mode_detailed='PerGame'
        ).get_data_frames()[0]

        # Merge them together on TEAM_ID
        team_stats = pd.merge(adv_stats, four_factors[['TEAM_ID', 'OPP_EFG_PCT']], on='TEAM_ID')
        
        # Rename columns for clarity
        cols_map = {'Pace': 'PACE', 'DefRtg': 'DEF_RATING', 'OPP_EFG_PCT': 'OPP_EFG'}
        team_stats.rename(columns={k:v for k,v in cols_map.items() if k in team_stats.columns}, inplace=True)
        
        # Create Context Maps
        team_ctx = {
            row['TEAM_ID']: {
                'Name': row['TEAM_NAME'], 
                'Pace': row['PACE'], 
                'DefRtg': row['DEF_RATING'],
                'OppEfg': row['OPP_EFG'] # Now this exists!
            } for _, row in team_stats.iterrows()
        }
        
        name_to_id_map = {row['TEAM_NAME']: row['TEAM_ID'] for _, row in team_stats.iterrows()}
        name_to_id_map['LA Clippers'] = 1610612746
        name_to_id_map['Los Angeles Clippers'] = 1610612746
        
        lg_pace = team_stats['PACE'].mean()
        lg_def = team_stats['DEF_RATING'].mean()
        lg_efg = team_stats['OPP_EFG'].mean()

        # 3. Player Stats (Base)
        base = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', measure_type_detailed_defense='Base', per_mode_detailed='PerGame').get_data_frames()[0]
        adv = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame').get_data_frames()[0]
        df = pd.merge(base[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'MIN', 'GP', 'PTS', 'REB', 'AST', 'STL', 'BLK']], adv[['PLAYER_ID', 'DEF_RATING', 'USG_PCT']], on='PLAYER_ID')
        
        # 4. L5 Stats
        l5 = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', last_n_games=5, per_mode_detailed='PerGame').get_data_frames()[0]
        l5 = l5[['PLAYER_ID', 'PTS', 'REB', 'AST']].rename(columns={'PTS': 'L5_PTS', 'REB': 'L5_REB', 'AST': 'L5_AST'})
        df = pd.merge(df, l5, on='PLAYER_ID', how='left')

        # 5. CONSISTENCY ENGINE (Standard Deviation)
        from nba_api.stats.endpoints import leaguegamelog
        logs = leaguegamelog.LeagueGameLog(season='2025-26', player_or_team_abbreviation='P').get_data_frames()[0]
        
        volatility_map = logs.groupby('PLAYER_ID')['PTS'].std().to_dict()
        df['PTS_VOLATILITY'] = df['PLAYER_ID'].map(volatility_map).fillna(5.0) 

        return df, team_ctx, name_to_id_map, lg_pace, lg_def, lg_efg
    except Exception as e: 
        st.error(f"NBA Data Error: {e}")
        return pd.DataFrame(), {}, {}, 100, 112, 0.55

@st.cache_data(ttl=1600, show_spinner=False)
def get_market_data(api_key, target_date):
    """Fetches Schedule + Spreads + Props."""
    lines = {}
    schedule = [] 
    game_spreads = {} # <--- NEW: Store spreads here
    
    # 1. GET SCHEDULE & SPREADS
    sched_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds?regions=us&markets=h2h,spreads&oddsFormat=american&apiKey={api_key}"
    
    try:
        sched_resp = requests.get(sched_url).json()
        if isinstance(sched_resp, dict) and 'message' in sched_resp:
            return {}, {}, {} 
            
        if not isinstance(sched_resp, list): return {}, {}, {}

        for game in sched_resp:
            # --- DATE FILTER ---
            try:
                start_str = game['commence_time'].replace('Z', '')
                start_dt = datetime.fromisoformat(start_str) - timedelta(hours=5)
                if start_dt.strftime('%Y-%m-%d') != target_date: continue
            except: continue 
            
            game_id = game['id']
            
            # --- NEW: EXTRACT SPREAD ---
            spread_val = 0.0
            book = next((b for b in game.get('bookmakers', []) if b['key'] == 'draftkings'), None)
            if book:
                for m in book.get('markets', []):
                    if m['key'] == 'spreads':
                        if len(m['outcomes']) > 0:
                            spread_val = abs(m['outcomes'][0].get('point', 0))
            
            schedule.append({'home_team': game['home_team'], 'away_team': game['away_team'], 'id': game_id})
            game_spreads[game['home_team']] = spread_val
            game_spreads[game['away_team']] = spread_val
            
            # 2. GET PROPS (Loop)
            prop_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{game_id}/odds?regions=us&markets=player_points,player_rebounds,player_assists&oddsFormat=american&apiKey={api_key}"
            try:
                prop_data = requests.get(prop_url).json()
                p_book = next((b for b in prop_data.get('bookmakers', []) if b['key'] == 'draftkings'), None)
                if not p_book and prop_data.get('bookmakers'): p_book = prop_data['bookmakers'][0]
                
                if p_book:
                    for m in p_book.get('markets', []):
                        m_key = 'PTS' if 'points' in m['key'] else 'REB' if 'rebounds' in m['key'] else 'AST'
                        for out in m.get('outcomes', []):
                            if out.get('point'):
                                if out['description'] not in lines: lines[out['description']] = {}
                                lines[out['description']][m_key] = out['point']
                time.sleep(0.1) 
            except: continue

        return lines, schedule, game_spreads # <--- Return 3 values

    except: return {}, {}, {}


def generate_memo(edge, signal):
    if edge >= 5.0: return "🚨 MATERIAL ERROR: Market Asleep."
    if "ELITE" in signal and edge > 2.0: return "⭐ STAR ASSET: Undervalued."
    if "GAMBLER" in signal: return "⚠️ HIGH RISK: Variance Warning."
    if edge >= 2.0: return "✅ AUDIT APPROVED: Solid Trends."
    return "📉 LOW PRIORITY: Minor Edge."

# --- MAIN APP ---
col1, col2, col3 = st.columns(3)
now_et = datetime.utcnow() - timedelta(hours=5)
today_str = now_et.strftime('%Y-%m-%d')
col1.metric("Audit Date", now_et.strftime('%Y-%m-%d %I:%M %p ET'))
col2.metric("Market Status", "Live", delta="Open")

# Use Odds API for EVERYTHING (Bypasses NBA Schedule Block)
with st.spinner('🔄 Running Crystal Ball Algorithms...'):
    # Unpack the new 6 return values
    df, team_ctx, name_to_id, lg_pace, lg_def, lg_efg = get_nba_data()
    # Unpack the new 3 return values
    market_lines, market_schedule, market_spreads = get_market_data(api_key, today_str)


col3.metric("Active Lines", len(market_lines))

# --- INSERT THIS BLOCK AFTER LINE 252 ---

# --- NEW: INJURY MANAGER (SIDEBAR) ---
# We add this here so it loads AFTER we get the data
# --- CHANGE THIS BLOCK AT THE BOTTOM ---

# Instead of "with st.sidebar:", we use the parking spot we made earlier
with injury_spot.container():
    st.divider()
    st.markdown("### 🚑 Injury Override")
    
    # 1. Select the Team with Injuries
    team_list = sorted(list(name_to_id.keys()))
    injury_team = st.selectbox("Select Team with Missing Star:", ["None"] + team_list)
    
    # ... (Rest of the code stays exactly the same) ...
    
    # 2. Define the Impact
    usage_bump = 1.0
    if injury_team != "None":
        st.warning(f"⚠️ Adjusting usage for {injury_team}...")
        bump_pct = st.slider(f"Usage Bump for {injury_team} Teammates:", 0, 30, 15)
        usage_bump = 1.0 + (bump_pct / 100.0)
        st.caption(f"Applying {usage_bump}x multiplier to projections.")

    # 3. The 'Questionable' Kill Switch
    # We pull team names from the market schedule
    game_teams = sorted(list(set([s['home_team'] for s in market_schedule] + [s['away_team'] for s in market_schedule])))
    void_games = st.multiselect("⛔ VOID Games (Too much uncertainty):", game_teams)

audit_results = []

if market_schedule and not df.empty:
    for game in market_schedule:
        h_name, v_name = game['home_team'], game['away_team']
        h_id, v_id = name_to_id.get(h_name, 0), name_to_id.get(v_name, 0)
        
        if h_id == 0 or v_id == 0: continue 

        # --- 1. GET SPREAD FOR BLOWOUT CHECK ---
        spread = market_spreads.get(h_name, 0)
        blowout_risk = spread > 12.5 # Threshold for blowout
        
        for tid in [h_id, v_id]:
            oid = v_id if tid == h_id else h_id
            is_home = (tid == h_id)
            
            # --- 2. ADVANCED FACTORS ---
            pace_factor = ((team_ctx.get(tid,{}).get('Pace',100) + team_ctx.get(oid,{}).get('Pace',100))/2) / lg_pace
            
            # Defense (NOW INCLUDES eFG% - Shot Quality)
            opp_def = team_ctx.get(oid,{}).get('DefRtg', 112)
            opp_efg = team_ctx.get(oid,{}).get('OppEfg', 0.55)
            
            # We average the Rating Factor and the Shot Quality Factor
            def_rating_factor = opp_def / lg_def
            shot_quality_factor = opp_efg / lg_efg
            combined_def_factor = (def_rating_factor + shot_quality_factor) / 2
            
            roster = df[df['TEAM_ID'] == tid].sort_values('MIN', ascending=False).head(9)
            
# --- REPLACE THE INSIDE OF THE PLAYER LOOP WITH THIS ---
            for _, p in roster.iterrows():
                if p['MIN'] < 12: continue
                
                # A. CHECK FOR VOID (Kill Switch)
                current_team_name = team_ctx.get(tid,{}).get('Name')
                if current_team_name in void_games: continue

                # B. APPLY INJURY BOOST (Usage Bump)
                # If this player is on the team you selected, boost them
                active_bump = usage_bump if current_team_name == injury_team else 1.0
                
               # --- 3. APPLY CONSISTENCY & STRESS TEST ---
                # "Safe Base" tests the floor (for Overs)
                safe_pts_base = p['PTS'] - (0.5 * p['PTS_VOLATILITY'])
                
                # "High Base" tests the ceiling (for Unders)
                high_pts_base = p['PTS'] + (0.5 * p['PTS_VOLATILITY'])
                
                # --- 4. APPLY BLOWOUT TAX & INJURY BUMP ---
                blowout_tax = 0.90 if blowout_risk else 1.0
                home_factor = 1.03 if (is_home and p['USG_PCT'] < 0.20) else 1.0
                
                # FINAL CRYSTAL BALL PROJECTION (Ranges)
                # We added 'active_bump' to the end here
                total_mult = pace_factor * combined_def_factor * home_factor * blowout_tax * active_bump
                
                # Low projection (Conservative) - Use for OVERS
                proj_pts_low = safe_pts_base * total_mult 
                proj_reb_low = p['REB'] * total_mult
                proj_ast_low = p['AST'] * total_mult

                # High projection (Aggressive) - Use for UNDERS
                proj_pts_high = high_pts_base * total_mult
                proj_reb_high = (p['REB'] + (0.5 * 2.0)) * total_mult 
                proj_ast_high = (p['AST'] + (0.5 * 1.5)) * total_mult
                
                # Calc Edges (Bidirectional Audit)
                # CHECK OVERS (Compare vs Low Projection)
                if l_pts != 999 and proj_pts_low > (l_pts + 2.0): 
                    val_add += (proj_pts_low - l_pts)
                    bet_str += f"PTS > {l_pts} "
                if l_reb != 999 and proj_reb_low > (l_reb + 1.5): 
                    val_add += (proj_reb_low - l_reb)
                    bet_str += f"REB > {l_reb} "
                if l_ast != 999 and proj_ast_low > (l_ast + 1.5): 
                    val_add += (proj_ast_low - l_ast)
                    bet_str += f"AST > {l_ast} "

                # CHECK UNDERS (Compare vs High Projection)
                if l_pts != 999 and proj_pts_high < (l_pts - 2.0):
                    val_add += (l_pts - proj_pts_high)
                    bet_str += f"PTS < {l_pts} "
                if l_reb != 999 and proj_reb_high < (l_reb - 1.5):
                    val_add += (l_reb - proj_reb_high)
                    bet_str += f"REB < {l_reb} "
                if l_ast != 999 and proj_ast_high < (l_ast - 1.5):
                    val_add += (l_ast - proj_ast_high)
                    bet_str += f"AST < {l_ast} "
                
                # Signal Generation
                signal = "-"
                if blowout_risk: signal = "⚠️ BLOWOUT" 
                elif p['PTS_VOLATILITY'] > 8.0: signal = "⚠️ VOLATILE"
                elif p['PTS'] + 1.2*p['REB'] + 1.5*p['AST'] > 45: signal = "ELITE"

                if val_add >= min_edge or show_all:
                    memo = generate_memo(val_add, signal)
                    
                    # Display logic (Show the projection that triggered the bet)
                    display_pts = proj_pts_high if "PTS <" in bet_str else proj_pts_low
                    display_reb = proj_reb_high if "REB <" in bet_str else proj_reb_low
                    display_ast = proj_ast_high if "AST <" in bet_str else proj_ast_low
                    
                    d_pts = f"{round(display_pts,1)} ({l_pts})" if l_pts!=999 else "-"
                    d_reb = f"{round(display_reb,1)} ({l_reb})" if l_reb!=999 else "-"
                    d_ast = f"{round(display_ast,1)} ({l_ast})" if l_ast!=999 else "-"
                    
                    audit_results.append({
                        "Date": today_str,
                        "Player": p['PLAYER_NAME'],
                        "Team": team_ctx.get(tid,{}).get('Name','UNK'),
                        "Signal": signal,
                        "Manager Memo": memo,
                        "Bet": bet_str,
                        "Edge": round(val_add, 1),
                        "PTS": d_pts,
                        "REB": d_reb,
                        "AST": d_ast
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
                    if item['Edge'] >= min_edge:
                        sheet.append_row([item['Date'], item['Player'], item['Team'], item['Bet'], item['Edge'], "PENDING"])
                st.success("✅ Updated Ledger!"); st.balloons()
            except Exception as e: st.error(f"Error: {e}")
        else: st.error("Sheet connection not active.")
else:
    # If schedule is empty (from Odds API), show this:
    if not market_schedule:
        st.warning("No Active Games found in the Betting Market. (Vegas is asleep).")
    else:
        st.info("No discrepancies found. Market is sharp today.")

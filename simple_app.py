import streamlit as st
import anthropic
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import re
import traceback
from datetime import datetime
import threading
import time
import json
import altair as alt
# Import the artist profiles module
from artist_profiles import render_artist_profiles_tab, get_all_artist_profiles
# Airtable integration removed

load_dotenv()
# Initialize Claude client only if API key is present
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    client = None

def load_data_file(file, data_source="Unknown"):
    """Load data from various file formats"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        else:
            return None, "Unsupported file format"
        
        # Add metadata about data source
        if 'data_source' not in df.columns:
            df['data_source'] = data_source
            
        return df, f"Successfully loaded {len(df)} rows from {data_source} data"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def export_dataframe_as_csv(df, filename_prefix="analysis_results"):
    """Create downloadable CSV from DataFrame"""
    csv_buffer = df.to_csv(index=False)
    return csv_buffer

# Initialize basic data loading
def load_all_data_files():
    """Load all available data files for cross-platform analysis"""
    all_dataframes = []
    
    # Check streaming data directory
    streaming_dir = "data/streaming_data"
    if os.path.exists(streaming_dir):
        for filename in os.listdir(streaming_dir):
            if filename.endswith(('.csv', '.xlsx', '.json')):
                file_path = f"{streaming_dir}/{filename}"
                try:
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif filename.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file_path)
                    elif filename.endswith('.json'):
                        df = pd.read_json(file_path)
                    else:
                        continue
                    
                    # Add data source metadata
                    if 'data_source' not in df.columns:
                        df['data_source'] = 'streaming_data'
                    if 'data_category' not in df.columns:
                        df['data_category'] = 'Streaming data'
                    
                    all_dataframes.append((df, 'streaming_data', filename))
                except Exception as e:
                    st.warning(f"Could not load {filename}: {str(e)}")
    
    # Check social media data directory
    social_dir = "data/social_media_data"
    if os.path.exists(social_dir):
        for filename in os.listdir(social_dir):
            if filename.endswith(('.csv', '.xlsx', '.json')):
                file_path = f"{social_dir}/{filename}"
                try:
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif filename.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file_path)
                    elif filename.endswith('.json'):
                        df = pd.read_json(file_path)
                    else:
                        continue
                    
                    # Add data source metadata
                    if 'data_source' not in df.columns:
                        df['data_source'] = 'social_media_data'
                    if 'data_category' not in df.columns:
                        df['data_category'] = 'Social media data'
                    
                    all_dataframes.append((df, 'social_media_data', filename))
                except Exception as e:
                    st.warning(f"Could not load {filename}: {str(e)}")
    
    # Check legacy data directory for backward compatibility
    if os.path.exists("data"):
        for filename in os.listdir("data"):
            if filename.endswith(('.csv', '.xlsx', '.json')) and not filename.startswith('.'):
                file_path = f"data/{filename}"
                try:
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif filename.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file_path)
                    elif filename.endswith('.json'):
                        df = pd.read_json(file_path)
                    else:
                        continue
                    
                    # Extract data source from filename
                    source_name = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
                    if 'data_source' not in df.columns:
                        df['data_source'] = source_name
                    if 'data_category' not in df.columns:
                        df['data_category'] = 'Legacy data'
                    
                    all_dataframes.append((df, source_name, filename))
                except Exception as e:
                    st.warning(f"Could not load {filename}: {str(e)}")
    
    return all_dataframes

def save_artist_profile(profile_data):
    """Save comprehensive artist profile to JSON file"""
    os.makedirs("data/artists_for_analytics", exist_ok=True)
    
    # Ensure required timestamp fields
    if 'created_date' not in profile_data:
        profile_data['created_date'] = datetime.now().isoformat()
    profile_data['last_updated'] = datetime.now().isoformat()
    
    safe_filename = profile_data['name'].lower().replace(" ", "_").replace("'", "")
    file_path = f"data/artists_for_analytics/{safe_filename}.json"
    
    with open(file_path, 'w') as f:
        json.dump(profile_data, f, indent=2)
    
    return f"✅ Artist profile saved: {profile_data['name']}"

def create_empty_profile_template(name="", age=25, gender="", genre=""):
    """Create an empty profile template with all required fields"""
    return {
        "name": name,
        "age": age,
        "gender": gender,
        "genre": genre,
        "descriptions": {
            "albums_songs_eps": "",
            "tiktok": "",
            "instagram": "",
            "spotify_performance": "",
            "other_info": ""
        },
        "social_links": {
            "tiktok": "",
            "instagram": "",
            "spotify": "",
            "youtube": ""
        },
        "created_date": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }

def load_artist_profile(name):
    """Load artist profile from JSON file"""
    safe_filename = name.lower().replace(" ", "_").replace("'", "")
    file_path = f"data/artists_for_analytics/{safe_filename}.json"
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                import json
                return json.load(f)
        except:
            return None
    return None

# Removed duplicate get_all_artist_profiles function
# Use the one from artist_profiles.py

def get_alta_music_artists():
    """Get list of Alta Music signed artists from profiles"""
    profiles = get_all_artist_profiles()
    return [profile['name'] for profile in profiles]

def get_signed_artists(df):
    """Extract list of signed artists from the data and artist profiles"""
    artists_from_data = []
    
    # Get artists from uploaded data
    if df is not None:
        artist_columns = []
        potential_cols = ['artist', 'artist_name', 'name', 'creator', 'channel', 'account']
        
        for col in df.columns:
            if any(term in col.lower() for term in potential_cols):
                artist_columns.append(col)
        
        if artist_columns:
            all_artists = set()
            for col in artist_columns:
                if col in df.columns:
                    try:
                        artists = df[col].dropna().astype(str).unique()
                        all_artists.update(artists)
                    except Exception:
                        continue
            
            # Filter out empty or 'nan' strings
            all_artists = {artist for artist in all_artists if artist and artist.lower() != 'nan'}
            artists_from_data = list(all_artists)
    
    # Get artists from profiles
    profile_artists = get_alta_music_artists()
    
    # Combine and deduplicate
    return sorted(list(all_artists))

# Chat history persistence functions
def save_chat_history(history):
    """Save chat history to a persistent file"""
    try:
        os.makedirs("data", exist_ok=True)
        chat_file = "data/alta_music_chat_history.json"
        
        # Load existing history if it exists
        existing_history = []
        if os.path.exists(chat_file):
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    existing_history = json.load(f)
            except:
                existing_history = []
        
        # Append new messages to existing history
        if isinstance(history, list):
            for msg in history:
                if msg not in existing_history:  # Avoid duplicates
                    existing_history.append({
                        **msg,
                        "timestamp": datetime.now().isoformat(),
                        "session_id": st.session_state.get('session_id', 'default')
                    })
        
        # Keep only last 100 messages to prevent file from getting too large
        if len(existing_history) > 100:
            existing_history = existing_history[-100:]
        
        # Save updated history
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(existing_history, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        st.warning(f"Could not save chat history: {str(e)}")

def load_chat_history():
    """Load persistent chat history"""
    try:
        chat_file = "data/alta_music_chat_history.json"
        if os.path.exists(chat_file):
            with open(chat_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                return history if isinstance(history, list) else []
        return []
    except Exception as e:
        st.warning(f"Could not load chat history: {str(e)}")
        return []

def get_relevant_chat_context(current_query, max_messages=10):
    """Get relevant chat history context for the current query"""
    full_history = load_chat_history()
    if not full_history:
        return ""
    
    # Get recent messages (last 10 by default)
    recent_history = full_history[-max_messages:] if len(full_history) > max_messages else full_history
    
    # Format for context
    context_parts = []
    for msg in recent_history:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')[:200]  # Truncate long messages
        timestamp = msg.get('timestamp', '')
        
        if role == 'user':
            context_parts.append(f"User ({timestamp[:10]}): {content}")
        elif role == 'assistant':
            context_parts.append(f"Claude ({timestamp[:10]}): {content}")
    
    if context_parts:
        return f"\n**RECENT ALTA MUSIC CONVERSATIONS**:\n" + "\n".join(context_parts) + "\n"
    return ""

def save_data_description(filename, category, description):
    """Save data description for a uploaded file"""
    os.makedirs("data/descriptions", exist_ok=True)
    
    description_data = {
        "filename": filename,
        "category": category,
        "description": description,
        "upload_date": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }
    
    safe_filename = filename.replace(".", "_").replace(" ", "_")
    desc_file_path = f"data/descriptions/{safe_filename}_description.json"
    
    with open(desc_file_path, 'w') as f:
        json.dump(description_data, f, indent=2)
    
    return f"✅ Data description saved for: {filename}"

def load_data_description(filename):
    """Load data description for a specific file"""
    safe_filename = filename.replace(".", "_").replace(" ", "_")
    desc_file_path = f"data/descriptions/{safe_filename}_description.json"
    
    if os.path.exists(desc_file_path):
        try:
            with open(desc_file_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def get_all_data_descriptions():
    """Get all data descriptions for Claude context"""
    descriptions = []
    desc_dir = "data/descriptions"
    
    if os.path.exists(desc_dir):
        for filename in os.listdir(desc_dir):
            if filename.endswith('_description.json'):
                try:
                    with open(f"{desc_dir}/{filename}", 'r') as f:
                        description = json.load(f)
                        descriptions.append(description)
                except:
                    continue
    
    return descriptions

def format_data_context_for_claude():
    """Format all data descriptions for Claude context"""
    descriptions = get_all_data_descriptions()
    if not descriptions:
        return ""
    
    context_parts = ["\n**UPLOADED DATA CONTEXT**:"]
    for desc in descriptions:
        context_parts.append(f"- **{desc['filename']}** ({desc['category']}): {desc['description']}")
    
    return "\n".join(context_parts) + "\n"

# Display logo at the top of the app
st.image('ChatGPT.png', width=400)

# Test comment for auto-reload

# Initialize session ID for chat history tracking
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Simplified data summary
existing_data_files = []
existing_case_studies = []

# Check for categorized data files
data_directories = ["data/streaming_data", "data/social_media_data", "data/artists_for_analytics"]
for data_dir in data_directories:
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.xlsx', '.xls', '.json')) and not file.startswith('.'):
                existing_data_files.append(f"{data_dir}/{file}")

# Check legacy data directory (main data folder)
if os.path.exists("data"):
    for file in os.listdir("data"):
        if file.endswith(('.csv', '.xlsx', '.xls', '.json')) and not file.startswith('.') and not os.path.isdir(f"data/{file}"):
            existing_data_files.append(f"data/{file}")

# Check for case studies
if os.path.exists("data/case_studies"):
    existing_case_studies = os.listdir("data/case_studies")

# Load data automatically if available
df = None
if existing_data_files:
    # Auto-load all data files and combine them
    all_data = []
    total_size_mb = 0
    for file_path in existing_data_files:
        try:
            # Get file size
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            total_size_mb += file_size
            
            filename = os.path.basename(file_path)
            
            if filename.endswith('.csv'):
                temp_df = pd.read_csv(file_path)
            elif filename.endswith(('.xlsx', '.xls')):
                temp_df = pd.read_excel(file_path)
            elif filename.endswith('.json'):
                temp_df = pd.read_json(file_path)
            else:
                continue
            
            # Add data source and category from file path
            if 'data_source' not in temp_df.columns:
                if 'streaming_data' in file_path:
                    temp_df['data_source'] = 'streaming_data'
                    temp_df['data_category'] = 'Streaming data'
                elif 'social_media_data' in file_path:
                    temp_df['data_source'] = 'social_media_data'
                    temp_df['data_category'] = 'Social media data'
                else:
                    source_name = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
                    temp_df['data_source'] = source_name
                    temp_df['data_category'] = 'Legacy data'
            
            all_data.append(temp_df)
        except Exception as e:
            st.warning(f"Could not load {filename}: {str(e)}")
    
    if all_data:
        # Combine all dataframes
        df = pd.concat(all_data, ignore_index=True, sort=False)
        
        # Simplified one-line summary
        case_studies_text = f" + {len(existing_case_studies)} case studies" if existing_case_studies else ""
        st.info(f"📊 **Data loaded:** {total_size_mb:.1f}MB across {len(existing_data_files)} files ({len(df):,} total rows){case_studies_text}")

elif not existing_data_files and not existing_case_studies:
    st.warning("⚠️ No data files found. Please upload data files in the Upload tab to get started.")
    st.stop()

# ---
# ALTA MUSIC ANALYTICS SECTIONS
# ---

# Get the main dataset (currently the loaded data file)
main_df = df
signed_artists = get_signed_artists(main_df) if main_df is not None else []

# Create tabs for different analysis types
tab1, tab2, tab3 = st.tabs(["🚀 Campaign Insights", "👤 Artist Profiles", "📁 Upload Files"])

# Shared Claude functions

def execute_code_block(code):
    """Executes a code block and returns output or error."""
    try:
        # Provide access to all processed data and helper functions
        exec(code, {
            "df": main_df,  # Main processed dataset
            "raw_df": df,   # Original single file dataset  
            "main_df": main_df,  # Alias for clarity
            "signed_artists": signed_artists,  # List of detected artists
            "pd": pd, 
            "st": st, 
            "os": os,
            "datetime": datetime,
            "export_dataframe_as_csv": export_dataframe_as_csv,
            "read_case_study_file": read_case_study_file,
            "get_signed_artists": get_signed_artists,
            "load_all_data_files": load_all_data_files,
            "load_artist_profile": load_artist_profile,
            "get_all_artist_profiles": get_all_artist_profiles,
            "get_alta_music_artists": get_alta_music_artists
        })
        return "✅ Code executed successfully.", None
    except Exception as e:
        tb = traceback.format_exc()
        return f"❌ Error executing code:\n{tb}", e

def extract_code_blocks(text):
    """Extract and execute Python code blocks"""
    # Handle None input
    if text is None:
        return ""
    
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    for code in code_blocks:
        msg, err = execute_code_block(code.strip())
        if err:
            st.error(msg)
        else:
            st.success(msg)
    
    # Return text without code blocks
    return re.sub(r"```python.*?```", "", text, flags=re.DOTALL).strip()

def read_case_study_file(filename):
    """Helper function to read case study files"""
    file_path = f"data/case_studies/{filename}"
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            try:
                # Try with different encoding if UTF-8 fails
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except:
                return f"Could not read {filename} - unsupported file format or encoding"
    return f"File {filename} not found"

def call_claude(messages, context_type="general"):
    """Call Claude API with context-aware system message"""
    try:
        # Simple system message for general queries
        system_message = f"""You are a knowledgeable music industry consultant working with Alta Music Group. You have access to comprehensive data about our artists and can provide insights on music industry trends, artist development, and strategic planning.

**AVAILABLE DATA:**
- Artist profiles and demographics
- Streaming and social media performance data
- Industry benchmarks and case studies

**GUIDELINES:**
- Provide helpful, accurate information about music industry topics
- Use available data to support your responses when relevant
- Offer practical advice for artist development and music marketing
- Stay current with industry trends and best practices

Please provide helpful and informative responses based on the available data and your music industry expertise."""

        # Convert messages to proper format if needed
        if isinstance(messages, list) and len(messages) > 0:
            # Handle chat history format
            if isinstance(messages[0], dict) and 'content' in messages[0]:
                # Extract just the user messages, ignoring system messages
                user_messages = [msg for msg in messages if msg.get('role') == 'user']
                if user_messages:
                    user_message = user_messages[-1]['content']  # Get the last user message
                else:
                    user_message = "Please provide analysis of the available data."
            else:
                # It's a list of strings
                user_message = str(messages[-1])
        else:
            # Single message
            user_message = str(messages) if messages else "Please provide analysis of the available data."
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": f"{system_message}\n\nUser request: {user_message}"}
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        st.error(f"❌ Error calling Claude API: {str(e)}")
        return f"Error: Unable to get response from Claude. Please check your API key and try again.\n\nError details: {str(e)}"

# --- Utility: Load TikTok dataset for campaign chooser ---
def load_tiktok_dataset():
    tiktok_file = "data/social_media_data/dataset_tiktok-profile-scraper_2025-07-10_00-00-34-357.csv"
    if os.path.exists(tiktok_file):
        try:
            df = pd.read_csv(tiktok_file)
            df = df.dropna(subset=['authorMeta/name'])
            df['followers'] = pd.to_numeric(df.get('authorMeta/fans', 0), errors='coerce').fillna(0)
            df['total_likes'] = pd.to_numeric(df.get('authorMeta/heart', 0), errors='coerce').fillna(0)
            df['verified'] = df.get('authorMeta/verified', False).astype(bool)
            return df
        except Exception as e:
            st.error(f"Error loading TikTok dataset: {str(e)}")
            return None
    else:
        st.error("TikTok dataset not found. Please ensure the file exists in data/social_media_data/")
        return None

# TAB 1: ARTIST CAMPAIGN (SIMPLIFIED)
with tab1:
        st.header("🎶 TikTok insights")
        

        # TikTok Correlation Pie Charts
        st.subheader("Latest correlation to streams")
        import plotly.graph_objects as go
        # Correlation coefficients and colors
        corrs = {
            "Creates": 0.25,
            "Likes": 0.4,
            "Views": 0.35
        }
        colors = {
            "Creates": "#c82613",
            "Likes": "#2b6f39",
            "Views": "#c82613"
        }
        # Create three donut charts
        figs = []
        for label, value in corrs.items():
            fig = go.Figure(data=[go.Pie(
                values=[value, 1-value],
                labels=[label, ""],
                marker_colors=[colors[label], "lightgrey"],
                hole=0.6,
                textinfo='none',
                sort=False
            )])
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                width=200,
                height=200,
                annotations=[dict(
                    text=f"<b>{value:.2f}</b>",
                    x=0.5, y=0.5, font_size=28, showarrow=False, font_color=colors[label],
                    xanchor='center', yanchor='middle', align='center'
                )]
            )
            figs.append(fig)
        # Display titles and charts horizontally
        cols = st.columns(3)
        for i, (label, fig) in enumerate(zip(corrs.keys(), figs)):
            with cols[i]:
                st.markdown(f"<div style='text-align:center;font-weight:bold'>{label}</div>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
        # Caption below the chart
        st.markdown("*Dataset (08/07/25): Top 500 tracks on Spotify by weekly growth rate")
        
        # --- LATEST TRENDS SECTION ---
        st.subheader('Latest trends')
        # Show a loading spinner while trends are being generated
        if 'tiktok_trends_summary' not in st.session_state:
            with st.spinner('Loading latest TikTok trends...'):
                @st.cache_data
                def load_tiktok_trends_dataset():
                    tiktok_file = "data/social_media_data/dataset_tiktok-profile-scraper_2025-07-10_00-00-34-357.csv"
                    if os.path.exists(tiktok_file):
                        df = pd.read_csv(tiktok_file)
                        return df
                    else:
                        return None
                tiktok_trends_df = load_tiktok_trends_dataset()
                if tiktok_trends_df is not None:
                    sample_rows = tiktok_trends_df.head(100).to_dict(orient='records')
                    summary_prompt = f"""
At the top of the response, write: Based on  TikTok data from the top 500 songs on Spotify over the past week
Analyze the following TikTok dataset (each dict is a video/profile row) and summarize the latest 5 trends you see in TikTok music content, focusing on content type, music type, and any notable patterns. Be concise and specific. Output as 5 bullet points.
We want to know the content trends so that we can design content and choose TikTok accounts for our arists. 
I need one specific bullet on what influencers are doing with regard to content (i.e. dance challenges, lip-syncs, etc.)

DATA SAMPLE:
{json.dumps(sample_rows)[:8000]}  # Truncate to avoid token overflow
"""
                    try:
                        api_key = os.getenv("ANTHROPIC_API_KEY")
                        if not api_key or client is None:
                            st.session_state['tiktok_trends_summary'] = "Error: Claude API key is missing. Please set ANTHROPIC_API_KEY in your .env file."
                        else:
                            response = client.messages.create(
                                model="claude-3-5-sonnet-20241022",
                                max_tokens=400,
                                messages=[{"role": "user", "content": summary_prompt}]
                            )
                            st.session_state['tiktok_trends_summary'] = response.content[0].text.strip()
                    except Exception as e:
                        st.session_state['tiktok_trends_summary'] = f"Error generating TikTok trends summary: {str(e)}"
                else:
                    st.session_state['tiktok_trends_summary'] = "TikTok dataset not found or could not be loaded for trend analysis."
        st.markdown(st.session_state['tiktok_trends_summary'])
        # --- END LATEST TRENDS SECTION ---
        
        st.markdown("---")  # Add separator
        
        # TikTok Campaign Chooser Section
        st.header("🎯 Campaign Insights")
        st.write("*Select an artist and let AltAI help with TikTok campaign insights*")
        alta_artists = get_alta_music_artists()
        # Only run the TikTok Campaign Chooser UI if there are artists
        if alta_artists:
            # Use a stable key for artist selection to avoid unnecessary reruns
            selected_campaign_artist = st.selectbox(
                "",
                ["Select an artist..."] + alta_artists,
                key="campaign_artist_selector"
            )
            # Only show the rest of the UI if an artist is selected
            if selected_campaign_artist != "Select an artist...":
                # --- TikTok Campaign Chat-like Flow ---
                # System prompt for all messages
                system_prompt = f"""
You are a knowledgeable music industry consultant working with Alta Music Group. You have access to comprehensive data about our artists and can provide insights on music industry trends, artist development, and strategic planning.

For TikTok campaign recommendations, always consider the following guidance:
- Recommend five different TikTok influencer accounts (not artists) for campaign content placement, based on recent performance, trend alignment, and fit to the artist genre and vibe.
- Please be mindful to recommend TikTok accounts who you actually think we can place our content with, taking into account the current popularity of our artist, the fit and potential cost.
- Present recommendations in a clear, structured format, with:
  - Recent performance
  - Profile overview
  - Reason for fit (must reference latest trends and the importance of likes)
- If the user asks a different campaign question, answer as appropriate, but always keep the influencer recommendation guidance in mind.
"""
                # Initialize chat history in session state
                chat_key = f"campaign_chat_history_{selected_campaign_artist}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                # Step 1: Generate button acts as first message
                generate_clicked = st.button("🚀 Generate TikTok Campaign Insights", key="generate_tiktok_campaign", use_container_width=True)
                if generate_clicked and not st.session_state[chat_key]:
                    # Build initial context and message
                    import glob
                    artist_dir = f"data/artists_for_analytics/{selected_campaign_artist.lower().replace(' ', '_')}"
                    artist_data = {}
                    for fjson in glob.glob(os.path.join(artist_dir, "*.json")):
                        try:
                            with open(fjson, "r", encoding="utf-8") as f:
                                try:
                                    artist_data[os.path.basename(fjson).replace(".json", "")] = json.load(f)
                                except Exception:
                                    f.seek(0)
                                    artist_data[os.path.basename(fjson).replace(".json", "")] = pd.read_json(fjson)
                        except Exception as e:
                            st.warning(f"Could not load artist JSON {fjson}: {str(e)}")
                    for fcsv in glob.glob(os.path.join(artist_dir, "*.csv")):
                        try:
                            artist_data[os.path.basename(fcsv).replace(".csv", "")] = pd.read_csv(fcsv)
                        except Exception as e:
                            st.warning(f"Could not load artist CSV {fcsv}: {str(e)}")
                    scraped_dir = os.path.join(artist_dir, "scraped_data")
                    if os.path.exists(scraped_dir):
                        for fjson in glob.glob(os.path.join(scraped_dir, "*.json")):
                            try:
                                with open(fjson, "r", encoding="utf-8") as f:
                                    try:
                                        artist_data[os.path.basename(fjson).replace(".json", "")] = json.load(f)
                                    except Exception:
                                        f.seek(0)
                                        artist_data[os.path.basename(fjson).replace(".json", "")] = pd.read_json(fjson)
                            except Exception as e:
                                st.warning(f"Could not load artist scraped JSON {fjson}: {str(e)}")
                        for fcsv in glob.glob(os.path.join(scraped_dir, "*.csv")):
                            try:
                                artist_data[os.path.basename(fcsv).replace(".csv", "")] = pd.read_csv(fcsv)
                            except Exception as e:
                                st.warning(f"Could not load artist scraped CSV {fcsv}: {str(e)}")
                    influencer_dfs = []
                    for data_dir in ["data/social_media_data", "data/streaming_data"]:
                        if os.path.exists(data_dir):
                            for file in os.listdir(data_dir):
                                fpath = os.path.join(data_dir, file)
                                if file.endswith('.csv'):
                                    try:
                                        influencer_dfs.append(pd.read_csv(fpath))
                                    except Exception as e:
                                        st.warning(f"Could not load {fpath}: {str(e)}")
                                elif file.endswith('.json'):
                                    try:
                                        influencer_dfs.append(pd.read_json(fpath))
                                    except Exception as e:
                                        st.warning(f"Could not load {fpath}: {str(e)}")
                    tiktok_df = load_tiktok_dataset() if os.path.exists('data/social_media_data/dataset_tiktok-profile-scraper_2025-07-10_00-00-34-357.csv') else (pd.concat(influencer_dfs, ignore_index=True) if influencer_dfs else None)
                    if tiktok_df is not None:
                        artist_handles = set()
                        for k, v in artist_data.items():
                            if isinstance(v, dict) and 'handle' in v:
                                artist_handles.add(v['handle'].lower())
                        tiktok_df = tiktok_df[~tiktok_df['authorMeta/name'].str.lower().isin(artist_handles)]
                        music_keywords = ["music", "song", "artist", "band", "singer", "producer", "indie", "pop", "folk", "songwriter"]
                        tiktok_df['bio_lower'] = tiktok_df['authorMeta/signature'].fillna('').str.lower()
                        tiktok_df['music_relevance'] = tiktok_df['bio_lower'].apply(lambda x: any(kw in x for kw in music_keywords))
                        tiktok_df['score'] = (
                            tiktok_df['followers'].rank(pct=True) * 0.4 +
                            tiktok_df['total_likes'].rank(pct=True) * 0.2 +
                            tiktok_df['music_relevance'].astype(int) * 0.2 +
                            tiktok_df['verified'].astype(int) * 0.1 +
                            np.random.rand(len(tiktok_df)) * 0.1
                        )
                        if 'createTimeISO' in tiktok_df.columns:
                            tiktok_df['recent_post'] = pd.to_datetime(tiktok_df['createTimeISO'], errors='coerce')
                            recent_cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=60)
                            if tiktok_df['recent_post'].dt.tz is None:
                                tiktok_df['recent_post'] = tiktok_df['recent_post'].dt.tz_localize('UTC')
                            tiktok_df['active_recently'] = tiktok_df['recent_post'] > recent_cutoff
                            tiktok_df['score'] += tiktok_df['active_recently'].astype(int) * 0.1
                        top5 = tiktok_df.drop_duplicates(subset=['authorMeta/name']).sort_values('score', ascending=False).head(5)
                        trends_summary = st.session_state.get('tiktok_trends_summary', None)
                        if not trends_summary:
                            trends_summary = ''
                        recommendations_str = "\n".join([
                            f"{idx+1}. {row['authorMeta/nickName']} (@{row['authorMeta/name']}): {int(row['followers']):,} followers, {int(row['total_likes']):,} likes. Bio: {row['authorMeta/signature']}" for idx, (_, row) in enumerate(top5.iterrows())
                        ])
                        initial_user_message = f"Please recommend the top five TikTok accounts for campaign content placement for {selected_campaign_artist}, based on recent performance, trend alignment, and fit to the artist."
                        context_for_claude = f"""
ARTIST: {selected_campaign_artist}

RECOMMENDATIONS:
{recommendations_str}

TRENDS: {trends_summary}

LIKES are the most important metric for streaming growth.

RESPONSE FORMAT: List five different TikTok influencer recommendations (do not include artists), each as a bold heading, with:
- Recent performance
- Profile overview
- Reason for fit (must reference latest trends and the importance of likes)

Start with a very short bullet or two on the general TikTok influencer marketing approach for the artist and be incredibly specific.
"""
                        with st.spinner("Claude is analyzing and preparing your campaign insights..."):
                            try:
                                api_key = os.getenv("ANTHROPIC_API_KEY")
                                if not api_key or client is None:
                                    st.session_state[chat_key].append({"role": "assistant", "content": "Error: Claude API key is missing. Please set ANTHROPIC_API_KEY in your .env file."})
                                else:
                                    claude_response = client.messages.create(
                                        model="claude-3-5-sonnet-20241022",
                                        max_tokens=600,
                                        messages=[{"role": "user", "content": system_prompt + "\n" + context_for_claude}]
                                    )
                                    st.session_state[chat_key].append({"role": "user", "content": initial_user_message})
                                    st.session_state[chat_key].append({"role": "assistant", "content": claude_response.content[0].text.strip()})
                            except Exception as e:
                                st.session_state[chat_key].append({"role": "assistant", "content": f"Error getting Claude response: {str(e)}"})
                # Step 2: Display chat history (insights + follow-ups)
                for msg in st.session_state[chat_key]:
                    if msg['role'] == 'user':
                        st.markdown(f"**You:** {msg['content']}")
                    else:
                        st.markdown(f"**Claude:** {msg['content']}")
                # Step 3: Input box and Ask Claude button only after initial insights (i.e., after generate is hit)
                if st.session_state[chat_key]:
                    user_claude_query = st.text_area(
                        "Ask a follow-up question...",
                        placeholder="e.g., Can you focus on TikTok influencers with more Gen Z followers? Or, suggest a general TikTok campaign approach.",
                        height=100,
                        key=f"claude_refine_query_{selected_campaign_artist}"
                    )
                    ask_button = st.button("Ask Claude (sometimes have to click again / wait)", key=f"claude_refine_button_{selected_campaign_artist}", use_container_width=True)
                    if ask_button and user_claude_query.strip():
                        with st.spinner("Claude is analyzing your question..."):
                            try:
                                api_key = os.getenv("ANTHROPIC_API_KEY")
                                if not api_key or client is None:
                                    st.session_state[chat_key].append({"role": "assistant", "content": "Error: Claude API key is missing. Please set ANTHROPIC_API_KEY in your .env file."})
                                else:
                                    messages = [
                                        {"role": "user", "content": system_prompt}
                                    ]
                                    for m in st.session_state[chat_key]:
                                        messages.append({"role": m['role'], "content": m['content']})
                                    messages.append({"role": "user", "content": user_claude_query})
                                    claude_response = client.messages.create(
                                        model="claude-3-5-sonnet-20241022",
                                        max_tokens=600,
                                        messages=messages
                                    )
                                    st.session_state[chat_key].append({"role": "user", "content": user_claude_query})
                                    st.session_state[chat_key].append({"role": "assistant", "content": claude_response.content[0].text.strip()})
                            except Exception as e:
                                st.session_state[chat_key].append({"role": "assistant", "content": f"Error getting Claude response: {str(e)}"})
# TAB 2: ARTIST PROFILES
with tab2:
    render_artist_profiles_tab()

# TAB 3: UPLOAD FILES
with tab3:
    st.header("📁 Upload Data Files")
    st.write("*Upload any file to train AltAI (streaming data, social media insights, case studies, articles and shower thoughts)*")
    
    # Always visible data type selector
    data_type = st.selectbox(
        "What type of data are you uploading?",
        ["Streaming Data", "Social Media Data", "Case Studies", "Other"],
        key="upload_data_type",
        help="Select the category that best describes your data"
    )
    
    # Always visible description field
    description = st.text_area(
        "Description",
        placeholder="Describe in detail what this data contains",
        key="upload_description",
        height=100,
        help="This description helps Claude understand and use your data effectively"
    )
    
    # File uploader section
    uploaded_files = st.file_uploader(
        "Choose data files to upload",
        type=['csv', 'xlsx', 'xls', 'json'],
        accept_multiple_files=True,
        key="file_upload",
        help="Supported formats: CSV, Excel (.xlsx), JSON"
    )
    
    # Process uploaded files
    if uploaded_files and description.strip():
        for file in uploaded_files:
            st.divider()
            st.subheader(f"📄 Processing: {file.name}")
            
            # Show file info
            file_size = len(file.getbuffer()) / 1024  # KB
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{file_size:.1f} KB")
            with col2:
                st.metric("Type", data_type)
            with col3:
                st.metric("Status", "Ready to Save")
            
            # Save button
            if st.button(f"💾 Save {file.name}", key=f"save_{file.name}", type="primary", use_container_width=True):
                # Determine save location based on data type
                if data_type == "Streaming Data":
                    os.makedirs("data/streaming_data", exist_ok=True)
                    file_path = f"data/streaming_data/{file.name}"
                elif data_type == "Social Media Data":
                    os.makedirs("data/social_media_data", exist_ok=True)
                    file_path = f"data/social_media_data/{file.name}"
                elif data_type == "Case Studies":
                    os.makedirs("data/case_studies", exist_ok=True)
                    file_path = f"data/case_studies/{file.name}"
                else:  # Other
                    os.makedirs("data", exist_ok=True)
                    file_path = f"data/{file.name}"
                
                # Save the file
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Save description
                desc_result = save_data_description(file.name, data_type, description)
                
                # Validate data file
                if file.name.endswith(('.csv', '.xlsx', '.xls', '.json')):
                    result = load_data_file(file, data_type)
                    if result is not None and isinstance(result, tuple):
                        df_temp, message = result
                        if df_temp is not None:
                            st.success(f"✅ {file.name} saved successfully!")
                            st.success(desc_result)
                            # Show data preview
                            with st.expander(f"Preview: {file.name}"):
                                st.dataframe(df_temp.head())
                                st.write(f"**Columns:** {', '.join(df_temp.columns.tolist())}")
                                st.write(f"**Rows:** {len(df_temp):,}")
                        else:
                            st.error(f"❌ Error loading {file.name}: {message if message else 'Unknown error'}")
                    else:
                        st.error(f"❌ Error loading {file.name}: Unknown error")
                else:
                    st.success(f"✅ {file.name} saved successfully!")
                    st.success(desc_result)
    
    elif uploaded_files and not description.strip():
        st.warning("⚠️ Please provide a description before uploading files.")

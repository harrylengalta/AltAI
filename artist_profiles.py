import streamlit as st
import openai
import os
import json
import traceback
from datetime import datetime
import random
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client only if API key is present
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    openai.api_key = None

def simulate_tiktok_scrape(handle):
    """Simulate TikTok scraping (in real implementation, this would use actual APIs/scrapers)"""
    try:
        # Clean handle
        if handle.startswith('http'):
            handle = handle.split('/')[-1]
        handle = handle.replace('@', '')
        
        # Simulate realistic TikTok data
        followers = random.randint(1000, 500000)
        following = random.randint(50, 2000)
        total_likes = random.randint(followers * 10, followers * 50)
        video_count = random.randint(20, 300)
        
        return {
            "handle": handle,
            "followers": followers,
            "following": following,
            "total_likes": total_likes,
            "video_count": video_count,
            "bio": f"🎵 Music artist | Follow for daily content | Collab: contact@{handle}.com",
            "verified": random.choice([True, False]),
            "scraped_at": datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Error scraping TikTok data: {str(e)}")
        return None

def simulate_instagram_scrape(handle):
    """Simulate Instagram scraping (in real implementation, this would use actual APIs/scrapers)"""
    try:
        # Clean handle
        if handle.startswith('http'):
            handle = handle.split('/')[-1]
        handle = handle.replace('@', '')
        
        # Simulate realistic Instagram data
        followers = random.randint(500, 100000)
        following = random.randint(100, 1500)
        posts = random.randint(50, 500)
        
        return {
            "handle": handle,
            "followers": followers,
            "following": following,
            "posts": posts,
            "bio": f"🎤 Singer/Songwriter | 📍 Based in LA | 🎵 New music out now!",
            "verified": random.choice([True, False]),
            "scraped_at": datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Error scraping Instagram data: {str(e)}")
        return None

def generate_unified_profile(artist_name, artist_description, collected_data):
    """Use GPT to generate a detailed, factual profile and short platform summaries (no Spotify)"""
    try:
        # Prepare data summary for GPT
        data_summary = f"Artist: {artist_name}\n\n"
        if 'tiktok' in collected_data:
            tiktok = collected_data['tiktok']
            data_summary += f"TikTok Data:\n- Handle: @{tiktok['handle']}\n- Followers: {tiktok['followers']:,}\n- Bio: {tiktok['bio']}\n\n"
        if 'instagram' in collected_data:
            instagram = collected_data['instagram']
            data_summary += f"Instagram Data:\n- Handle: @{instagram['handle']}\n- Followers: {instagram['followers']:,}\n- Bio: {instagram['bio']}\n\n"
        prompt = f'''
        Write a JSON file for the artist {artist_name} including the following elements:
        - name
        - genre (as a single, human-readable string, e.g. "Rock, Blues Rock, Hard Rock")
        - description (see below)
        - vibe
        - tiktok_summary
        - instagram_summary
        - overall_score
        - growth_potential
        
        Guidance for the elements:
        1. Use the following artist description to help find the artist and inform the description: {artist_description}
        2. For the description field, write a long, richly detailed, historical + forward looking, factual and engaging summary of the artist (at least 10-12 sentences). Include their musical characteristics, influences, background, creative process, genre, vibe, personality, career journey, and any unique qualities or trends in their style. Avoid hype, focus on facts, nuance, and atmosphere. Paint a vivid, immersive picture of the artist as a person and creator, with plenty of specifics and context.
        3. For the genre field, output a single, human-readable string (not a list or array).
        4. For tiktok_summary and instagram_summary, use the following data summary to inform concise, factual summaries of the artist's presence and performance on each platform:
        {data_summary}
        5. For overall_score and growth_potential, provide a brief, factual assessment based on the data provided.
        
        Output JSON with keys: name, genre, description, vibe, tiktok_summary, instagram_summary, overall_score, growth_potential.
        The 'description' field in the JSON should be a long, richly detailed and atmospheric summary as described above.
        For the genre field, output a single, human-readable string (e.g. "Rock, Blues Rock, Hard Rock") rather than a list or array.
        '''
        if not OPENAI_API_KEY:
            st.error("Error: OpenAI API key is missing. Please set OPENAI_API_KEY in your .env file or Streamlit secrets.")
            return None
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a music industry expert who creates detailed artist profiles. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=900,
            temperature=0.5
        )
        text = response.choices[0].message["content"].strip()
        # Try to extract JSON from the response robustly
        import json, re
        try:
            # Try direct JSON parse first
            profile = json.loads(text)
        except Exception:
            # Try to extract the first JSON object from the text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    profile = json.loads(match.group(0))
                except Exception as e2:
                    st.error(f"Error parsing profile JSON: {str(e2)}\nRaw output: {text[:1000]}...")
                    return None
            else:
                st.error(f"Error: No valid JSON found in GPT output. Raw output: {text[:1000]}...")
                return None
        return profile
    except Exception as e:
        st.error(f"Error generating profile: {str(e)}")
        return None

def save_artist_profile(profile_data, scraped_data=None):
    """Save comprehensive artist profile to JSON file in the artist's dedicated folder only, and create platform summary files."""
    # Ensure required timestamp fields
    if 'created_date' not in profile_data:
        profile_data['created_date'] = datetime.now().isoformat()
    profile_data['last_updated'] = datetime.now().isoformat()

    # Save only in the artist's dedicated folder as profile_summary.json
    safe_folder = profile_data['name'].lower().replace(" ", "_").replace("'", "")
    artist_dir = f"data/artists_for_analytics/{safe_folder}"
    os.makedirs(artist_dir, exist_ok=True)
    file_path = os.path.join(artist_dir, "profile_summary.json")
    with open(file_path, 'w') as f:
        json.dump(profile_data, f, indent=2)

    # Save TikTok and Instagram platform summary files if data is available
    if scraped_data:
        # TikTok
        if 'tiktok' in scraped_data:
            tiktok_summary = {
                "summary": profile_data.get("tiktok_summary", "No summary available."),
                "followers": scraped_data['tiktok'].get("followers", "N/A"),
                "following": scraped_data['tiktok'].get("following", "N/A"),
                "total_likes": scraped_data['tiktok'].get("total_likes", "N/A"),
                "video_count": scraped_data['tiktok'].get("video_count", "N/A"),
                "bio": scraped_data['tiktok'].get("bio", "N/A")
            }
            with open(os.path.join(artist_dir, "tiktok_profile.json"), 'w') as f:
                json.dump(tiktok_summary, f, indent=2)
        # Instagram
        if 'instagram' in scraped_data:
            instagram_summary = {
                "summary": profile_data.get("instagram_summary", "No summary available."),
                "followers": scraped_data['instagram'].get("followers", "N/A"),
                "following": scraped_data['instagram'].get("following", "N/A"),
                "posts": scraped_data['instagram'].get("posts", "N/A"),
                "bio": scraped_data['instagram'].get("bio", "N/A")
            }
            with open(os.path.join(artist_dir, "instagram_profile.json"), 'w') as f:
                json.dump(instagram_summary, f, indent=2)
    return f"✅ Artist profile saved: {profile_data['name']}"

def get_all_artist_profiles():
    """Get list of all saved artist profiles (excluding working directory)"""
    profiles = []
    profile_dir = "data/artists_for_analytics"
    if os.path.exists(profile_dir):
        for entry in os.listdir(profile_dir):
            entry_path = os.path.join(profile_dir, entry)
            if os.path.isdir(entry_path) and entry != "working":
                profile_path = os.path.join(entry_path, "profile_summary.json")
                if os.path.exists(profile_path):
                    try:
                        with open(profile_path, 'r') as f:
                            profile = json.load(f)
                            profiles.append(profile)
                    except:
                        continue
    return profiles

def render_artist_profiles_tab():
    """Render the Artist Profiles tab"""
    st.header("👤 Artist Profiles")
    st.write("*Create comprehensive artist profiles with automated data collection and Claude analysis*")
    
    # Single unified form
    st.subheader("🚀 Create Artist Profile")
    st.write("*Enter artist details and social handles - our system will scrape data and generate a comprehensive profile*")
    
    # Profile creation form
    with st.form("create_profile_form"):
        artist_name = st.text_input("Artist Name*", placeholder="e.g., Cat Matthews")
        artist_description = st.text_area(
            "Brief Description*", 
            placeholder="e.g., Rising female singer songwriter from Los Angeles...",
            height=100,
            help="Provide any context about the artist that will help Claude generate an accurate profile"
        )
        
        # Social media handles
        col1, col2 = st.columns(2)
        with col1:
            tiktok_handle = st.text_input("TikTok Handle", 
                                         placeholder="@username or full URL",
                                         help="TikTok username or URL")
        with col2:
            instagram_handle = st.text_input("Instagram Handle", 
                                           placeholder="@username or full URL",
                                           help="Instagram username or URL")
        
        submit_profile = st.form_submit_button("🚀 Create Profile", type="primary")
        
        if submit_profile and artist_name and artist_description:
            try:
                # Use a working directory for unsaved profiles
                working_dir = "data/artists_for_analytics/working"
                scraped_data_dir = os.path.join(working_dir, "scraped_data")
                # Clear the working directory if it exists
                import shutil
                if os.path.exists(working_dir):
                    shutil.rmtree(working_dir)
                os.makedirs(scraped_data_dir, exist_ok=True)
                collected_data = {}
                # Collect data from social platforms if handles provided
                if tiktok_handle:
                    with st.spinner("Collecting TikTok data..."):
                        tiktok_data = simulate_tiktok_scrape(tiktok_handle)
                        if tiktok_data:
                            with open(os.path.join(scraped_data_dir, "tiktok_data.json"), "w") as f:
                                json.dump(tiktok_data, f, indent=2)
                            collected_data["tiktok"] = tiktok_data
                            st.success("✅ TikTok data collected")
                if instagram_handle:
                    with st.spinner("Collecting Instagram data..."):
                        instagram_data = simulate_instagram_scrape(instagram_handle)
                        if instagram_data:
                            with open(os.path.join(scraped_data_dir, "instagram_data.json"), "w") as f:
                                json.dump(instagram_data, f, indent=2)
                            collected_data["instagram"] = instagram_data
                            st.success("✅ Instagram data collected")
                # Always load scraped data from files for profile generation
                loaded_data = {}
                for platform in ["tiktok", "instagram"]:
                    file_path = os.path.join(scraped_data_dir, f"{platform}_data.json")
                    if os.path.exists(file_path):
                        with open(file_path, "r") as f:
                            loaded_data[platform] = json.load(f)
                # Generate comprehensive profile with GPT using both scraped data and description
                with st.spinner("GPT is analyzing data and generating profile..."):
                    profile = generate_unified_profile(artist_name, artist_description, loaded_data)
                    if profile:
                        with open(f"{working_dir}/profile_summary.json", "w") as f:
                            json.dump(profile, f, indent=2)
                        st.session_state.editing_profile = profile
                        st.session_state.editing_data = loaded_data
                        st.session_state.working_dir = working_dir
                        st.success("🎉 Artist profile created successfully! You can now edit it below.")
                    else:
                        st.error("Failed to generate profile.")
            except Exception as e:
                st.error(f"Error during profile creation: {str(e)}")
                st.error(traceback.format_exc())
        
        elif submit_profile:
            st.error("Please provide both artist name and description.")
    
    # Edit profile section
    if 'editing_profile' in st.session_state:
        st.divider()
        st.subheader("✏️ Edit Profile Before Saving")
        profile = st.session_state.editing_profile
        data = st.session_state.editing_data

        # Unified edit form: text fields + analytics + save/cancel
        with st.form("edit_profile_form_unified"):
            profile['name'] = st.text_input("Artist Name", value=profile['name'])
            profile['genre'] = st.text_input("Genre", value=profile.get('genre', ''))
            profile['description'] = st.text_area("Description", value=profile.get('description', ''), height=150)
            profile['vibe'] = st.text_area("Musical description", value=profile.get('vibe', ''), height=80)

            st.markdown('#### Platforms')
            if 'tiktok' in data:
                with st.expander("📱 TikTok Analytics", expanded=True):
                    tiktok = data['tiktok']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Followers", f"{tiktok['followers']:,}")
                    with col2:
                        st.metric("Following", f"{tiktok['following']:,}")
                    with col3:
                        st.metric("Total Likes", f"{tiktok['total_likes']:,}")
                    with col4:
                        st.metric("Videos", f"{tiktok['video_count']:,}")
                    st.write(f"**Bio:** {tiktok['bio']}")
            if 'instagram' in data:
                with st.expander("📸 Instagram Analytics", expanded=True):
                    instagram = data['instagram']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Followers", f"{instagram['followers']:,}")
                    with col2:
                        st.metric("Following", f"{instagram['following']:,}")
                    with col3:
                        st.metric("Posts", f"{instagram['posts']:,}")
                    st.write(f"**Bio:** {instagram['bio']}")
            # Save/Cancel buttons after analytics
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 Save Profile", type="primary"):
                    # Move working dir to permanent artist dir
                    import shutil
                    artist_dir = f"data/artists_for_analytics/{profile['name'].replace(' ', '_').lower()}"
                    if os.path.exists(artist_dir):
                        shutil.rmtree(artist_dir)
                    shutil.move(st.session_state.working_dir, artist_dir)
                    result = save_artist_profile(profile, data)
                    st.success(result)
                    for key in ['editing_profile', 'editing_data', 'working_dir']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            with col2:
                if st.form_submit_button("🗑️ Cancel"):
                    for key in ['editing_profile', 'editing_data', 'working_dir']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
    
    st.divider()
    
    # Display existing profiles with editing capability
    st.subheader("📋 Saved Artist Profiles")
    
    profiles = get_all_artist_profiles()
    if profiles:
        # Display profiles with edit capability
        for i, profile in enumerate(profiles):
            # Defensive: load GPT-generated platform data for each profile
            artist_dir = f"data/artists_for_analytics/{profile['name'].replace(' ', '_').lower()}"
            tiktok_profile_path = os.path.join(artist_dir, "tiktok_profile.json")
            instagram_profile_path = os.path.join(artist_dir, "instagram_profile.json")
            data = {}
            try:
                if os.path.exists(tiktok_profile_path):
                    with open(tiktok_profile_path, 'r') as f:
                        data['tiktok'] = json.load(f)
                if os.path.exists(instagram_profile_path):
                    with open(instagram_profile_path, 'r') as f:
                        data['instagram'] = json.load(f)
            except Exception:
                data = {}
            with st.expander(f"🎤 {profile['name']} ({profile.get('genre', 'Unknown')})"):
                # Edit mode toggle
                edit_key = f"edit_profile_{i}"
                col_edit, col_delete = st.columns([4, 1])
                with col_edit:
                    if st.button(f"✏️ Edit {profile['name']}", key=f"edit_btn_{i}"):
                        st.session_state[edit_key] = not st.session_state.get(edit_key, False)
                with col_delete:
                    if st.button(f"🗑️ Delete", key=f"delete_btn_{i}"):
                        import shutil
                        try:
                            shutil.rmtree(artist_dir)
                            st.success(f"Deleted profile: {profile['name']}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete profile: {e}")
                if st.session_state.get(edit_key, False):
                    # Edit form
                    with st.form(f"edit_form_{i}"):
                        new_name = st.text_input("Name", value=profile['name'])
                        new_genre = st.text_input("Genre", value=profile.get('genre', ''))
                        new_description = st.text_area("Description", value=profile.get('description', ''), height=100)
                        new_vibe = st.text_area("Vibe", value=profile.get('vibe', ''), height=80)
                        if st.form_submit_button("💾 Save Changes"):
                            profile['name'] = new_name
                            profile['genre'] = new_genre
                            profile['description'] = new_description
                            profile['vibe'] = new_vibe
                            profile['last_updated'] = datetime.now().isoformat()
                            result = save_artist_profile(profile, data)
                            st.success(result)
                            st.session_state[edit_key] = False
                            st.rerun()
                else:
                    # --- Main Profile Section ---
                    st.write(f"**Artist:** {profile['name']}")
                    st.write(f"**Genre:** {profile['genre']}")
                    # Show full description
                    st.write(f"**Description:** {profile['description']}")
                    st.write(f"**Vibe:** {profile.get('vibe', 'N/A')}")
                    st.markdown('<span style="font-size:1.1rem;font-weight:600">Platforms</span>', unsafe_allow_html=True)
                    if 'tiktok' in data:
                        with st.expander("📱 TikTok", expanded=True):
                            st.write(f"<span style='font-size:0.95rem'>{data['tiktok'].get('summary', 'No summary available.')}</span>", unsafe_allow_html=True)
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Followers", f"{data['tiktok'].get('followers', 'N/A'):,}")
                            with col2:
                                st.metric("Following", f"{data['tiktok'].get('following', 'N/A'):,}")
                            with col3:
                                st.metric("Total Likes", f"{data['tiktok'].get('total_likes', 'N/A'):,}")
                            with col4:
                                st.metric("Videos", f"{data['tiktok'].get('video_count', 'N/A'):,}")
                            st.write(f"**Bio:** {data['tiktok'].get('bio', 'N/A')}")
                    # Instagram
                    if 'instagram' in data:
                        with st.expander("📸 Instagram", expanded=True):
                            st.write(f"<span style='font-size:0.95rem'>{data['instagram'].get('summary', 'No summary available.')}</span>", unsafe_allow_html=True)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Followers", f"{data['instagram'].get('followers', 'N/A'):,}")
                            with col2:
                                st.metric("Following", f"{data['instagram'].get('following', 'N/A'):,}")
                            with col3:
                                st.metric("Posts", f"{data['instagram'].get('posts', 'N/A'):,}")
                        
                            st.write(f"**Bio:** {data['instagram'].get('bio', 'N/A')}")

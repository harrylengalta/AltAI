# Alta Music Analytics App

This Streamlit app provides artist and campaign management, influencer recommendations, persistent chat, and robust analytics for music industry data.

## Features
- Artist profile management and analytics
- TikTok campaign chooser and influencer recommendations
- Persistent chat with Claude (Anthropic API)
- Data upload and description for streaming/social/case study files
- Robust error handling and user-friendly UI
- Visual analytics with Plotly and Altair

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Setup
1. Clone this repository or copy the code to your local machine.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your API keys:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_key
   APIFY_TOKEN=your_apify_token
   TASK_ID=your_task_id
   CHARTMETRIC_REFRESH_TOKEN=your_chartmetric_token
   ```
4. Run the app:
   ```sh
   streamlit run simple_app.py
   ```

## File Structure
- `simple_app.py` — Main Streamlit app
- `artist_profiles.py` — Artist profile logic
- `requirements.txt` — Python dependencies
- `README.md` — This file
- `data/` — Data files and subfolders for uploads, analytics, and descriptions

## Notes
- Make sure the `data/` directory and its subfolders are writable.
- For TikTok/Instagram analytics, ensure the relevant CSV/JSON files are present in `data/social_media_data/`.
- The app uses Plotly for donut charts and Altair for other visualizations.

## Troubleshooting
- If you see a 403 error on upload, check file permissions and that you are running Streamlit locally with write access.
- For API errors, ensure your `.env` file is correct and all required keys are set.

## License
Proprietary / Internal use only.

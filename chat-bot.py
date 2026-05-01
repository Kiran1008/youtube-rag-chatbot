import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

video_id = "Gfr50f6ZBvo"  # only ID

try:
    api = YouTubeTranscriptApi()

    # Fetch transcript (new method)
    transcript_data = api.fetch(video_id)

    # Convert to plain text
    transcript = " ".join([chunk.text for chunk in transcript_data])

    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

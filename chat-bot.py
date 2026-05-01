import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

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


splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splitter.create_documents([transcript])

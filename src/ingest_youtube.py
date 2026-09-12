"""Ingest YouTube video transcripts, clean them, and save to data/transcripts/."""

import argparse
import datetime
import os
import re
import sys
from typing import Dict, Optional, Tuple

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from various URL formats or raw ID."""
    url_or_id = url_or_id.strip("<> '\"\t\n\r")
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/|v\/|shorts\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id


def slugify(value: str) -> str:
    """Normalize string into a clean filename slug."""
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def get_video_metadata(url: str) -> Dict[str, str]:
    """Retrieve video metadata using yt-dlp without downloading media."""
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        upload_date = info.get('upload_date')
        if upload_date and len(upload_date) == 8:
            formatted_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
        else:
            formatted_date = datetime.date.today().isoformat()
        
        return {
            'id': info.get('id', ''),
            'title': info.get('title', 'Unknown Title'),
            'url': info.get('webpage_url', url),
            'date': formatted_date,
            'uploader': info.get('uploader', ''),
            'duration': str(info.get('duration', '')),
        }


def clean_vtt_text(vtt_content: str) -> str:
    """Clean raw WebVTT content: remove headers, timestamps, formatting tags, and repeated lines."""
    lines = vtt_content.splitlines()
    cleaned_lines = []
    
    # Strip headers, timestamps, and formatting
    timestamp_pattern = re.compile(
        r'^\s*(\d{2}:)?\d{2}:\d{2}[\.,]\d{3}\s*-->\s*(\d{2}:)?\d{2}:\d{2}[\.,]\d{3}.*$'
    )
    tag_pattern = re.compile(r'<[^>]+>')
    
    prev_line = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
            continue
        if line.isdigit():  # Cue index
            continue
        if timestamp_pattern.match(line):
            continue
            
        line = tag_pattern.sub('', line)
        line = line.replace('&nbsp;', ' ')
        line = re.sub(r'\s+', ' ', line).strip()
        
        if line and line != prev_line:
            cleaned_lines.append(line)
            prev_line = line

    return group_into_paragraphs(cleaned_lines)


def group_into_paragraphs(lines: list, max_words_per_para: int = 80) -> str:
    """Group caption lines into cohesive paragraphs."""
    paragraphs = []
    current_para = []
    current_word_count = 0

    for line in lines:
        words = line.split()
        current_para.append(line)
        current_word_count += len(words)

        # Break paragraph if ends with strong punctuation or reaches length threshold
        if (line.endswith(('.', '!', '?')) and current_word_count >= 50) or current_word_count >= max_words_per_para:
            paragraphs.append(' '.join(current_para))
            current_para = []
            current_word_count = 0

    if current_para:
        paragraphs.append(' '.join(current_para))

    return '\n\n'.join(paragraphs)


def fetch_transcript_api(video_id: str, languages: list = None) -> Tuple[Optional[str], Optional[str]]:
    """Fetch transcript using youtube-transcript-api with language fallback."""
    if languages is None:
        languages = ['en', 'en-US', 'en-GB']
    try:
        ytt = YouTubeTranscriptApi()
        snippets = None
        lang_detected = None

        # 1. Try preferred languages
        try:
            snippets = ytt.fetch(video_id, languages=languages)
            lang_detected = 'en'
        except Exception:
            pass

        # 2. Try listing and finding preferred or translated
        if not snippets:
            try:
                transcripts = ytt.list(video_id)
                try:
                    t = transcripts.find_transcript(languages)
                    snippets = t.fetch()
                    lang_detected = t.language_code
                except Exception:
                    # Fallback to the first available transcript (e.g. Hindi, Spanish, etc.)
                    for t in transcripts:
                        snippets = t.fetch()
                        lang_detected = f"{t.language} ({t.language_code})"
                        print(f"[*] English transcript not available; using {lang_detected}")
                        break
            except Exception:
                pass

        if not snippets:
            return None, None

        lines = []
        prev = ""
        for entry in snippets:
            text = getattr(entry, 'text', '') if hasattr(entry, 'text') else str(entry)
            text = re.sub(r'\s+', ' ', text.strip())
            if text and text != prev:
                lines.append(text)
                prev = text
        return group_into_paragraphs(lines), lang_detected
    except Exception as e:
        print(f"[!] Warning: youtube-transcript-api encountered: {e}")
        return None, None


def fetch_transcript_ytdlp(url: str, output_dir: str, video_id: str) -> Optional[str]:
    """Fetch transcript using yt-dlp subtitle extraction."""
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'en-US'],
        'subtitlesformat': 'vtt',
        'ignoreerrors': True,
        'outtmpl': os.path.join(output_dir, f"{video_id}"),
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Locate generated vtt file
        for filename in os.listdir(output_dir):
            if filename.startswith(video_id) and filename.endswith('.vtt'):
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                return clean_vtt_text(content)
    except Exception:
        return None
    return None


def ingest_youtube_transcript(url: str, output_dir: str = "data/transcripts") -> str:
    """Ingest a YouTube URL, extract metadata and transcript, save markdown file."""
    url = url.strip("<> '\"\t\n\r")
    os.makedirs(output_dir, exist_ok=True)
    video_id = extract_video_id(url)
    
    print(f"[+] Fetching video metadata for ID: {video_id}...")
    try:
        metadata = get_video_metadata(url)
    except Exception as e:
        print(f"[!] Warning: Failed to extract rich metadata with yt-dlp: {e}")
        metadata = {
            'id': video_id,
            'title': f"YouTube Video {video_id}",
            'url': f"https://www.youtube.com/watch?v={video_id}" if not url.startswith('http') else url,
            'date': datetime.date.today().isoformat(),
            'uploader': 'Unknown',
            'duration': '',
        }

    print("[+] Extracting transcript...")
    transcript_text = None
    detected_lang = None
    
    # Strategy 1: youtube-transcript-api
    transcript_text, detected_lang = fetch_transcript_api(video_id)
    
    # Strategy 2: yt-dlp subtitle download fallback
    if not transcript_text:
        print("[*] Trying yt-dlp subtitle extraction...")
        transcript_text = fetch_transcript_ytdlp(metadata['url'], output_dir, video_id)
        if transcript_text:
            detected_lang = 'en'
        
    if not transcript_text:
        raise RuntimeError(f"Could not retrieve transcript for video {video_id}. Subtitles may be unavailable or disabled.")

    title_safe = slugify(metadata['title'])[:50]
    out_filename = f"{title_safe or video_id}.md"
    out_filepath = os.path.join(output_dir, out_filename)

    frontmatter = f"""---
title: "{metadata['title'].replace('"', '\\"')}"
url: "{metadata['url']}"
date: "{metadata['date']}"
video_id: "{metadata['id']}"
uploader: "{metadata.get('uploader', '')}"
language: "{detected_lang or 'unknown'}"
---

# {metadata['title']}

{transcript_text}
"""

    with open(out_filepath, 'w', encoding='utf-8') as f:
        f.write(frontmatter)

    print(f"[✓] Successfully saved transcript to: {out_filepath}")
    return out_filepath


def main():
    parser = argparse.ArgumentParser(description="Ingest YouTube video transcripts into markdown files.")
    parser.add_argument("url", help="YouTube video URL or video ID")
    parser.add_argument("--output-dir", default="data/transcripts", help="Directory to save transcripts (default: data/transcripts)")
    args = parser.parse_args()

    try:
        saved_path = ingest_youtube_transcript(args.url, args.output_dir)
        print(saved_path)
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

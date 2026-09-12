---
name: ingest-youtube
description: Ingests a YouTube video URL, downloads subtitles/transcripts using yt-dlp, cleans the content, and stores it in data/transcripts/ with frontmatter containing title, url, and date.
---

# Ingest YouTube Transcript Skill

This skill automates downloading, cleaning, and cataloging YouTube video transcripts for downstream ML analysis and knowledge extraction.

---

## 1. Input Requirements

- **YouTube URL or Video ID**:
  - Full URL: `https://www.youtube.com/watch?v=VIDEO_ID`
  - Short URL: `https://youtu.be/VIDEO_ID`
  - Raw ID: `VIDEO_ID`
- **Target Output Directory**: `data/transcripts/`

---

## 2. Output Specification

The generated file is saved at `data/transcripts/<slug-or-id>.md` using this exact schema:

```markdown
---
title: "Title of the Video"
url: "https://www.youtube.com/watch?v=..."
date: "YYYY-MM-DD"
video_id: "VIDEO_ID"
uploader: "Channel Name"
---

# Title of the Video

Paragraph of cleaned transcript text...

Subsequent paragraph formatted for clarity and readability...
```

---

## 3. Execution Instructions

### Method A: Automated Ingestion Tool (Recommended)

Run the project's ingestion script inside the virtual environment:

```bash
./.venv/bin/python src/ingest_youtube.py "<YOUTUBE_URL>"
```

Optional arguments:
```bash
./.venv/bin/python src/ingest_youtube.py "<YOUTUBE_URL>" --output-dir data/transcripts
```

### Method B: Direct CLI Execution with `yt-dlp`

If executing directly using CLI tools:

1. **Extract Metadata**:
   ```bash
   TITLE=$(./.venv/bin/yt-dlp --print "%(title)s" --skip-download "<YOUTUBE_URL>")
   DATE=$(./.venv/bin/yt-dlp --print "%(upload_date>%Y-%m-%d)s" --skip-download "<YOUTUBE_URL>")
   VIDEO_ID=$(./.venv/bin/yt-dlp --print "%(id)s" --skip-download "<YOUTUBE_URL>")
   ```

2. **Download Subtitles/Transcripts**:
   ```bash
   ./.venv/bin/yt-dlp \
     --skip-download \
     --write-auto-sub \
     --write-sub \
     --sub-lang "en.*,en" \
     --sub-format vtt \
     -o "data/transcripts/%(id)s" \
     "<YOUTUBE_URL>"
   ```

3. **Clean WebVTT Transcript**:
   - Strip WEBVTT headers (`WEBVTT`, `Kind:`, `Language:`).
   - Strip timestamps (`00:00:00.000 --> 00:00:02.000`).
   - Strip formatting tags (`<c>`, `</c>`, `<font>`, HTML entities like `&nbsp;`).
   - Deduplicate consecutive lines produced by rolling auto-generated captions.
   - Group sentences into readable paragraphs (50-80 words).

4. **Write Markdown File**:
   Assemble frontmatter with `title`, `url`, and `date`, followed by the cleaned text, and save to `data/transcripts/<title-slug>.md`.

---

## 4. Verification Checklist

- [ ] File exists in `data/transcripts/`.
- [ ] Frontmatter includes valid `title`, `url`, and `date` in ISO format (`YYYY-MM-DD`).
- [ ] Transcript text is devoid of WebVTT timestamps, tags, or repeated rolling captions.
- [ ] Text is readable with paragraph breaks.

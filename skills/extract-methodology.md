---
name: extract-methodology
description: Reads a transcript file from data/transcripts/ and extracts the core technique taught, code patterns shown, and key warnings/gotchas, saving the output as a structured note in data/knowledge/.
---

# Extract Methodology Skill

This skill extracts structured machine learning knowledge notes from video transcripts. It rigorously adheres to the source material: extracting only what is explicitly discussed without adding external knowledge or hallucinations.

---

## 1. Core Rules: Strict Source Fidelity

> [!IMPORTANT]
> **No Outside Knowledge**: Only extract information, concepts, warnings, and workflows that are directly present or described in the transcript text. Do not supplement with external textbook definitions, unmentioned libraries, or general domain knowledge.
>
> **Code Patterns Constraint**: Only include code, CLI commands, or pseudocode if it was **explicitly mentioned, spoken, or described** in the transcript. If no code was shown or described, you MUST write:
> `No code shown in this video — conceptual explanation only.`
> Do not generate, infer, or synthesize code that was not in the source.

---

## 2. Input & Output Requirements

- **Input**:
  - Filepath: `data/transcripts/<transcript-file>.md`
  - Content: Markdown file with YAML frontmatter (`title`, `url`, `date`) and transcript body.
- **Output**:
  - Filepath: `data/knowledge/<technique-slug>.md`
  - Format: Structured markdown document with metadata frontmatter and standardized sections.

---

## 3. Note Structure Specification

Every output note saved to `data/knowledge/` must strictly adhere to this schema:

```markdown
---
title: "Name of the Core Technique"
source_transcript: "data/transcripts/<source-file>.md"
source_url: "https://www.youtube.com/watch?v=..."
video_title: "Original Video Title"
date_extracted: "YYYY-MM-DD"
technique: "Name of Technique"
tags: ["extracted-tag-1", "extracted-tag-2"]
---

# <Core Technique Name>

> **Summary**: One to two sentence summary strictly based on the speaker's explanation in the transcript.

---

## 1. Core Technique Taught

### Overview & Concepts
- Detailed summary of the technique as explained by the speaker.
- Specific problem or comparison described in the video (e.g., how the video frames the problem and why the method is used).
- Examples, analogies, or datasets referenced in the video.

### Workflow / Diagram
(Optional: Include a Mermaid diagram only if it reflects the workflow explicitly described in the video.)

---

## 2. Code Patterns Shown

No code shown in this video — conceptual explanation only.

(OR if code/syntax was explicitly spoken or demonstrated:)
```<language>
# Exact or directly described code pattern from the video
```

---

## 3. Key Warnings & Gotchas

List only the specific pitfalls, warnings, trade-offs, and bad practices explicitly highlighted by the speaker:

- **Mentioned Pitfall 1**: Specific warning from the transcript.
- **Mentioned Pitfall 2**: Specific limitation or constraint mentioned in the video.
- (If the video mentions no warnings, state: "No specific warnings or gotchas were mentioned in the source.")
```

---

## 4. Execution Procedure for Agents

When requested to extract methodology from a transcript:

1. **Read the Transcript**:
   - Inspect the transcript file in `data/transcripts/`.
   - Read the YAML frontmatter to retrieve `title`, `url`, and `date`.

2. **Strict Extraction (Ground Truth Only)**:
   - **Core Technique**: Extract only the ideas, algorithms, terminology, and analogies provided by the speaker.
   - **Code Patterns**: Check if the speaker provided actual code, commands, or pseudocode. If not, write:
     `No code shown in this video — conceptual explanation only.`
     Never infer or invent code from outside knowledge.
   - **Warnings & Gotchas**: Record only the warnings, mistakes, or bad practices explicitly called out in the video (e.g. "a terrible approach would be...").

3. **Format and Save**:
   - Create a clean slug for the filename based on the technique (e.g. `data/knowledge/cross-validation.md`).
   - Save the finalized note to `data/knowledge/<technique-slug>.md`.

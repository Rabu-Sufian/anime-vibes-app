# utils/subtitle_parser.py

import re

def parse_srt_file(uploaded_file):
    content = uploaded_file.read().decode("utf-8")
    lines = content.split('\n')
    
    dialogues = []
    buffer = []
    for line in lines:
        if re.match(r"^\d+$", line.strip()) or "-->" in line:
            continue  # Skip index and timestamp lines
        elif line.strip() == "":
            if buffer:
                dialogues.append(" ".join(buffer).strip())
                buffer = []
        else:
            buffer.append(line.strip())
    
    if buffer:
        dialogues.append(" ".join(buffer).strip())
    
    return dialogues


def parse_ass_file(uploaded_file):
    dialogues = []
    for line in uploaded_file:
        line = line.decode("utf-8", errors="ignore")
        if line.startswith("Dialogue:"):
            parts = line.split(",", 9)  # The 10th part contains the actual dialogue
            if len(parts) >= 10:
                text = parts[9].strip().replace("\\N", " ").replace("{\\.*?}", "")
                if text:
                    dialogues.append(text)
    return dialogues

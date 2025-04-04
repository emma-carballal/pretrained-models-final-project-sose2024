import re
import os

def process_subtitle_file(input_file):
    try:
        with open(input_file, 'r', encoding='ISO-8859-1') as file:
            lines = file.readlines()
    except UnicodeDecodeError as e:
        print(f"Error reading file: {input_file}")
        print(f"UnicodeDecodeError: {e}")
        return []

    processed_lines = []
    subtitle_text = []


    for line in lines:
        line = line.strip()

        # Skip subtitle number and timing lines
        if re.match(r"^\d+$", line) or re.match(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$", line):
            continue

        if not line:
            if subtitle_text:
                # Check if the subtitle has more than two lines
                if len(subtitle_text) > 2:
                    subtitle_text = []
                    continue
                # Join the subtitle text with <eol> if it's multiline, add <bos> at the beginning, <eos> at the end
                if len(subtitle_text) > 1:
                    processed_subtitle = f"<bos>{'<eol>'.join(subtitle_text)}<eos>"
                else:
                    processed_subtitle = f"<bos>{subtitle_text[0]}<eos>"
                processed_lines.append(processed_subtitle)
                subtitle_text = []
        else:
            subtitle_text.append(line)

    # Process any remaining subtitle text
    if subtitle_text:
        if len(subtitle_text) > 1:
            processed_subtitle = f"<bos>{'<eol>'.join(subtitle_text)}<eos>"
        else:
            processed_subtitle = f"<bos>{subtitle_text[0]}<eos>"
        processed_lines.append(processed_subtitle)

    return processed_lines

def process_text(subtitles, output_file):
    processed_output = []
    for text in subtitles:
        # Find the indices of the tags
        bos_index = text.find("<bos>") + len("<bos>")
        eos_index = text.find("<eos>")
        eol_index = text.find("<eol>")

        # Adjust indices if <eol> is not found
        if eol_index == -1:
            eol_index = eos_index
            bos_to_eol = text[bos_index:eol_index]
            eol_to_eos = ''
        else:
            bos_to_eol = text[bos_index:eol_index]
            eol_to_eos = text[eol_index + len("<eol>"):eos_index]

        # Count the characters in both sections
        bos_to_eol_count = len(bos_to_eol)
        eol_to_eos_count = len(eol_to_eos)

        total_length = bos_to_eol_count + eol_to_eos_count

        # If total characters are less than 41, replace <eol> with a space
        if total_length < 41:
            new_text = text.replace("<eol>", " ")
        else:
            new_text = text

        processed_output.append(new_text + '\n')
            
    # Write to output file
    with open(output_file, 'w') as file:
        file.writelines(processed_output)

# Function to process all files in the "data" folder
def process_all_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    episode_number = 1

    for file_name in files:
        # Only process .srt files
        if file_name.endswith('.srt'):
            padded_episode_number = str(episode_number).zfill(2)
            input_file = os.path.join(folder_path, file_name)
            output_file = os.path.join(folder_path, f'episode_7_{padded_episode_number}.txt')

            process_text(process_subtitle_file(input_file), output_file)

            print(f"Processed {input_file} -> {output_file}")
            episode_number += 1

# Process all files in the "data" folder
folder_path = './data'
process_all_files_in_folder(folder_path)
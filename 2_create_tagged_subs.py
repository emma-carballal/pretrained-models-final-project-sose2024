import os

def combine_all_files_with_tags(folder_path, combined_output_file):
    all_text = []

    files = sorted([f for f in os.listdir(folder_path) if f.startswith('episode_7_') and f.endswith('.txt')])
    print(f"Files found: {files}")
    if not os.path.exists(folder_path):
        print(f"Folder path does not exist: {folder_path}")

    for file_name in files:
        input_file = os.path.join(folder_path, file_name)
        with open(input_file, 'r', encoding='ISO-8859-1') as file:
            file_text = file.read()
            all_text.append(file_text)
        print(f"Processed file: {input_file}")

    combined_text = ''.join(all_text).replace('<bos>', '').replace('<eos>', '').replace('<eol>', ' <eol> ').replace('00:00:0,500 --> 00:00:2,00 <eol> <font color="#ffff00" size=14>www.tvsubtitles.net</font>', '')

    # Write the combined text with tags to the output file
    with open(combined_output_file, 'w', encoding='ISO-8859-1') as output:
        output.write(combined_text)

    print(f"All files combined and saved to {combined_output_file}")

def remove_eol_and_save_text(input_file, output_file):
    with open(input_file, 'r', encoding='ISO-8859-1') as file:
        combined_text = file.read()

    cleaned_text = combined_text.replace(' <eol> ', ' ')

    with open(output_file, 'w', encoding='ISO-8859-1') as output:
        output.write(cleaned_text)

    print(f"Cleaned text saved to {output_file}")

folder_path = './data'
combined_output_file = 'data/combined_with_tags.txt'  # Combined file with tags
cleaned_output_file = 'data/combined_without_tags.txt'  # Cleaned file without <eol>

# Combine all files with tags
combine_all_files_with_tags(folder_path, combined_output_file)

# Remove <eol> tags and save cleaned version
remove_eol_and_save_text(combined_output_file, cleaned_output_file)

import json
from sklearn.model_selection import train_test_split

with open('data/combined_with_tags.txt', 'r', encoding='ISO-8859-1') as f:
    subtitles = f.readlines()

def process_subtitle(subtitle):
    subtitle = subtitle.strip()
    segments = subtitle.split('<eol>')
    words = []
    labels = []

    for i, segment in enumerate(segments):
        segment = segment.strip()
        segment_words = segment.split()
        words.extend(segment_words)
        segment_labels = [0] * len(segment_words)

        # If not the last segment, mark the last word as 1
        if i != len(segments) - 1 and len(segment_words) > 0:
            segment_labels[-1] = 1

        labels.extend(segment_labels)

    text = ' '.join(words)
    return {'text': text, 'words': words, 'labels': labels}

processed_data = [process_subtitle(sub) for sub in subtitles]

# Filter out subtitles where the length is less than or equal to 42 characters
filtered_data = [item for item in processed_data if len(item['text']) > 42]

# Split data into training and testing sets (60% training, 40% testing)
train_data, test_data = train_test_split(processed_data, test_size=0.4, random_state=42)

train_output = 'data/train_full_data.jsonl'
with open(train_output, 'w', encoding='ISO-8859-1') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Training data has been saved in JSON Lines format to {train_output}.")

test_output = 'data/evaluation/en/test_full_data.jsonl'
with open(test_output, 'w', encoding='ISO-8859-1') as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Testing data has been saved in JSON Lines format to {test_output}.")

# Use all data as test data for Spanish
# test_output = 'data/evaluation/es/test_full_data.jsonl'
# with open(test_output, 'w', encoding='ISO-8859-1') as f:
#     for item in processed_data:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n')
# print(f"Testing data has been saved in JSON Lines format to {test_output}.")

# filtered_test_output = 'data/evaluation/es/test_filtered_data.jsonl'
# with open(filtered_test_output, 'w', encoding='ISO-8859-1') as f:
#     for item in filtered_data:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n')
# print(f"Filtered testing data has been saved in JSON Lines format to {filtered_test_output}. Number of subtitles: {len(filtered_data)}")

# Split the filtered data into training and testing sets (60% training, 40% testing)
train_filtered_data, test_filtered_data = train_test_split(filtered_data, test_size=0.4, random_state=42)

train_filtered_output = 'data/train_filtered_data.jsonl'
with open(train_filtered_output, 'w', encoding='ISO-8859-1') as f:
    for item in train_filtered_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Training filtered data has been saved in JSON Lines format to {train_filtered_output}.")

test_filtered_output = 'data/evaluation/en/test_filtered_data.jsonl'
with open(test_filtered_output, 'w', encoding='ISO-8859-1') as f:
    for item in test_filtered_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Testing filtered data has been saved in JSON Lines format to {test_filtered_output}.")

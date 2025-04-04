import json
import torch
from tqdm import tqdm
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizerFast

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

def predict_linebreak(subtitle, model, tokenizer, prob_threshold=0.1):    
    subtitle = subtitle.strip()
    words = subtitle.split()
    
    # Skip linebreak prediction if sentence length is less than or equal to 42 characters
    if len(subtitle) <= 42:
        return subtitle, 0.0
    
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", padding='max_length', max_length=100)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=2)
    probabilities = probabilities[0]
    prob_label_1 = probabilities[:, 1]  # Probability of label 1 for each token

    word_ids = inputs.word_ids()
    word_probabilities = {}
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None:
            prob = prob_label_1[idx].item()
            if word_idx in word_probabilities:
                word_probabilities[word_idx].append(prob)
            else:
                word_probabilities[word_idx] = [prob]

    # Aggregate probabilities per word (mean)
    aggregated_probabilities = {}
    for word_idx, probs in word_probabilities.items():
        aggregated_probabilities[word_idx] = sum(probs) / len(probs)

    # Find the word with the highest probability
    max_prob_word_idx = max(aggregated_probabilities, key=aggregated_probabilities.get)
    max_prob = aggregated_probabilities[max_prob_word_idx]

    # Insert <eol> after the word with the highest probability
    index = max_prob_word_idx + 1  # +1 to insert after the word
    words_with_break = words[:index] + ['<eol>'] + words[index:]

    new_subtitle = ' '.join(words_with_break)
    return new_subtitle, max_prob

def write_results_to_jsonl(sentences, model, output_file, prob_threshold=0.1):
    with open(output_file, 'w', encoding='ISO-8859-1') as f:
        for sentence in tqdm(sentences, desc="Progress on evaluation dataset:"):
            # Skip sentences with length <= 42 characters
            if len(sentence) <= 42:
                continue

            sentence_result = {
                "original_sentence": sentence,
                "model_predictions": {}
            }

            predicted_sentence, max_prob = predict_linebreak(sentence, model, tokenizer, prob_threshold=prob_threshold)
            sentence_result["model_predictions"] = {
                "predicted_sentence": predicted_sentence.replace(" <eol> ", "\n"),
                "max_probability": max_prob
            }
            
            f.write(json.dumps(sentence_result, ensure_ascii=False) + "\n")

def load_sentences_from_jsonl(input_file):
    sentences = []
    with open(input_file, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            data = json.loads(line)
            if 'text' in data:
                sentences.append(data['text'].strip())
    return sentences

input_file = "data/evaluation/es/human_feedback_dataset.jsonl"
output_file = "data/evaluation/es/predictions/predictions.jsonl"

sentences = load_sentences_from_jsonl(input_file)

model_path = './model/xlm-roberta-base'
tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path)
model = XLMRobertaForTokenClassification.from_pretrained(model_path).to(device)

write_results_to_jsonl(sentences, model, output_file)
print(f"Predictions saved to {output_file}")
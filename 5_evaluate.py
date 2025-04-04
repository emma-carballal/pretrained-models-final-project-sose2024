import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizerFast, XLMRobertaForTokenClassification, XLMRobertaTokenizerFast
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import tokenize_and_align_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

test_datasets = {
    # 'english': 'data/evaluation/en/test_full_data.jsonl',
    'english': 'data/evaluation/en/test_filtered_data.jsonl',    
    # 'spanish': 'data/evaluation/es/test_full_data.jsonl',
    'spanish': 'data/evaluation/es/test_filtered_data.jsonl',
}

# Fine-tuned models
models = {
    'bert': {
        'model_path': './model/bert-base-multilingual-cased',
        'tokenizer': BertTokenizerFast,
        'model': BertForTokenClassification
    },
    'xlm-roberta': {
        'model_path': './model/xlm-roberta-base',
        'tokenizer': XLMRobertaTokenizerFast,
        'model': XLMRobertaForTokenClassification
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# Function to evaluate a model on a test dataset
def evaluate_model(model, tokenizer, test_data_path, metrics_output_file, device):
    # Load test data
    test_data = []
    with open(test_data_path, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            test_data.append(json.loads(line))

    test_dataset = Dataset.from_list(test_data)

    tokenized_test = test_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    tokenized_test = tokenized_test.remove_columns(['text', 'words'])
    tokenized_test.set_format('torch')

    test_dataloader = DataLoader(tokenized_test, batch_size=16)

    # Evaluate the model
    model.eval()
    true_labels, predictions = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            word_ids_batch = batch['word_ids']

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Aggregate predictions at the word level
            for i, word_ids in enumerate(word_ids_batch):
                logits_per_word = {}
                for j, word_id in enumerate(word_ids):
                    if word_id == -1 or labels[i, j] == -100:  # Skip special tokens and padding tokens
                        continue
                    if word_id not in logits_per_word:
                        logits_per_word[word_id] = []
                    logits_per_word[word_id].append(logits[i, j])

                for word_id, word_logits in logits_per_word.items():
                    avg_logits = torch.mean(torch.stack(word_logits), dim=0)
                    predicted_label = torch.argmax(avg_logits).item()
                    predictions.append(predicted_label)

                # Remove -100 labels
                true_labels.extend([label for label in labels[i].tolist() if label != -100])

    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')

    print(f"Eval Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Predicted Not Break', 'Predicted Break'], 
                            yticklabels=['True Not Break', 'True Break'])
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    plt.savefig(conf_matrix_output_file)
    plt.close()

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    output_dir = os.path.dirname(metrics_output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(metrics_output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"Metrics saved to {metrics_output_file}")

for model_name, info in models.items():
    # Load the model and tokenizer
    model = info['model'].from_pretrained(info['model_path'])
    tokenizer = info['tokenizer'].from_pretrained(info['model_path'])
    model.to(device)

    for lang, test_data_path in test_datasets.items():
        print(f"Evaluating {model_name} model on filtered {lang} test data...")
        
        # metrics_output_file = f'./data/evaluation/results/{model_name}_full_{lang}_metrics.json'
        # conf_matrix_output_file = f'./data/evaluation/results/{model_name}_full_{lang}_conf_matrix.png'

        metrics_output_file = f'./data/evaluation/results/{model_name}_filtered_{lang}_metrics.json'
        conf_matrix_output_file = f'./data/evaluation/results/{model_name}_filtered_{lang}_conf_matrix.png'


        evaluate_model(model, tokenizer, test_data_path, metrics_output_file, device)
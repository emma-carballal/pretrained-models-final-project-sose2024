import json
import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizerFast, XLMRobertaForTokenClassification, XLMRobertaTokenizerFast, AdamW
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import tokenize_and_align_labels
import os

data = []
# jsonl_file = 'data/train/train_full_data.jsonl'
jsonl_file = 'data/train/train_filtered_data.jsonl'

with open(jsonl_file, 'r', encoding='ISO-8859-1') as f:
    for line in f:
        data.append(json.loads(line))

# Dataset Creation
dataset = Dataset.from_list(data)
train_test = dataset.train_test_split(test_size=0.2)
datasets = DatasetDict({
    'train': train_test['train'],
    'test': train_test['test']
})

models = [
    ('bert-base-multilingual-cased', BertForTokenClassification, BertTokenizerFast),
    ('xlm-roberta-base', XLMRobertaForTokenClassification, XLMRobertaTokenizerFast)
]

# Train models
for model_name, model, tokenizer in models:
    print(f"Training model: {model_name}")
    
    tokenizer = tokenizer.from_pretrained(model_name)
    model = model.from_pretrained(model_name, num_labels=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Tokenize the datasets
    tokenized_datasets = datasets.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['text', 'words'])
    tokenized_datasets.set_format('torch')

    # Create DataLoader
    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=16, shuffle=True)
    test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=16)

    # Set optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            word_ids_batch = batch['word_ids']  # Use word_ids for logits averaging

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

            # Custom word-level loss computation
            averaged_logits = []
            new_labels = []
            for i, word_ids in enumerate(word_ids_batch):
                logits_per_word = {}
                labels_per_word = {}
                for j, word_id in enumerate(word_ids):
                    if word_id == -1:  # Skip special tokens
                        continue
                    if word_id not in logits_per_word:
                        logits_per_word[word_id] = []
                    logits_per_word[word_id].append(logits[i, j])
                    labels_per_word[word_id] = labels[i, j]
                for word_id, word_logits in logits_per_word.items():
                    averaged_logits.append(torch.mean(torch.stack(word_logits), dim=0))  # Average across subwords
                    new_labels.append(labels_per_word[word_id])

            # Stack logits and labels
            averaged_logits = torch.stack(averaged_logits).to(device)
            new_labels = torch.stack(new_labels).to(device)
            
            # Class '0' is the majority and class '1' is the minority
            class_weights = torch.tensor([1.0, 5.0]).to(device)

            # Compute loss
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
            loss = loss_fn(averaged_logits, new_labels)
            total_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')
    
    # Save trained model and tokenizer
    output_dir = f'./model/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")

    # Evaluation
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

                true_labels.extend([label for label in labels[i].tolist() if label != -100])

    # Compute metrics at the word level
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')

    print(f'Eval Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # metrics_output_file = f'{output_dir}/metrics.json'
    metrics_output_file = f'{output_dir}/metrics_filtered.json'

    with open(metrics_output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"Metrics saved to {metrics_output_file}")
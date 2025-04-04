# Tokenization and Label Alignment
def tokenize_and_align_labels(data, tokenizer):
    tokenized_inputs = tokenizer(
        data['words'],
        is_split_into_words=True,
        padding='max_length',
        max_length=100,
        truncation=True
    )

    labels = []
    word_ids_list = []
    for i, label in enumerate(data['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)
        word_ids_list.append(word_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_ids"] = word_ids_list  # Add word_ids to tokenized_inputs

    return tokenized_inputs
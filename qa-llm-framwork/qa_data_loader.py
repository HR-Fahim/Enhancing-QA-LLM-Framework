import pandas as pd
import torch
from torch.utils.data import DataLoader, default_data_collator
from data_preprocessing import DataProcessor

class QADataLoader:
    def __init__(self, train_data_path, validation_data_path):
        self.train_data = pd.read_csv(train_data_path)
        self.validation_data = pd.read_csv(validation_data_path)

    def process_data(self, tokenizer):
        self.train_data['answer_text'] = self.train_data['answers'].apply(DataProcessor.extract_answer_text)
        self.validation_data['answer_text'] = self.validation_data['answers'].apply(DataProcessor.extract_answer_text)

        self.train_data = self.train_data.dropna(subset=['context', 'question'])
        train_texts = DataProcessor.create_tokenizer_dataset(self.train_data)

        with open('qa_texts.txt', 'w') as f:
            for text in train_texts:
                f.write(f"{text}\n")

        preprocessed_train_data = [DataProcessor.preprocess_row(row, tokenizer) for _, row in self.train_data.iterrows()]
        preprocessed_val_data = [DataProcessor.preprocess_row(row, tokenizer) for _, row in self.validation_data.iterrows()]

        train_loader = DataLoader(preprocessed_train_data, batch_size=8, shuffle=True, collate_fn=default_data_collator)
        validation_loader = DataLoader(preprocessed_val_data, batch_size=8, collate_fn=default_data_collator)

        return train_loader, validation_loader

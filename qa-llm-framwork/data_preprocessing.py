import json
import csv
import pandas as pd
import re
import os
import torch
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast, default_data_collator

class DataPreprocessing:
    """Preprocesses of SQuAD json data for training and validation."""
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data

    @staticmethod
    def process_squad_to_csv(input_path, output_path):
        # Load the JSON data
        with open(input_path, 'r', encoding='utf-8') as json_file:
            squad_data = json.load(json_file)
        
        # Prepare a list to store rows
        rows = []
        
        # Extract data
        for article in squad_data['data']:
            title = article.get('title', 'No Title')
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question_id = qa['id']
                    question = qa['question']
                    
                    # Extract answers
                    answer_texts = "; ".join([ans['text'] for ans in qa['answers']])
                    answer_starts = "; ".join([str(ans['answer_start']) for ans in qa['answers']])
                    
                    # Add the row to the list
                    rows.append([
                        title, context, question_id, question, answer_texts, answer_starts
                    ])
        
        # Save the data to a CSV file
        with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "Article Title", "context", "Question ID", "question",
                "answer_text", "Answer Start Positions"
            ])
            csv_writer.writerows(rows)
        
        # Convert rows to a DataFrame and display the head
        df = pd.DataFrame(rows, columns=[
            "Article Title", "context", "Question ID", "question",
            "answer_text", "Answer Start Positions"
        ])
        print(f"Head of {output_path}:")
        print(df.head())

    def preprocess_row(self, row, tokenizer):
        context = str(row['context']) if not pd.isna(row['context']) else ""
        question = str(row['question']) if not pd.isna(row['question']) else ""
        answer_text = str(row['answer_text']) if not pd.isna(row['answer_text']) else ""

        start_char = context.find(answer_text)
        end_char = start_char + len(answer_text) if start_char != -1 else -1

        # Step 1: Clean special characters
        context = re.sub(r'[^\x00-\x7F]+', ' ', context)  # Remove non-ASCII characters (e.g., â€“)
        context = re.sub(r'\s+', ' ', context).strip()    # Normalize whitespace

        question = re.sub(r'[^\x00-\x7F]+', ' ', question)  # Remove non-ASCII characters (e.g., â€“)
        question = re.sub(r'\s+', ' ', question).strip()    # Normalize whitespace

        tokenized = tokenizer(
            text=question,
            text_pair=context,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_offsets_mapping=True
        )

        offsets = tokenized['offset_mapping']
        start_token, end_token = None, None

        for idx, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_token = idx
            if start < end_char <= end:
                end_token = idx
                break

        if start_token is None or end_token is None:
            start_token, end_token = 0, 0

        tokenized['start_positions'] = start_token
        tokenized['end_positions'] = end_token
        tokenized.pop('offset_mapping', None)

        return {key: torch.tensor(val) for key, val in tokenized.items()}

    def create_tokenizer_dataset(self):
        texts = self.train_data['context'].tolist() + self.train_data['question'].tolist()
        return texts

    def train_validation_loaders(self):
        # Save dataset text for ByteLevelBPETokenizer
        texts = self.create_tokenizer_dataset()
        with open('qa_texts.txt', 'w') as f:
            for text in texts:
                f.write(f"{text}\n")

        # Train ByteLevelBPETokenizer
        bpe_tokenizer = ByteLevelBPETokenizer()
        bpe_tokenizer.train(files=['qa_texts.txt'], vocab_size=10000, min_frequency=5, special_tokens=[
            "<s>", "<pad>", "</s>", "<unk>", "<mask>"
        ])

        os.makedirs("/content/tokenizer_bpe", exist_ok=True)
        bpe_tokenizer.save_model("/content/tokenizer_bpe")

        # Load the trained tokenizer
        vocab_file = "/content/tokenizer_bpe/vocab.json"
        merges_file = "/content/tokenizer_bpe/merges.txt"

        tokenizer = RobertaTokenizerFast(
            vocab_file=vocab_file,
            merges_file=merges_file,
            model_max_length=512,
            pad_token="<pad>",
            clean_up_tokenization_spaces=False
        )

        # Preprocess train and validation data
        preprocessed_train_data = [self.preprocess_row(row, tokenizer) for _, row in self.train_data.iterrows()]
        preprocessed_val_data = [self.preprocess_row(row, tokenizer) for _, row in self.val_data.iterrows()]

        train_loader = DataLoader(preprocessed_train_data, batch_size=8, shuffle=True, collate_fn=default_data_collator)
        validation_loader = DataLoader(preprocessed_val_data, batch_size=8, collate_fn=default_data_collator)

        return train_loader, validation_loader

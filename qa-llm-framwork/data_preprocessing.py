import json
import csv
import random
import string
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

    @staticmethod
    def remove_ascii_symbols(input_string):
        symbols = string.punctuation
        symbols_set = set(symbols) - {'?'}
        cleaned_string = ''.join(char for char in input_string if char not in symbols_set)
        return cleaned_string

    @staticmethod
    def preprocess_row(row, tokenizer):
        context = str(row['context']) if not pd.isna(row['context']) else ""
        question = str(row['question']) if not pd.isna(row['question']) else ""
        answer_text = str(row['answer_text']) if not pd.isna(row['answer_text']) else ""

        start_char = context.find(answer_text)
        end_char = start_char + len(answer_text) if start_char != -1 else -1

        context = DataPreprocessing.remove_ascii_symbols(context)
        question = DataPreprocessing.remove_ascii_symbols(question)

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

        # Handle NaN values in train_data['answer_text']
        self.train_data['answer_text'] = self.train_data['answer_text'].fillna('')

        # Few-shot sampling
        with_answers = self.train_data[self.train_data['answer_text'].str.len() > 0]
        without_answers = self.train_data[self.train_data['answer_text'].str.len() == 0]

        total_samples = 16
        half_of_total_samples = total_samples // 2

        few_shot_with_answers = with_answers.sample(
            n=min(half_of_total_samples, len(with_answers)), random_state=45, replace=True
        )

        if len(without_answers) >= half_of_total_samples:
            few_shot_without_answers = without_answers.sample(
                n=min(half_of_total_samples, len(without_answers)), random_state=45, replace=True
            )
            few_shot_samples = pd.concat([few_shot_with_answers, few_shot_without_answers])
        else:
            few_shot_samples = with_answers.sample(n=min(total_samples, len(with_answers)), random_state=45, replace=True)

        # Augment data
        augmented_samples = []

        def augment_context(row):
            context = row['context']
            words = context.split()
            random.shuffle(words)
            return ' '.join(words)

        def augment_row(row):
            row['context'] = augment_context(row)
            return row

        for _, row in few_shot_samples.iterrows():
            augmented_samples.append(augment_row(row))

        few_shot_samples = pd.concat([few_shot_samples, pd.DataFrame(augmented_samples)])

        # Preprocess train and validation data
        preprocessed_train_data = [self.preprocess_row(row, tokenizer) for _, row in few_shot_samples.iterrows()]
        preprocessed_val_data = [self.preprocess_row(row, tokenizer) for _, row in self.val_data.iterrows()]

        train_loader = DataLoader(preprocessed_train_data, batch_size=8, shuffle=True, collate_fn=default_data_collator)
        validation_loader = DataLoader(preprocessed_val_data, batch_size=8, collate_fn=default_data_collator)

        return train_loader, validation_loader

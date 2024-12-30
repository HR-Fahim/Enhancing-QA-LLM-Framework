import pandas as pd
import ast
import re
import os
import torch
import warnings
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast, RobertaModel, default_data_collator

class DataPreprocessing:
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data

    def extract_answer_text(self, answer):
        if isinstance(answer, str):
            try:
                # Replace 'array(..., dtype=object)' with list-like structure for parsing
                cleaned_string = re.sub(r'array\((.*?), dtype=.*?\)', r'[\1]', answer)
                data = ast.literal_eval(cleaned_string)
            except (ValueError, SyntaxError):
                return None
        else:
            data = answer  # If already a dictionary or not a string

        # Extract the first string from 'text' if it's a list
        if isinstance(data, dict) and 'text' in data and isinstance(data['text'], list):
            first_text = data['text'][0]  # Extract the first element
            if isinstance(first_text, list) and first_text:
                return first_text[0].strip("[]").replace("'", "")  # Clean up the string
        return None

    def extract_answer_start(self, answer):
        if isinstance(answer, str):
            try:
                # Replace 'array(..., dtype=int32)' with a list-like structure for parsing
                cleaned_string = re.sub(r'array\((.*?), dtype=.*?\)', r'[\1]', answer)
                data = ast.literal_eval(cleaned_string)
            except (ValueError, SyntaxError):
                return None
        else:
            data = answer  # If already a dictionary or not a string

        # Extract the first integer from 'answer_start', handling nested lists
        if isinstance(data, dict) and 'answer_start' in data:
            answer_start = data['answer_start']
            if isinstance(answer_start, list) and answer_start:
                # If nested lists, flatten and take the first integer
                if isinstance(answer_start[0], list):
                    return int(answer_start[0][0])  # Handle nested lists
                return int(answer_start[0])  # Handle single-level lists
            elif isinstance(answer_start, int):
                return int(answer_start)  # Directly return the integer
        return None

    def preprocess_row(self, row, tokenizer):
        context = str(row['context']) if not pd.isna(row['context']) else ""
        question = str(row['question']) if not pd.isna(row['question']) else ""
        answer_text = str(row['answer_text']) if not pd.isna(row['answer_text']) else ""

        start_char = context.find(answer_text)
        end_char = start_char + len(answer_text) if start_char != -1 else -1

        # Step 1: Clean special characters
        context = re.sub(r'[^\x00-\x7F]+', ' ', context)  # Remove non-ASCII characters (e.g., â€“)
        context = re.sub(r'\s+', ' ', context).strip()    # Normalize whitespace

        # # Step 1: Clean special characters
        question = re.sub(r'[^\x00-\x7F]+', ' ', context)  # Remove non-ASCII characters (e.g., â€“)
        question = re.sub(r'\s+', ' ', context).strip()    # Normalize whitespace

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

    def train_validation_loaders(self, tokenizer):
        texts = self.create_tokenizer_dataset()
        with open('qa_texts.txt', 'w') as f:
            for text in texts:
                f.write(f"{text}\n")

        bpe_tokenizer = ByteLevelBPETokenizer()
        bpe_tokenizer.train(files=['qa_texts.txt'], vocab_size=10000, min_frequency=5, special_tokens=[
            "<s>", "<pad>", "</s>", "<unk>", "<mask>"
        ])

        os.makedirs("/content/tokenizer_bpe", exist_ok=True)
        bpe_tokenizer.save_model("/content/tokenizer_bpe")

        vocab_file = "/content/tokenizer_bpe/vocab.json"
        merges_file = "/content/tokenizer_bpe/merges.txt"

        tokenizer = RobertaTokenizerFast(
            vocab_file=vocab_file,
            merges_file=merges_file,
            model_max_length=512,  # Increased max length for better context understanding
            pad_token="<pad>",
            clean_up_tokenization_spaces=False
        )

        preprocessed_train_data = [self.preprocess_row(row, tokenizer) for _, row in self.train_data.iterrows()]
        preprocessed_val_data = [self.preprocess_row(row, tokenizer) for _, row in self.val_data.iterrows()]

        train_loader = DataLoader(preprocessed_train_data, batch_size=8, shuffle=True, collate_fn=default_data_collator)
        validation_loader = DataLoader(preprocessed_val_data, batch_size=8, collate_fn=default_data_collator)

        return train_loader, validation_loader
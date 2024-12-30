import re
import ast
import pandas as pd
import torch

class DataProcessor:
    @staticmethod
    def extract_answer_text(answer):
        if isinstance(answer, str):
            try:
                cleaned_string = re.sub(r'array\((.*?), dtype=.*?\)', r'[\1]', answer)
                data = ast.literal_eval(cleaned_string)
            except (ValueError, SyntaxError):
                return None
        else:
            data = answer

        if isinstance(data, dict) and 'text' in data and isinstance(data['text'], list):
            first_text = data['text'][0]
            if isinstance(first_text, list) and first_text:
                return first_text[0].strip("[]").replace("'", "")
        return None

    @staticmethod
    def create_tokenizer_dataset(data):
        return data['context'].tolist() + data['question'].tolist()

    @staticmethod
    def preprocess_row(row, tokenizer):
        context = str(row['context']) if not pd.isna(row['context']) else ""
        question = str(row['question']) if not pd.isna(row['question']) else ""
        answer_text = str(row['answer_text']) if not pd.isna(row['answer_text']) else ""

        start_char = context.find(answer_text)
        end_char = start_char + len(answer_text) if start_char != -1 else -1

        context = re.sub(r'[^\x00-\x7F]+', ' ', context).strip()
        context = re.sub(r'\s+', ' ', context).strip()   

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


import pandas as pd
from transformers import RobertaTokenizerFast
from train_eval import train_and_evaluate_model
from data_preprocessing import DataPreprocessing
from transformers import BertModel, RobertaModel
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score

def main():

    class_dict = {
    "RoBERTa-Base": lambda: RobertaModel.from_pretrained("roberta-base"),
    #"RoBERTa-Large": lambda: RobertaModel.from_pretrained("roberta-large"),
    "BERT-Base-Uncased": lambda: BertModel.from_pretrained("bert-base-uncased"),
    #"BERT-Large-Uncased": lambda: BertModel.from_pretrained("bert-large-uncased"),
    #"BERT-Base-Cased": lambda: BertModel.from_pretrained("bert-base-cased"),
    #"BERT-Large-Cased": lambda: BertModel.from_pretrained("bert-large-cased"),
    "SpanBERT": lambda: BertModel.from_pretrained("SpanBERT/spanbert-base-cased"),
    "SPLiTTER": lambda: BertModel.from_pretrained("tau/splinter-base"),
    "ALBERT-Base": lambda: BertModel.from_pretrained("albert-base-v2"),
    #"ALBERT-Large": lambda: AutoModelForQuestionAnswering.from_pretrained("albert-large-v2"),
    "DistilBERT-Uncased": lambda: BertModel.from_pretrained("distilbert-base-uncased"),
    # "DistilBERT-Cased": lambda: AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased"),
    # "ELECTRA-Base": lambda: AutoModelForQuestionAnswering.from_pretrained("google/electra-base-discriminator"),
    # "ELECTRA-Large": lambda: AutoModelForQuestionAnswering.from_pretrained("google/electra-large-discriminator"),
    }

    # Load the train and validation datasets
    train_data = pd.read_csv("/dataset/train.csv")
    validation_data = pd.read_csv("/dataset/validation.csv")
    
    # Instantiate the DataPreprocessing class
    data_preprocessing = DataPreprocessing(train_data, validation_data)

    # Load the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    # Preprocess the data and get the train and validation loaders
    train_loader, validation_loader = data_preprocessing.train_validation_loaders(tokenizer)

    # Loop through each model
    for model_name, model_class_fn in class_dict.items():
        print(f"Training {model_name}...")
        train_and_evaluate_model(model_name, model_class_fn, train_loader, validation_loader)

if __name__ == "__main__":
    main()


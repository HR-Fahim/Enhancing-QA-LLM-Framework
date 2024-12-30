import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForQuestionAnswering, RobertaModel, BertModel
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from learnable_positional_encoding import LearnablePositionalEncoding
from multihead_dynamic_attention import MultiHeadDynamicAttention
from moe_layer import MoELayer

from enchanced_qa_model import EnhancedQAModel

def train_and_evaluate_model(model_name, model_class_fn, train_loader, validation_loader):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = model_class_fn()
    hidden_size = model_class.config.hidden_size if hasattr(model_class.config, 'hidden_size') else model_class.config.d_model
    model = EnhancedQAModel(model_class, hidden_size).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Compute Metrics
    def compute_metrics(pred_start, pred_end, true_start, true_end):
        exact_match = 0
        f1_scores = []

        for ps, pe, ts, te in zip(pred_start, pred_end, true_start, true_end):
            if ps == ts and pe == te:
                exact_match += 1

            pred_tokens = set(range(ps, pe + 1))
            true_tokens = set(range(ts, te + 1))
            common_tokens = pred_tokens & true_tokens

            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(true_tokens) if true_tokens else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

        em_score = exact_match / len(true_start) if len(true_start) > 0 else 0
        f1_score_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        return em_score, f1_score_avg

    def validation(model, validation_loader):
        model.eval()
        start_preds, end_preds = [], []
        start_targets, end_targets = [], []
        val_loss = 0

        with torch.no_grad():
            for batch in validation_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                logits = model(input_ids, attention_mask)  # Fixed unpacking
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)

                loss = (criterion(start_logits, start_positions) + criterion(end_logits, end_positions)) / 2
                val_loss += loss.item()

                start_preds.extend(torch.argmax(start_logits, dim=1).tolist())
                end_preds.extend(torch.argmax(end_logits, dim=1).tolist())
                start_targets.extend(start_positions.tolist())
                end_targets.extend(end_positions.tolist())

        em, f1 = compute_metrics(start_preds, end_preds, start_targets, end_targets)
        return val_loss / len(validation_loader), em, f1

    # Early stopping parameters
    patience = 2  # Number of epochs to wait before stopping
    no_improve_epochs = 0  # Counter for epochs without improvement
    best_val_f1 = -float("inf")  # Best F1 score

    for epoch in range(5):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                logits = model(input_ids, attention_mask)
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)
                loss = (criterion(start_logits, start_positions) + criterion(end_logits, end_positions)) / 2

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

        print(f"{model_name} Epoch {epoch + 1}, Train Loss: {epoch_loss / len(train_loader):.4f}")

        val_loss, em, f1 = validation(model, validation_loader)
        print(f"{model_name} Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val EM: {em:.4f}, Val F1: {f1:.4f}")

        # Check for improvement
        if f1 > best_val_f1:
            best_val_f1 = f1
            no_improve_epochs = 0  # Reset counter if there's improvement
        else:
            no_improve_epochs += 1  # Increment counter if no improvement

        # Stop training if patience is exceeded
        if no_improve_epochs >= patience:
            print(f"{model_name}: No improvement for {patience} consecutive epochs. Stopping early.")
            break

        # Additional stopping criterion for high F1 score
        if f1 > 0.9:
            print(f"{model_name}: Achieved desired F1 score. Stopping training.")
            break


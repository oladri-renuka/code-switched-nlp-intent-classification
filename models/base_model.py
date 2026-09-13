#!/usr/bin/env python3
"""
XLM-RoBERTa model with supervised contrastive loss for code-switched intent classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel, AutoConfig


class ContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for embeddings.
    Pulls together embeddings with the same label, pushes apart different labels.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate supervised contrastive loss.

        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size] - class labels

        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Calculate pairwise similarities
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature

        # Create label mask (1 where labels match, 0 otherwise)
        label_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()

        # Remove diagonal (positive samples with themselves)
        mask = torch.ones_like(label_mask)
        mask.fill_diagonal_(0)
        label_mask = label_mask * mask

        # Calculate log-softmax similarities
        log_probs = F.log_softmax(similarity_matrix, dim=1)

        # Calculate loss: -log(exp(sim(i,j))/sum_k(exp(sim(i,k))))
        # Sum over all positive pairs
        pos_count = label_mask.sum(dim=1, keepdim=True)
        pos_count = torch.clamp(pos_count, min=1e-6)

        loss = -(label_mask * log_probs).sum(dim=1) / pos_count.squeeze()
        loss = loss.mean()

        return loss


class XLMRobertaForIntentClassification(PreTrainedModel):
    """
    XLM-RoBERTa with contrastive loss for intent classification in code-switched text.
    """

    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)

        self.xlm_roberta = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Classification head
        self.intent_classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Embedding projection for contrastive loss
        self.embedding_projection = nn.Linear(config.hidden_size, 256)

        # Contrastive loss
        self.contrastive_loss_fn = ContrastiveLoss(temperature=0.07)

        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        intent_labels: torch.Tensor = None,
        use_contrastive_loss: bool = True,
    ):
        """
        Forward pass with optional contrastive loss.

        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            token_type_ids: [batch_size, seq_length]
            intent_labels: [batch_size] - intent class labels
            use_contrastive_loss: Whether to include contrastive loss

        Returns:
            Dictionary with logits, loss, embeddings, etc.
        """
        # Get XLM-RoBERTa output
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        # Use [CLS] token representation
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]

        # Apply dropout
        cls_output = self.dropout(cls_output)

        # Get intent classification logits
        intent_logits = self.intent_classifier(cls_output)  # [batch_size, num_labels]

        # Get embeddings for contrastive loss
        embeddings = self.embedding_projection(cls_output)  # [batch_size, 256]

        result = {
            "logits": intent_logits,
            "embeddings": embeddings,
            "sequence_output": sequence_output,
        }

        # Calculate losses if labels provided
        if intent_labels is not None:
            # Cross-entropy loss for classification
            ce_loss = nn.CrossEntropyLoss()(intent_logits, intent_labels)

            # Contrastive loss
            contrastive_loss = 0.0
            if use_contrastive_loss and intent_labels.unique().shape[0] > 1:
                contrastive_loss = self.contrastive_loss_fn(embeddings, intent_labels)

            # Total loss with weighting
            lambda_contrastive = 0.1
            total_loss = ce_loss + lambda_contrastive * contrastive_loss

            result["loss"] = total_loss
            result["ce_loss"] = ce_loss
            result["contrastive_loss"] = contrastive_loss

        return result


def load_model_for_training(
    model_name: str = "xlm-roberta-large",
    num_intent_labels: int = 6,
    device: str = "cuda",
) -> XLMRobertaForIntentClassification:
    """
    Load and initialize model for training.

    Args:
        model_name: HuggingFace model identifier
        num_intent_labels: Number of intent classes
        device: Device to load model on

    Returns:
        Initialized model
    """
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_intent_labels

    model = XLMRobertaForIntentClassification(config)
    model = model.to(device)

    return model


def load_base_model(
    model_name: str = "xlm-roberta-large",
    num_intent_labels: int = 6,
    device: str = "cuda",
) -> PreTrainedModel:
    """
    Load base XLM-RoBERTa model for baseline comparison.

    Args:
        model_name: HuggingFace model identifier
        num_intent_labels: Number of intent classes
        device: Device to load model on

    Returns:
        Base model for classification
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_intent_labels,
    )
    model = model.to(device)

    return model


if __name__ == "__main__":
    # Test model
    print("Testing XLMRobertaForIntentClassification...")

    model = load_model_for_training(num_intent_labels=6)
    print(f"Model loaded: {model}")

    # Create dummy input
    batch_size = 2
    seq_length = 128
    num_labels = 6

    input_ids = torch.randint(0, 50000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    intent_labels = torch.randint(0, num_labels, (batch_size,))

    # Forward pass
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        intent_labels=intent_labels,
        use_contrastive_loss=True,
    )

    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"CE Loss: {output['ce_loss'].item():.4f}")
    print(f"Contrastive Loss: {output['contrastive_loss'].item():.4f}")
    print("✓ Model test passed!")

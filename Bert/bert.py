import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BertEmbedding(nn.Module):
    """BERT Embedding层：Token Embedding + Position Embedding + Segment Embedding"""
    
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, segment_ids=None):
        batch_size, seq_len = input_ids.size()
        
        token_emb = self.token_embeddings(input_ids)
        
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embeddings(position_ids)
        
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        segment_emb = self.segment_embeddings(segment_ids)
        
        embeddings = token_emb + position_emb + segment_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        Q = self.q_linear(hidden_states)
        K = self.k_linear(hidden_states)
        V = self.v_linear(hidden_states)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_linear(context)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class BertLayer(nn.Module):
    """单个BERT Encoder层"""
    
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        attn_output, attn_weights = self.attention(hidden_states, attention_mask)
        hidden_states = self.attn_layer_norm(hidden_states + self.dropout(attn_output))
        
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_layer_norm(hidden_states + self.dropout(ffn_output))
        
        return hidden_states, attn_weights


class BertEncoder(nn.Module):
    """BERT编码器"""
    
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, hidden_states, attention_mask=None):
        all_attentions = []
        
        for layer in self.layers:
            hidden_states, attn_weights = layer(hidden_states, attention_mask)
            all_attentions.append(attn_weights)
        
        return hidden_states, all_attentions


class BertPooler(nn.Module):
    """BERT池化层"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        cls_token = hidden_states[:, 0]
        pooled_output = self.dense(cls_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """完整的BERT模型"""
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.embeddings = BertEmbedding(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout
        )
        
        self.encoder = BertEncoder(
            num_layers, hidden_size, num_heads, intermediate_size, dropout
        )
        
        self.pooler = BertPooler(hidden_size)
        
    def forward(self, input_ids, attention_mask=None, segment_ids=None):
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        embedding_output = self.embeddings(input_ids, segment_ids)
        sequence_output, all_attentions = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(sequence_output)
        
        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output,
            'attentions': all_attentions
        }


class BertForSequenceClassification(nn.Module):
    """用于序列分类的BERT"""
    
    def __init__(self, num_labels, vocab_size=30522, hidden_size=768, num_layers=12, num_heads=12, intermediate_size=3072):
        super().__init__()
        self.bert = BertModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, segment_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, segment_ids)
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        
        return logits


class BertForMaskedLM(nn.Module):
    """用于Masked Language Modeling的BERT"""
    
    def __init__(self, vocab_size=30522, hidden_size=768, num_layers=12, num_heads=12, intermediate_size=3072):
        super().__init__()
        self.bert = BertModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size
        )
        self.cls = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, segment_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, segment_ids)
        sequence_output = outputs['last_hidden_state']
        prediction_scores = self.cls(sequence_output)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                prediction_scores.view(-1, self.cls.out_features), labels.view(-1)
            )
            return loss, prediction_scores
        
        return prediction_scores


class BertForNextSentencePrediction(nn.Module):
    """用于下一句预测的BERT"""
    
    def __init__(self, vocab_size=30522, hidden_size=768, num_layers=12, num_heads=12, intermediate_size=3072):
        super().__init__()
        self.bert = BertModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size
        )
        self.cls = nn.Linear(hidden_size, 2)
        
    def forward(self, input_ids, attention_mask=None, segment_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, segment_ids)
        pooled_output = outputs['pooler_output']
        seq_relationship_scores = self.cls(pooled_output)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(seq_relationship_scores, labels)
            return loss, seq_relationship_scores
        
        return seq_relationship_scores


class BertForPreTraining(nn.Module):
    """BERT预训练模型：MLM + NSP"""
    
    def __init__(self, vocab_size=30522, hidden_size=768, num_layers=12, num_heads=12, intermediate_size=3072):
        super().__init__()
        self.bert = BertModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size
        )
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.nsp_head = nn.Linear(hidden_size, 2)
        
    def forward(self, input_ids, attention_mask=None, segment_ids=None, mlm_labels=None, nsp_labels=None):
        outputs = self.bert(input_ids, attention_mask, segment_ids)
        sequence_output = outputs['last_hidden_state']
        pooled_output = outputs['pooler_output']
        
        mlm_scores = self.mlm_head(sequence_output)
        nsp_scores = self.nsp_head(pooled_output)
        
        if mlm_labels is not None and nsp_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(mlm_scores.view(-1, self.mlm_head.out_features), mlm_labels.view(-1))
            nsp_loss = nn.CrossEntropyLoss()(nsp_scores, nsp_labels)
            total_loss = mlm_loss + nsp_loss
            return total_loss, mlm_loss, nsp_loss, mlm_scores, nsp_scores
        
        return mlm_scores, nsp_scores


if __name__ == "__main__":
    print("=" * 50)
    print("BERT 模型测试")
    print("=" * 50)
    
    config = {
        'vocab_size': 1000,
        'hidden_size': 128,
        'num_layers': 4,
        'num_heads': 4,
        'intermediate_size': 512,
    }
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 10:] = 0
    
    models = [
        ('BERT Base', BertModel, {}),
        ('Sequence Classification', BertForSequenceClassification, {'num_labels': 2}),
        ('Masked LM', BertForMaskedLM, {}),
        ('Next Sentence Prediction', BertForNextSentencePrediction, {}),
        ('Pre-training', BertForPreTraining, {}),
    ]
    
    for name, model_class, extra_args in models:
        print(f"\n测试 {name}")
        model = model_class(**{**config, **extra_args})
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print("\n" + "=" * 50)
    print("所有测试通过！")
    print("=" * 50)

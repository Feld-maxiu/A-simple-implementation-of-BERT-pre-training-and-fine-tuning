"""
BERT训练数据生成器
支持MLM (Masked Language Modeling) 和 NSP (Next Sentence Prediction) 任务
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Tuple


class BertTokenizer:
    """简单的BERT分词器（用于演示，实际使用应加载预训练tokenizer）"""
    
    def __init__(self, vocab_size=30522, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # 特殊token
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4
        
        # 创建简单的词表映射（实际应使用预训练的词表）
        self.word2id = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
        self.id2word = {v: k for k, v in self.word2id.items()}
        
    def encode(self, text: str, max_length=None) -> List[int]:
        """将文本编码为token ids"""
        if max_length is None:
            max_length = self.max_length
            
        # 简单的字符级编码（实际应使用WordPiece/BPE）
        tokens = [self.cls_token_id]
        for char in text[:max_length-2]:
            token_id = self.word2id.get(char, self.unk_token_id)
            if token_id == self.unk_token_id and char not in self.word2id:
                # 为新字符分配id
                if len(self.word2id) < self.vocab_size:
                    token_id = len(self.word2id)
                    self.word2id[char] = token_id
                    self.id2word[token_id] = char
                else:
                    token_id = self.unk_token_id
            tokens.append(token_id)
        tokens.append(self.sep_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """将token ids解码为文本"""
        chars = []
        for tid in token_ids:
            if tid in [self.pad_token_id, self.cls_token_id, self.sep_token_id]:
                continue
            chars.append(self.id2word.get(tid, "[UNK]"))
        return "".join(chars)


class BertPretrainingDataset(Dataset):
    """
    BERT预训练数据集
    生成MLM和NSP任务的训练数据
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: BertTokenizer,
        max_length: int = 128,
        mlm_probability: float = 0.15,
        nsp_probability: float = 0.5
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.nsp_probability = nsp_probability
        
        # 将文本分割成句子
        self.sentences = []
        for text in texts:
            # 简单按标点分割句子
            sents = text.replace("!", ".").replace("?", ".").split(".")
            self.sentences.extend([s.strip() for s in sents if len(s.strip()) > 5])
    
    def __len__(self):
        return len(self.sentences) // 2
    
    def _create_mlm_labels(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        创建MLM标签
        返回: (masked_input_ids, mlm_labels)
        """
        masked_input_ids = input_ids.copy()
        mlm_labels = [-100] * len(input_ids)  # -100表示不参与loss计算
        
        for i in range(1, len(input_ids) - 1):  # 跳过[CLS]和[SEP]
            if random.random() < self.mlm_probability:
                mlm_labels[i] = input_ids[i]  # 记录真实标签
                
                # 80%概率替换为[MASK]
                # 10%概率替换为随机token
                # 10%概率保持不变
                prob = random.random()
                if prob < 0.8:
                    masked_input_ids[i] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    masked_input_ids[i] = random.randint(5, self.tokenizer.vocab_size - 1)
                # else: 保持不变
        
        return masked_input_ids, mlm_labels
    
    def _create_nsp_pair(self, idx: int) -> Tuple[List[int], List[int], int]:
        """
        创建NSP句子对
        返回: (input_ids, segment_ids, is_next_label)
        """
        # 获取第一句
        sent_a = self.sentences[idx * 2]
        
        # 50%概率获取真实的下一句，50%概率获取随机句子
        if random.random() < self.nsp_probability and idx * 2 + 1 < len(self.sentences):
            sent_b = self.sentences[idx * 2 + 1]
            is_next = 1  # 是下一句
        else:
            # 随机选择另一句
            random_idx = random.randint(0, len(self.sentences) - 1)
            sent_b = self.sentences[random_idx]
            is_next = 0  # 不是下一句
        
        # 编码句子
        tokens_a = self.tokenizer.encode(sent_a, max_length=self.max_length // 2)
        tokens_b = self.tokenizer.encode(sent_b, max_length=self.max_length // 2)
        
        # 合并句子对: [CLS] A [SEP] B [SEP]
        input_ids = tokens_a + tokens_b[1:]  # 去掉B的[CLS]
        
        # 创建segment ids: 0表示句子A，1表示句子B
        sep_idx = tokens_a.index(self.tokenizer.sep_token_id)
        segment_ids = [0] * (sep_idx + 1) + [1] * (len(tokens_b) - 1)
        
        # 截断或填充
        input_ids = self._pad_or_truncate(input_ids)
        segment_ids = self._pad_or_truncate(segment_ids)
        
        return input_ids, segment_ids, is_next
    
    def _pad_or_truncate(self, sequence: List[int]) -> List[int]:
        """将序列填充或截断到max_length"""
        if len(sequence) < self.max_length:
            sequence = sequence + [self.tokenizer.pad_token_id] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        return sequence
    
    def __getitem__(self, idx):
        # 创建NSP句子对
        input_ids, segment_ids, is_next = self._create_nsp_pair(idx)
        
        # 创建MLM标签
        masked_input_ids, mlm_labels = self._create_mlm_labels(input_ids)
        
        # 创建attention mask
        attention_mask = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in input_ids]
        
        return {
            'input_ids': torch.tensor(masked_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
            'nsp_labels': torch.tensor(is_next, dtype=torch.long)
        }


class TextClassificationDataset(Dataset):
    """文本分类数据集"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        
        # 填充
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        else:
            input_ids = input_ids[:self.max_length]
        
        # attention mask
        attention_mask = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data_from_file(filepath: str) -> Tuple[List[str], List[int]]:
    """
    从文件加载数据
    文件格式: 每行一条数据，格式为 "label\ttext"
    例如: 1\t这部电影非常好看
    """
    texts = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                label, text = parts
                labels.append(int(label))
                texts.append(text)
    
    return texts, labels


def load_train_data(data_dir: str = './data') -> Tuple[List[str], List[int]]:
    """加载训练数据"""
    train_path = f"{data_dir}/train.txt"
    print(f"加载训练数据: {train_path}")
    texts, labels = load_data_from_file(train_path)
    print(f"  - 总样本数: {len(texts)}")
    print(f"  - 正样本: {sum(labels)}")
    print(f"  - 负样本: {len(labels) - sum(labels)}")
    return texts, labels


def load_test_data(data_dir: str = './data') -> Tuple[List[str], List[int]]:
    """加载测试数据"""
    test_path = f"{data_dir}/test.txt"
    print(f"加载测试数据: {test_path}")
    texts, labels = load_data_from_file(test_path)
    print(f"  - 总样本数: {len(texts)}")
    return texts, labels


def load_pretrain_data(data_dir: str = './data') -> List[str]:
    """
    加载预训练数据
    文件格式: 每行一条文本数据
    例如: 人工智能正在改变我们的生活方式。
    """
    pretrain_path = f"{data_dir}/pretrain.txt"
    print(f"加载预训练数据: {pretrain_path}")
    texts = []
    
    with open(pretrain_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(line)
    
    print(f"  - 总文本数: {len(texts)}")
    print(f"  - 总字符数: {sum(len(text) for text in texts)}")
    return texts


if __name__ == "__main__":
    print("=" * 50)
    print("BERT数据生成器测试")
    print("=" * 50)
    
    tokenizer = BertTokenizer(vocab_size=1000)
    
    # 测试从文件加载数据
    print("\n测试从文件加载数据:")
    try:
        train_texts, train_labels = load_train_data('./data')
        test_texts, test_labels = load_test_data('./data')
        print("  数据加载成功！")
    except FileNotFoundError:
        print("  数据文件不存在，跳过文件加载测试")
        train_texts = ["这是示例文本1。", "这是示例文本2。"]
    
    # 测试预训练数据集
    print("\n测试预训练数据集:")
    pretrain_dataset = BertPretrainingDataset(
        texts=train_texts[:10],
        tokenizer=tokenizer,
        max_length=64,
        mlm_probability=0.15
    )
    print(f"  数据集大小: {len(pretrain_dataset)}")
    
    sample = pretrain_dataset[0]
    print(f"  样本形状: {sample['input_ids'].shape}")
    
    # 测试分类数据集
    print("\n测试分类数据集:")
    cls_dataset = TextClassificationDataset(
        texts=train_texts[:10],
        labels=train_labels[:10],
        tokenizer=tokenizer,
        max_length=32
    )
    print(f"  数据集大小: {len(cls_dataset)}")
    
    print("\n" + "=" * 50)
    print("所有测试通过！")
    print("=" * 50)

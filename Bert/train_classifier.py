"""
BERT文本分类微调脚本
示例：情感分析（正面/负面二分类）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
import os

from bert import BertForSequenceClassification
from data import TextClassificationDataset, BertTokenizer, load_train_data, load_test_data


def train_epoch(model, dataloader, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='BERT Text Classification')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--seq_len', type=int, default=64, help='sequence length')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--intermediate_size', type=int, default=512, help='intermediate size')
    parser.add_argument('--pretrained_path', type=str, default=None, help='pretrained model path')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='save directory')
    parser.add_argument('--fp16', action='store_true', help='use mixed precision training')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建tokenizer
    tokenizer = BertTokenizer(vocab_size=1000)
    
    # 从文件加载数据
    print("\n" + "=" * 50)
    print("加载数据")
    print("=" * 50)
    
    # 加载训练数据
    train_texts, train_labels = load_train_data(data_dir='./data')
    
    # 加载测试数据
    test_texts, test_labels = load_test_data(data_dir='./data')
    
    # 创建数据集
    train_dataset = TextClassificationDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=args.seq_len
    )
    
    test_dataset = TextClassificationDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=args.seq_len
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")
    
    # 创建模型
    model = BertForSequenceClassification(
        num_labels=2,
        vocab_size=1000,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size
    )
    
    # 加载预训练权重（如果提供）
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"加载预训练模型: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        pretrained_state_dict = checkpoint['model_state_dict']
        
        # 只加载bert部分的权重，过滤掉mlm_head和nsp_head，并去掉bert.前缀
        bert_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if key.startswith('bert.'):
                # 去掉bert.前缀，例如 "bert.embeddings.xxx" -> "embeddings.xxx"
                new_key = key[5:]
                bert_state_dict[new_key] = value
        
        # 加载bert权重
        model.bert.load_state_dict(bert_state_dict, strict=False)
        print(f"预训练权重加载成功！加载了{len(bert_state_dict)}个参数")
    
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 混合精度训练
    scaler = GradScaler() if args.fp16 and torch.cuda.is_available() else None
    
    # 训练循环
    print("\n" + "=" * 50)
    print("开始训练")
    print("=" * 50)
    
    best_train_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scaler)
        
        # 测试
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")
        
        # 保存训练集准确率最高的模型
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'config': {
                    'num_labels': 2,
                    'vocab_size': 1000,
                    'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers,
                    'num_heads': args.num_heads,
                    'intermediate_size': args.intermediate_size
                }
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'classifier_best.pt'))
            print(f"保存最佳模型 (train_acc: {train_acc*100:.2f}%)")
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)
    print(f"最佳训练准确率: {best_train_acc*100:.2f}%")
    print(f"\n使用以下命令测试模型:")
    print(f"  python test_classifier.py --model_path {os.path.join(args.save_dir, 'classifier_best.pt')}")


if __name__ == "__main__":
    main()

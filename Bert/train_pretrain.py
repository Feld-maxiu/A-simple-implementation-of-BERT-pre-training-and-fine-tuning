"""
BERT预训练脚本
支持MLM (Masked Language Modeling) 和 NSP (Next Sentence Prediction) 任务
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
import os

from bert import BertForPreTraining
from data import BertPretrainingDataset, BertTokenizer, load_pretrain_data


def train_epoch(model, dataloader, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_mlm_loss = 0
    total_nsp_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        mlm_labels = batch['mlm_labels'].to(device)
        nsp_labels = batch['nsp_labels'].to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                total_loss_batch, mlm_loss, nsp_loss, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    mlm_labels=mlm_labels,
                    nsp_labels=nsp_labels
                )
            
            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss_batch, mlm_loss, nsp_loss, _, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                mlm_labels=mlm_labels,
                nsp_labels=nsp_labels
            )
            
            total_loss_batch.backward()
            optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_mlm_loss += mlm_loss.item()
        total_nsp_loss += nsp_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mlm_loss = total_mlm_loss / len(dataloader)
    avg_nsp_loss = total_nsp_loss / len(dataloader)
    
    return avg_loss, avg_mlm_loss, avg_nsp_loss


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_mlm_loss = 0
    total_nsp_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)
            nsp_labels = batch['nsp_labels'].to(device)
            
            total_loss_batch, mlm_loss, nsp_loss, _, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                mlm_labels=mlm_labels,
                nsp_labels=nsp_labels
            )
            
            total_loss += total_loss_batch.item()
            total_mlm_loss += mlm_loss.item()
            total_nsp_loss += nsp_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mlm_loss = total_mlm_loss / len(dataloader)
    avg_nsp_loss = total_nsp_loss / len(dataloader)
    
    return avg_loss, avg_mlm_loss, avg_nsp_loss


def main():
    parser = argparse.ArgumentParser(description='BERT Pre-training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--seq_len', type=int, default=128, help='sequence length')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--intermediate_size', type=int, default=512, help='intermediate size')
    parser.add_argument('--mlm_prob', type=float, default=0.15, help='MLM probability')
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
    
    # 从文件加载预训练数据
    print("\n" + "=" * 50)
    print("加载数据")
    print("=" * 50)
    train_texts = load_pretrain_data(data_dir='./data')
    
    # 划分训练集和验证集 (90%训练, 10%验证)
    train_size = int(0.9 * len(train_texts))
    val_texts = train_texts[train_size:]
    train_texts = train_texts[:train_size]
    print(f"训练集: {len(train_texts)}条, 验证集: {len(val_texts)}条")
    
    # 创建数据集
    train_dataset = BertPretrainingDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=args.seq_len,
        mlm_probability=args.mlm_prob
    )
    
    val_dataset = BertPretrainingDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_length=args.seq_len,
        mlm_probability=args.mlm_prob
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = BertForPreTraining(
        vocab_size=1000,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size
    )
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 混合精度训练
    scaler = GradScaler() if args.fp16 and torch.cuda.is_available() else None
    
    # 训练循环
    print("\n" + "=" * 50)
    print("开始训练")
    print("=" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_mlm, train_nsp = train_epoch(
            model, train_loader, optimizer, device, scaler
        )
        
        # 验证
        val_loss, val_mlm, val_nsp = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} (MLM: {train_mlm:.4f}, NSP: {train_nsp:.4f})")
        print(f"Val Loss: {val_loss:.4f} (MLM: {val_mlm:.4f}, NSP: {val_nsp:.4f})")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'config': {
                    'vocab_size': 1000,
                    'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers,
                    'num_heads': args.num_heads,
                    'intermediate_size': args.intermediate_size
                }
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"保存最佳模型 (val_loss: {val_loss:.4f})")
        
        # 保存最新模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'config': {
                'vocab_size': 1000,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'intermediate_size': args.intermediate_size
            }
        }
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_model.pt'))
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)
    print(f"最佳验证损失: {best_loss:.4f}")
    print(f"模型保存在: {args.save_dir}")


if __name__ == "__main__":
    main()

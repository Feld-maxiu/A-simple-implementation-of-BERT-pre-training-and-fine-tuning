
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os

from bert import BertForSequenceClassification
from data import TextClassificationDataset, BertTokenizer, load_train_data, load_test_data


def evaluate(model, dataloader, device):
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


def predict(model, tokenizer, text, device, max_length=128):
    model.eval()
    
    input_ids = tokenizer.encode(text, max_length=max_length)
    if len(input_ids) < max_length:
        input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
    else:
        input_ids = input_ids[:max_length]
    
    attention_mask = [1 if tid != tokenizer.pad_token_id else 0 for tid in input_ids]
    
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        prediction = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
    
    return prediction.item(), probs[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='BERT Text Classification Test')
    parser.add_argument('--model_path', type=str, required=True, help='path to trained model')
    parser.add_argument('--seq_len', type=int, default=64, help='sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    tokenizer = BertTokenizer(vocab_size=1000)
    
    print("\n" + "=" * 50)
    print("加载数据")
    print("=" * 50)
    
    test_texts, test_labels = load_test_data(data_dir='./data')
    
    test_dataset = TextClassificationDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=args.seq_len
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"测试集大小: {len(test_dataset)}")
    
    print("\n" + "=" * 50)
    print("加载模型")
    print("=" * 50)
    
    model = BertForSequenceClassification(
        num_labels=2,
        vocab_size=1000,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        intermediate_size=512
    )
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"模型加载成功: {args.model_path}")
    if 'epoch' in checkpoint:
        print(f"训练轮数: {checkpoint['epoch']}")
    if 'train_acc' in checkpoint:
        print(f"训练集准确率: {checkpoint['train_acc']*100:.2f}%")
    if 'test_acc' in checkpoint:
        print(f"测试集准确率: {checkpoint['test_acc']*100:.2f}%")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print("\n" + "=" * 50)
    print("测试集评估")
    print("=" * 50)
    
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"测试集 Loss: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc*100:.2f}%")
    
    print("\n" + "=" * 50)
    print("详细预测结果")
    print("=" * 50)
    
    for i, text in enumerate(test_texts):
        true_label = test_labels[i]
        pred, probs = predict(model, tokenizer, text, device, args.seq_len)
        sentiment = "正面" if pred == 1 else "负面"
        true_sentiment = "正面" if true_label == 1 else "负面"
        correct = "✓" if pred == true_label else "✗"
        
        print(f"\n[{i+1}] {text}")
        print(f"    真实标签: {true_sentiment}")
        print(f"    预测标签: {sentiment} {correct}")
        print(f"    正面概率: {probs[1]*100:.2f}%, 负面概率: {probs[0]*100:.2f}%")
    
    print("\n" + "=" * 50)
    print("统计信息")
    print("=" * 50)
    
    predictions = []
    for text in test_texts:
        pred, _ = predict(model, tokenizer, text, device, args.seq_len)
        predictions.append(pred)
    
    correct_count = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
    total_count = len(test_labels)
    
    print(f"总样本数: {total_count}")
    print(f"正确预测: {correct_count}")
    print(f"错误预测: {total_count - correct_count}")
    print(f"准确率: {correct_count/total_count*100:.2f}%")
    
    print("\n" + "=" * 50)
    print("额外测试")
    print("=" * 50)
    
    extra_sentences = [
        "这部电影真的太好看了，强烈推荐给大家！",
        "完全看不下去，太无聊了，浪费时间。",
        "演员的表演非常出色，剧情也很吸引人。",
        "特效假得离谱，像五毛钱制作。",
    ]
    
    for i, text in enumerate(extra_sentences):
        pred, probs = predict(model, tokenizer, text, device, args.seq_len)
        sentiment = "正面" if pred == 1 else "负面"
        print(f"\n[{i+1}] {text}")
        print(f"    预测: {sentiment}")
        print(f"    正面概率: {probs[1]*100:.2f}%, 负面概率: {probs[0]*100:.2f}%")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

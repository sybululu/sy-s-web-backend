"""
模型诊断脚本 - 检查 checkpoint 结构
"""
import torch
import os

# 设置 HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "sybululu/bert-moe"

def diagnose_checkpoint():
    from huggingface_hub import hf_hub_download

    print("=" * 60)
    print("模型诊断报告")
    print("=" * 60)

    # 1. 下载 checkpoint
    print("\n[1] 下载 checkpoint...")
    try:
        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="multi_classification_bertmoe.ckpt",
            token=HF_TOKEN or None
        )
        print(f"✓ 下载成功: {ckpt_path}")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return

    # 2. 加载并分析结构
    print("\n[2] 分析 checkpoint 结构...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    print(f"Checkpoint 类型: {type(checkpoint)}")
    print(f"顶层键: {list(checkpoint.keys())[:20]}")  # 只显示前20个

    # 3. 分析 state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    print(f"\n[3] State Dict 分析:")
    print(f"  - 总参数量: {len(state_dict)}")
    
    # 打印所有键名
    print(f"\n  - 所有键名:")
    for i, k in enumerate(sorted(state_dict.keys())):
        shape = state_dict[k].shape if hasattr(state_dict[k], 'shape') else type(state_dict[k])
        print(f"    {i+1}. {k} -> {shape}")
    
    # 4. 检查配置
    print("\n[4] Config 分析:")
    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"  - Config 内容: {config}")
    else:
        print("  - 无 config")
    
    # 5. 测试键名清理逻辑
    print("\n[5] 键名清理测试:")
    test_keys = list(state_dict.keys())[:5]
    for k in test_keys:
        cleaned = k.replace("model.", "").replace("bert.", "")
        print(f"  {k}")
        print(f"    -> {cleaned}")
        if k != cleaned:
            print(f"    ⚠️ 键名被修改!")

    # 6. 检查分类头权重
    print("\n[6] 检查分类头 (classifier.dense.weight):")
    classifier_keys = [k for k in state_dict.keys() if 'classifier' in k.lower()]
    if classifier_keys:
        for k in classifier_keys:
            tensor = state_dict[k]
            print(f"  {k}: shape={tensor.shape}, mean={tensor.mean():.6f}, std={tensor.std():.6f}")
            print(f"    min={tensor.min():.6f}, max={tensor.max():.6f}")
    else:
        print("  ⚠️ 未找到分类头权重!")
    
    # 7. 检查 BERT 主干权重
    print("\n[7] 检查 BERT 主干权重 (embeddings):")
    embedding_keys = [k for k in state_dict.keys() if 'embedding' in k.lower()]
    if embedding_keys:
        for k in embedding_keys[:3]:
            tensor = state_dict[k]
            print(f"  {k}: shape={tensor.shape}, mean={tensor.mean():.6f}, std={tensor.std():.6f}")
    else:
        print("  ⚠️ 未找到 embedding 权重!")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_checkpoint()

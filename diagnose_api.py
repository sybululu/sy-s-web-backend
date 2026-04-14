"""
模型诊断 API - 用于排查权重加载问题
部署在 HuggingFace Spaces 后访问 /api/v1/diagnose 模型
"""
import torch
import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["diagnose"])


@router.post("/diagnose/generate")
async def test_mt5_generate(test_input: str = "这是一个隐私政策条款"):
    """
    测试 mT5 生成模型
    """
    from app import model_status
    
    result = {
        "generator_loaded": model_status.generator_loaded,
        "tokenizer_type": type(model_status.tokenizer_generator).__name__ if model_status.tokenizer_generator else None,
        "test_input": test_input
    }
    
    if not model_status.generator_loaded:
        result["status"] = "error"
        result["error"] = "Generator model not loaded"
        return result
    
    try:
        # 测试不同的 prompt 格式
        prompts = [
            f"rewrite: 把'{test_input}'改写得更规范",
            f"把'{test_input}'改写成合规版本",
            test_input,
            f"rewrite: {test_input}"
        ]
        
        results = []
        for i, prompt in enumerate(prompts):
            inputs = model_status.tokenizer_generator(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )
            
            with torch.no_grad():
                outputs = model_status.model_generator.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            generated = model_status.tokenizer_generator.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                "prompt_index": i,
                "prompt": prompt,
                "generated": generated,
                "generated_length": len(generated)
            })
            
            logger.info(f"Prompt {i}: {prompt[:50]}... -> {generated[:100]}...")
        
        result["status"] = "success"
        result["results"] = results
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result

@router.get("/diagnose/model")
async def diagnose_model():
    """
    诊断模型 checkpoint 结构
    返回键名、分类头权重统计等信息
    """
    from huggingface_hub import hf_hub_download
    
    REPO_ID = "sybululu/bert-moe"
    
    result = {
        "status": "running",
        "steps": []
    }
    
    # Step 1: 下载 checkpoint
    try:
        result["steps"].append({"step": "download", "status": "starting"})
        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="multi_classification_bertmoe.ckpt",
            token=None
        )
        result["steps"].append({
            "step": "download", 
            "status": "success",
            "path": str(ckpt_path)
        })
    except Exception as e:
        result["steps"].append({
            "step": "download", 
            "status": "error",
            "error": str(e)
        })
        result["status"] = "failed"
        return result
    
    # Step 2: 加载并分析结构
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        result["checkpoint_keys"] = list(checkpoint.keys())
        result["steps"].append({
            "step": "load_checkpoint", 
            "status": "success"
        })
    except Exception as e:
        result["steps"].append({
            "step": "load_checkpoint", 
            "status": "error",
            "error": str(e)
        })
        result["status"] = "failed"
        return result
    
    # Step 3: 分析 state_dict
    try:
        if "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
        else:
            sd = checkpoint
        
        result["state_dict_count"] = len(sd)
        result["state_dict_keys"] = sorted(sd.keys())
        result["steps"].append({
            "step": "extract_state_dict", 
            "status": "success",
            "count": len(sd)
        })
    except Exception as e:
        result["steps"].append({
            "step": "extract_state_dict", 
            "status": "error",
            "error": str(e)
        })
        result["status"] = "failed"
        return result
    
    # Step 4: 分析分类头
    try:
        classifier_keys = [k for k in sd.keys() if 'classifier' in k.lower()]
        result["classifier_keys"] = classifier_keys
        result["classifier_analysis"] = []
        
        for k in classifier_keys:
            t = sd[k]
            result["classifier_analysis"].append({
                "key": k,
                "shape": list(t.shape),
                "mean": float(t.mean()),
                "std": float(t.std()),
                "min": float(t.min()),
                "max": float(t.max()),
                "is_zero": bool(abs(t.mean()) < 1e-6),
                "is_random": bool(0.01 < t.std() < 1.0)
            })
        
        result["steps"].append({
            "step": "analyze_classifier", 
            "status": "success",
            "count": len(classifier_keys)
        })
    except Exception as e:
        result["steps"].append({
            "step": "analyze_classifier", 
            "status": "error",
            "error": str(e)
        })
    
    # Step 5: 检查键名前缀
    try:
        prefixes = {}
        for k in sd.keys():
            prefix = k.split('.')[0] if '.' in k else k
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        result["key_prefixes"] = prefixes
        result["steps"].append({
            "step": "analyze_prefixes", 
            "status": "success"
        })
    except Exception as e:
        result["steps"].append({
            "step": "analyze_prefixes", 
            "status": "error",
            "error": str(e)
        })
    
    # Step 6: 检查 BERT/RoBERTa 主干权重
    try:
        embedding_keys = [k for k in sd.keys() if 'embedding' in k.lower()]
        result["embedding_keys"] = embedding_keys
        
        if embedding_keys:
            t = sd[embedding_keys[0]]
            result["embedding_sample"] = {
                "key": embedding_keys[0],
                "shape": list(t.shape),
                "mean": float(t.mean()),
                "std": float(t.std())
            }
        
        result["steps"].append({
            "step": "analyze_backbone", 
            "status": "success"
        })
    except Exception as e:
        result["steps"].append({
            "step": "analyze_backbone", 
            "status": "error",
            "error": str(e)
        })
    
    result["status"] = "completed"
    return result


@router.post("/diagnose/load")
async def test_model_load(test_text: str = "这是一个测试文本"):
    """
    测试模型加载和预测
    返回原始 logits 用于诊断
    """
    # 延迟导入避免循环依赖
    from app import model_status
    
    result = {
        "model_loaded": model_status.classifier_loaded,
        "tokenizer_type": type(model_status.tokenizer_classifier).__name__ if model_status.tokenizer_classifier else None,
        "model_type": type(model_status.model_classifier).__name__ if model_status.model_classifier else None,
        "test_text": test_text
    }
    
    if not model_status.classifier_loaded:
        result["status"] = "error"
        result["error"] = "Model not loaded"
        return result
    
    try:
        # 原始 logits
        inputs = model_status.tokenizer_classifier(
            test_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model_status.model_classifier(**inputs)
            logits = outputs.logits.squeeze()
            probs = torch.sigmoid(logits).tolist()
        
        result["logits"] = logits.tolist()
        result["probabilities"] = probs
        result["predicted_class"] = int(probs.index(max(probs)))
        result["predicted_prob"] = float(max(probs))
        
        # 分析 logits 范围
        result["logits_analysis"] = {
            "mean": float(logits.mean()),
            "std": float(logits.std()),
            "min": float(logits.min()),
            "max": float(logits.max()),
            "all_similar": float(logits.std()) < 0.1  # std 太小说明模型没学到东西
        }
        
        result["status"] = "success"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

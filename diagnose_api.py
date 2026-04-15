"""
模型诊断 API - 用于排查权重加载问题
部署在 HuggingFace Spaces 后访问 /api/v1/diagnose 模型
"""
import torch
import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["diagnose"])

# mT5 vocab 兼容处理
_tokenizer_vocab_size = None

def safe_decode(tokenizer, token_ids, skip_special_tokens=True):
    """安全解码，忽略超出 tokenizer 词汇表的 token ID"""
    global _tokenizer_vocab_size
    if _tokenizer_vocab_size is None:
        _tokenizer_vocab_size = len(tokenizer)
    filtered_ids = [tid for tid in token_ids if tid < _tokenizer_vocab_size]
    if not filtered_ids:
        return ""
    return tokenizer.decode(filtered_ids, skip_special_tokens=skip_special_tokens)


@router.get("/diagnose/generator/weights")
async def diagnose_generator_weights():
    """诊断 mT5 生成模型权重和 tokenizer"""
    from app import model_status
    
    result = {
        "generator_loaded": model_status.generator_loaded
    }
    
    if not model_status.generator_loaded:
        result["status"] = "error"
        result["error"] = "Generator not loaded"
        return result
    
    try:
        model = model_status.model_generator
        tokenizer = model_status.tokenizer_generator
        
        # Tokenizer 信息
        result["tokenizer"] = {
            "type": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "model_max_length": tokenizer.model_max_length
        }
        
        # 检查是否有 spiece model
        if hasattr(tokenizer, 'sp_model') and tokenizer.sp_model:
            result["tokenizer"]["has_spiece_model"] = True
        else:
            result["tokenizer"]["has_spiece_model"] = False
        
        # 模型 vocab 大小
        result["model_config"] = {
            "vocab_size": model.config.vocab_size,
            "d_model": model.config.d_model,
            "d_ff": model.config.d_ff
        }
        
        # 检查是否匹配
        result["vocab_match"] = len(tokenizer) == model.config.vocab_size
        
        # 检查 decoder 权重
        decoder_weights = []
        for name, param in model.named_parameters():
            if 'decoder' in name and 'weight' in name:
                decoder_weights.append({
                    "name": name,
                    "shape": list(param.shape),
                    "mean": float(param.mean()),
                    "std": float(param.std())
                })
        result["decoder_weights_count"] = len(decoder_weights)
        result["decoder_weights_sample"] = decoder_weights[:2]
        
        # 检查 lm_head 权重
        lm_head_weights = []
        for name, param in model.named_parameters():
            if 'lm_head' in name or ('output' in name and 'weight' in name):
                lm_head_weights.append({
                    "name": name,
                    "shape": list(param.shape),
                    "mean": float(param.mean()),
                    "std": float(param.std())
                })
        result["lm_head_weights_count"] = len(lm_head_weights)
        result["lm_head_weights_sample"] = lm_head_weights[:2]
        
        result["status"] = "success"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


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
            
            generated = safe_decode(model_status.tokenizer_generator, outputs[0], skip_special_tokens=True)
            
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


@router.post("/diagnose/test-original-mt5")
async def test_original_mt5(test_input: str = "我们收集您的个人信息用于改善服务"):
    """
    测试原始 mT5-small 模型（不带微调权重）
    用于对比：确认是模型问题还是微调权重问题
    """
    try:
        from transformers import MT5ForConditionalGeneration, T5Tokenizer
        
        logger.info("加载原始 mT5-small 模型...")
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        tokenizer = T5Tokenizer.from_pretrained("google/mt5-small", legacy=True)
        model.eval()
        
        # 测试多种 prompt 格式
        prompts = [
            f"rewrite: {test_input}",
            f"将以下文本改写得更规范：{test_input}",
            f"refactor: {test_input}",
            test_input,
            f"summarize: {test_input}",
        ]
        
        results = []
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                "prompt": prompt,
                "generated": generated,
                "length": len(generated)
            })
            logger.info(f"[原始模型] Prompt {i}: {prompt[:30]}... -> {generated[:50]}...")
        
        return {
            "status": "success",
            "model": "google/mt5-small",
            "tokenizer_vocab_size": len(tokenizer),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"原始模型测试失败: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


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
        "repo_id": REPO_ID
    }
    
    try:
        # 下载 checkpoint
        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="rewrite_mT5_small.ckpt",
            token=None
        )
        result["checkpoint_path"] = ckpt_path
        
        # 加载 checkpoint
        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # 提取 state_dict
        if isinstance(raw_ckpt, dict):
            if "state_dict" in raw_ckpt:
                state_dict = raw_ckpt["state_dict"]
            elif "model_state_dict" in raw_ckpt:
                state_dict = raw_ckpt["model_state_dict"]
            else:
                state_dict = raw_ckpt
            if "config" in raw_ckpt:
                result["has_config"] = True
                result["config"] = raw_ckpt["config"]
            else:
                result["has_config"] = False
        else:
            state_dict = raw_ckpt
            result["has_config"] = False
        
        result["keys_count"] = len(state_dict)
        result["key_samples"] = list(state_dict.keys())[:20]
        
        # 检查 lm_head 权重
        lm_head_keys = [k for k in state_dict.keys() if 'lm_head' in k]
        result["lm_head_keys"] = lm_head_keys
        
        for key in lm_head_keys:
            tensor = state_dict[key]
            result[f"lm_head_shape_{key}"] = list(tensor.shape)
        
        result["status"] = "success"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result

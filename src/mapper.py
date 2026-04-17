"""
CAPP-130 11类 → 论文12类 映射器（多标签版）
==========================================

模型输出11类概率 → 取概率最高的类别 → 多标签映射到对应12类

映射关系（覆盖全部12类）：
0: 数据收集       → I1(过度收集) + I4(收集范围超出)
1: 权限获取       → I3(未获得明示同意)
2: 共享转让       → I5(第三方范围) + I6(单独授权) + I7(共享用途)
3: 使用           → I2(未说明目的)
4: 存储方式       → I8(留存期限)
5: 安全措施/销毁  → I9(销毁机制)
6: 特殊人群       → I3(未获得明示同意)
7: 权限管理       → I10(用户权利) + I12(响应时限)
8: 联系方式       → I11(行使途径)
9: 政策变更       → I11(行使途径)
10: 停止运营      → I9(销毁机制)

12类全覆盖检查：
I1✅ I2✅ I3✅ I4✅ I5✅ I6✅ I7✅ I8✅ I9✅ I10✅ I11✅ I12✅
"""
from typing import List, Union


# 11类 → 12类 多标签映射表（一个idx可映射到多个ID）
ID_MAPPING = {
    0: ["I1", "I4"],            # 数据收集 → 过度收集 + 范围超出
    1: ["I3"],                  # 权限获取 → 未获得明示同意
    2: ["I5", "I6", "I7"],      # 共享转让 → 第三方范围+单独授权+共享用途
    3: ["I2"],                  # 使用 → 未说明目的
    4: ["I8"],                  # 存储方式 → 留存期限
    5: ["I9"],                  # 安全措施 → 销毁机制
    6: ["I3"],                  # 特殊人群 → 未获得明示同意
    7: ["I10", "I12"],          # 权限管理 → 用户权利+响应时限
    8: ["I11"],                 # 联系方式 → 行使途径
    9: ["I11"],                 # 政策变更 → 行使途径
    10: ["I9"],                 # 停止运营 → 销毁机制
}

# 12类中文名称
VIOLATION_NAMES = {
    "I1": "过度收集敏感数据",
    "I2": "未说明收集目的",
    "I3": "未获得明示同意",
    "I4": "收集范围超出服务需求",
    "I5": "未明确第三方共享范围",
    "I6": "未获得单独共享授权",
    "I7": "未明确共享数据用途",
    "I8": "未明确留存期限",
    "I9": "未说明数据销毁机制",
    "I10": "未明确用户权利范围",
    "I11": "未提供便捷权利行使途径",
    "I12": "未明确权利响应时限",
}


def map_to_12_classes(probs: List[float], confidence: float = None) -> List[str]:
    """
    11类 → 12类 多标签映射
    
    Args:
        probs: 11类概率向量 [p0, p1, ..., p10]
        confidence: 可选的置信度（logits差值），用于替代概率阈值
        
    Returns:
        违规ID列表（可能包含多个ID）
    """
    if not probs:
        return []
    
    # 获取概率最高的类别索引
    max_idx = probs.index(max(probs))
    max_prob = probs[max_idx]
    
    # 使用 confidence 和概率双重限制
    if confidence is not None:
        CONFIDENCE_THRESHOLD = 3.0
        PROB_THRESHOLD = 0.6
        if confidence < CONFIDENCE_THRESHOLD or max_prob < PROB_THRESHOLD:
            return []
    
    # 多标签映射：一个 idx 可对应多个 violation ID
    target = ID_MAPPING.get(max_idx)
    if target:
        if isinstance(target, list):
            return target
        return [target]
    
    return []


def get_violation_name(violation_id: str) -> str:
    """获取违规类型中文名称"""
    return VIOLATION_NAMES.get(violation_id, violation_id)


__all__ = ["map_to_12_classes", "get_violation_name", "ID_MAPPING", "VIOLATION_NAMES"]

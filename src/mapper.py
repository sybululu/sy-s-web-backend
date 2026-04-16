"""
CAPP-130 11类 → 论文12类 映射器（直接映射版）
==========================================

模型输出11类概率 → 取概率最高的类别 → 直接映射到对应12类

映射关系：
0: 数据收集 → I1(过度收集敏感数据)
1: 权限获取 → I3(未获得明示同意)
2: 共享转让 → I5(未明确第三方共享范围)
3: 使用     → I2(未说明收集/使用目的)
4: 存储方式 → I8(未明确留存期限)
5: 安全措施 → I9(未说明数据安全机制)
6: 特殊人群 → I3(未获得明示同意)
7: 权限管理 → I10(未明确用户权利范围)
8: 联系方式 → I11(未提供便捷权利行使途径)
9: 政策变更 → I11(未提供便捷权利行使途径)
10: 停止运营 → I9(未说明数据销毁机制)
"""
from typing import List


# 11类 → 12类 直接映射表
ID_MAPPING = {
    0: "I1",   # 数据收集 → 过度收集敏感数据
    1: "I3",   # 权限获取 → 未获得明示同意
    2: "I5",   # 共享转让 → 未明确第三方共享范围
    3: "I2",   # 使用 → 未说明使用目的
    4: "I8",   # 存储方式 → 未明确留存期限
    5: "I9",   # 安全措施 → 未说明数据安全机制
    6: "I3",   # 特殊人群 → 未获得明示同意
    7: "I10",  # 权限管理 → 未明确用户权利范围
    8: "I11",  # 联系方式 → 未提供便捷权利行使途径
    9: "I11",  # 政策变更 → 未提供便捷权利行使途径
    10: "I9",  # 停止运营 → 未说明数据销毁机制
}

# 12类中文名称
VIOLATION_NAMES = {
    "I1": "过度收集敏感数据",
    "I2": "未说明收集/使用目的",
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
    11类 → 12类 直接映射
    
    Args:
        probs: 11类概率向量 [p0, p1, ..., p10]
        confidence: 可选的置信度（logits差值），用于替代概率阈值
        
    Returns:
        违规ID列表（返回概率最高的类别）
    """
    if not probs:
        return []
    
    # 获取概率最高的类别索引
    max_idx = probs.index(max(probs))
    max_prob = probs[max_idx]
    
    # 优先使用 confidence 阈值，如果没有则使用概率阈值
    if confidence is not None:
        # confidence 是 max(logits) - mean(logits)，通常在 1.5-4.0 之间
        # 使用 1.8 作为阈值（略低于平均值）
        THRESHOLD = 1.8
        if confidence < THRESHOLD:
            return []
    else:
        # 降级使用 softmax 概率（但这不太准确）
        SOFTMAX_THRESHOLD = 0.40
        if max_prob < SOFTMAX_THRESHOLD:
            return []
    
    # 直接映射
    target = ID_MAPPING.get(max_idx)
    if target:
        return [target]
    
    return []


def get_violation_name(violation_id: str) -> str:
    """获取违规类型中文名称"""
    return VIOLATION_NAMES.get(violation_id, violation_id)


__all__ = ["map_to_12_classes", "get_violation_name", "ID_MAPPING", "VIOLATION_NAMES"]

"""
12类违规指标体系 — 唯一定义源 (Single Source of Truth)
==========================================

前后端共享的违规类型配置。任何修改只需改此一处。

权重总和 = 1.00（与论文表3-1一致）
风险等级判定：高(>=0.12) / 中(0.08-0.11) / 低(<0.08)
"""

from typing import List, Dict, Optional


# ==========================================
# 核心指标定义（唯一权威数据源）
# ==========================================
INDICATORS = {
    "过度收集敏感数据": {
        "weight": 0.15,
        "legal_basis": "《个人信息保护法》第六条'最小必要'原则及第二十九条",
        "id": "I1",
        "risk_level": "high",
        "hint": "涉及收集个人信息，必须遵守最小必要原则，只能收集与服务直接相关的个人信息，禁止收集与服务无关的敏感信息。",
    },
    "未说明收集目的": {
        "weight": 0.12,
        "legal_basis": "《个人信息保护法》第十七条",
        "id": "I2",
        "risk_level": "high",
        "hint": "必须明确说明每项个人信息收集的具体目的和用途，不能使用模糊表述。",
    },
    "未获得明示同意": {
        "weight": 0.15,
        "legal_basis": "《个人信息保护法》第十四条",
        "id": "I3",
        "risk_level": "high",
        "hint": "涉及处理个人信息必须获得用户明确、知情、自愿的同意，不能捆绑授权。",
    },
    "收集范围超出服务需求": {
        "weight": 0.10,
        "legal_basis": "《个人信息保护法》第六条",
        "id": "I4",
        "risk_level": "medium",
        "hint": "收集范围不得超过实现处理目的的最小必要范围。",
    },
    "未明确第三方共享范围": {
        "weight": 0.08,
        "legal_basis": "《个人信息保护法》第二十三条",
        "id": "I5",
        "risk_level": "medium",
        "hint": "向第三方共享时必须明确说明接收方类型、共享目的、数据类型，禁止无限制共享。",
    },
    "未获得单独共享授权": {
        "weight": 0.12,
        "legal_basis": "《个人信息保护法》第二十三条",
        "id": "I6",
        "risk_level": "high",
        "hint": "向第三方提供个人信息必须单独取得用户明示同意。",
    },
    "未明确共享数据用途": {
        "weight": 0.08,
        "legal_basis": "《个人信息保护法》第二十三条及GDPR第四十六条",
        "id": "I7",
        "risk_level": "medium",
        "hint": "必须明确说明第三方使用数据的目的和范围。",
    },
    "未明确留存期限": {
        "weight": 0.05,
        "legal_basis": "《个人信息保护法》第十九条",
        "id": "I8",
        "risk_level": "low",
        "hint": "必须明确数据存储期限，期限届满应予以删除或匿名化。",
    },
    "未说明数据销毁机制": {
        "weight": 0.05,
        "legal_basis": "《个人信息保护法》第四十七条",
        "id": "I9",
        "risk_level": "low",
        "hint": "必须说明数据销毁机制，承诺在约定保存期限届满后主动删除或匿名化处理。",
    },
    "未明确用户权利范围": {
        "weight": 0.05,
        "legal_basis": "《个人信息保护法》第四十四至四十八条",
        "id": "I10",
        "risk_level": "low",
        "hint": "必须明确列举用户享有的各项权利及行使方式。",
    },
    "未提供便捷权利行使途径": {
        "weight": 0.03,
        "legal_basis": "《个人信息保护法》第五十条",
        "id": "I11",
        "risk_level": "low",
        "hint": "必须提供便捷的渠道供用户行使权利，渠道必须易于发现和操作。",
    },
    "未明确权利响应时限": {
        "weight": 0.02,
        "legal_basis": "《个人信息安全规范》GB/T 35273-2020",
        "id": "I12",
        "risk_level": "low",
        "hint": "必须明确权利响应时限，承诺在法定期限内处理用户请求。",
    },
}

# ==========================================
# 快捷查询字典（从 INDICATORS 自动生成）
# ==========================================

# ID → 中文名称 (如 "I1" → "过度收集敏感数据")
ID_TO_INDICATOR: Dict[str, str] = {v["id"]: k for k, v in INDICATORS.items()}

# ID → 整改提示 (如 "I1" → "涉及收集个人信息...")
ID_TO_HINT: Dict[str, str] = {v["id"]: v["hint"] for k, v in INDICATORS.items()}

# ID → 风险等级 (如 "I1" → "high")
ID_TO_RISK_LEVEL: Dict[str, str] = {v["id"]: v["risk_level"] for k, v in INDICATORS.items()}

# 所有指标名称列表（保持顺序）
INDICATOR_KEYS: List[str] = list(INDICATORS.keys())


# ==========================================
# 前端工具函数
# ==========================================

def get_risk_level(violation_id: str) -> str:
    """根据违规ID返回风险等级 (high/medium/low)"""
    return ID_TO_RISK_LEVEL.get(violation_id, "low")


def get_hint(violation_id: str) -> str:
    """根据违规ID返回整改提示"""
    return ID_TO_HINT.get(violation_id, "必须符合《个人信息保护法》相关要求。")


def get_legal_basis(violation_id: str) -> str:
    """根据违规ID返回法律依据"""
    indicator_name = ID_TO_INDICATOR.get(violation_id, "")
    if not indicator_name:
        return "《个人信息保护法》"
    return INDICATORS.get(indicator_name, {}).get("legal_basis", "《个人信息保护法》")


def to_frontend_list() -> List[Dict]:
    """生成前端筛选器需要的列表格式
    
    Returns:
        [{"id": "I1", "name": "过度收集敏感数据", "risk": "high"}, ...]
    """
    return [
        {"id": v["id"], "name": k, "risk": v["risk_level"]}
        for k, v in INDICATORS.items()
    ]


__all__ = [
    "INDICATORS", "ID_TO_INDICATOR", "ID_TO_HINT", "ID_TO_RISK_LEVEL",
    "INDICATOR_KEYS", "get_risk_level", "get_hint", "get_legal_basis",
    "to_frontend_list",
]

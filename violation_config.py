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

# ==========================================
# 双模式 Prompt 模板（预留：rectify_snippet 未来可重构为调用此模板）
# 当前 app.py/rectify_snippet 使用内联 prompt 构建，以下模板暂未启用
# ==========================================

SUMMARY_SYSTEM_PROMPT = """你是一位贴心的隐私保护顾问，擅长将晦涩的法律条款"翻译"成普通人能听懂的大白话。

你的任务是：帮助普通用户理解隐私政策条款中隐藏的风险，用最直白的语言告诉用户：
1. 这段话到底在说什么（翻译成人话）
2. 对用户有什么潜在风险或影响
3. 用户应该注意什么、可以怎么做

写作要求：
- 严禁使用任何法律术语（如"去标识化""最小必要原则""明示同意""信息控制者"等）
- 用生活化的比喻和日常语言，像给朋友解释一样
- 控制在 3-5 句话以内，每句不超过 20 个字
- 语气友好但不危言耸听，客观陈述事实
- 最后给出一句简短的行动建议"""

REWRITE_SYSTEM_PROMPT = """你是一位资深的隐私合规专家，精通《个人信息保护法》《数据安全法》《网络安全法》及 GB/T 35273 等法规标准。

你的任务是：根据提供的[法律依据]和[整改要求]，将[原句]重写为符合中国法律法规的合规文本。

重写必须满足以下硬性约束：
1. **最小必要原则**：只收集/使用实现目的所必需的最少信息
2. **目的明确**：每项数据处理活动必须有清晰、具体的合法目的
3. **知情同意**：涉及敏感个人信息须取得单独明示同意
4. **用户权利**：明确告知用户享有查阅、复制、更正、删除等权利
5. **第三方规范**：共享/委托/跨境须说明接收方类型、目的、方式
6. **留存期限**：明确存储期限及届满后的处理方式

语言风格：
- 使用标准的隐私政策表述范式（如"仅为...之目的""我们将在...范围内"）
- 可直接替换到 App 隐私政策的对应位置
- 避免模糊措辞（如"可能""必要时"），改用确定性表述"""


def build_summary_user_prompt(
    original_text: str,
    violation_name: str,
    violation_hint: str,
    legal_context: str = ""
) -> str:
    """
    构建摘要模式的 user prompt
    
    RAG 法律依据作为"背景知识"辅助理解，不直接暴露给用户
    """
    # 从法律依据中提炼通俗化的风险点
    risk_translation = _translate_legal_to_plain(legal_context, violation_hint)
    
    return f"""【原始条款】
{original_text}

【问题类型】
{violation_name}

【这条条款的问题】
{risk_translation}

请用大白话告诉用户这段话意味着什么，以及用户需要注意什么。"""


def build_rewrite_user_prompt(
    original_text: str,
    violation_name: str,
    violation_hint: str,
    legal_keywords: str
) -> str:
    """
    构建专业改写模式的 user prompt
    
    RAG 法律依据作为"合规标准"强力注入
    """
    return f"""【法律依据摘要】
{legal_keywords}

【整改要求】
{violation_hint}

【违规类型】
{violation_name}

【原句】
{original_text}

请根据以上要求，输出可直接使用的合规改写文本。只输出改写后的文本，不要加任何前缀或解释。"""


def _translate_legal_to_plain(legal_context: str, hint: str) -> str:
    """
    将法律术语转化为通俗问题描述（供摘要模式使用）
    
    这是摘要模式的核心：把 "未获得明示同意" 翻译为
    "App 在你没真正同意的情况下就收集了你的信息"
    """
    # 常见法律术语 → 大白话映射表
    term_map = {
        "最小必要": "App 收集的信息可能比它实际需要的多得多",
        "明示同意": "App 在你没有真正点头同意的情况下就做了这件事",
        "单独同意": "App 把各种权限打包在一起让你一次性同意，没给你逐项选择的机会",
        "告知": "App 没有清楚地告诉你它要用你的信息做什么",
        "第三方": "App 可能会把你的信息交给其他公司，但没告诉你给了谁",
        "共享": "你的信息可能被 App 传到了你不知道的地方",
        "匿名化": "App 说会'脱敏'你的信息，但没有说清楚具体怎么做的",
        "删除": "你想让 App 删掉你的信息时，可能找不到入口或者删不掉",
        "留存期限": "App 会一直保存你的信息，但没告诉你保存多久",
        "销毁": "App 没有承诺什么时候会彻底清除你的信息",
        "用户权利": "法律规定你有很多权利（比如查看、删除自己的信息），但 App 没告诉你",
        "响应时限": "你向 App 提出请求后，它可能拖很久都不处理",
        "跨境": "你的信息可能被传到了国外，但你并不知情",
        "敏感信息": "App 可能收集了你的私密信息（如位置、通讯录），这需要特别小心的保护",
        "目的限制": "App 说收集信息是为了 A，实际却拿去做了 B",
        "自动化决策": "App 用算法自动判断关于你的事情，但这些判断可能是错的或不公平的",
    }
    
    result = hint
    for legal_term, plain in term_map.items():
        if legal_term in hint or legal_term in legal_context:
            result = plain
            break
    
    # 如果 hint 本身已经够通俗就直接用
    if len(result) > len(hint) * 1.5:
        result = hint
    
    return result

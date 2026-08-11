"""
意图提取模块（v2）。
优先用规则引擎提取（确定性、0 延迟），
只有规则引擎无法处理的模糊查询才 fallback 到 LLM。

架构：
  normalize_question → rule_based_extract → [成功?直接返回 : LLM fallback]
"""

import asyncio
from dataclasses import dataclass
import json
import math
import re
import unicodedata

import config
from course_codes import extract_course_codes, normalize_course_code
from course_index import DEPARTMENT_NAMES
from credit_parser import parse_points_range
from provider_errors import classify_provider_failure


class IntentParseError(ValueError):
    """The model response did not contain a JSON object."""


class IntentValidationError(ValueError):
    """The model returned JSON, but it is not a valid query intent."""


@dataclass(frozen=True)
class IntentExtractionResult:
    """Intent plus provenance used by the SSE metadata contract."""

    intent: dict
    source: str
    fallback_used: bool = False
    fallback_reason: str | None = None


def fold_accents(text: str) -> str:
    """去掉变音符号，让西/法语词能被 [a-z]+ 完整切出。

    旧版直接用 re.findall(r'[a-z]+', ...)，会把 "computación" 切成 "computaci"、
    "recomiéndame" 切成 "recomi"+"ndame"，产生大量垃圾关键词并检索出无关课程。
    NFKD 分解后丢弃组合字符即可得到 "computacion" / "recomiendame"。
    CJK 字符不会产生 ASCII 输出，因此中文查询仍走映射表 + LLM fallback 路径。
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# ============================================================
# 1. 默认意图模板
# ============================================================

DEFAULT_INTENT = {
    "query_type": "general",
    "course_codes": [],
    "keywords": [],
    "department": None,
    # department_terms: 用户用来指代系别的原词（如 "computer science"→{"computer","science"}）。
    # 这些词在按 department 结构化过滤之后已经不具备区分度，
    # 必须排除在 Stage-2 关键词打分之外，否则全系课程同分、排序退化成课号字母序。
    "department_terms": [],
    "instructor": None,
    "time_preference": None,
    "day_preference": [],
    "points_range": None,
    "term": None,
    "comparison_targets": [],
    "suitability": None,
    "department_defaulted_from": None,
    "original_question": ""
}

STATS_QUERY_PATTERNS = (
    "how many departments",
    "how many department",
    "list all departments",
    "list departments",
    "what departments",
    "which departments",
    "how many courses",
    "total courses",
    "有多少系",
    "所有系别",
    "列出所有",
    "课程总数",
)

# 回忆型查询：用户在问「这段对话里出现过哪些课」，而不是发起新检索。
# 这是全项目唯一定义处，response_generator 从这里导入，
# 避免两份列表内容漂移（旧版两个文件各有一份，条目还不一样）。
RECALL_QUERY_PATTERNS = (
    "you mentioned",
    "you have mentioned",
    "you've mentioned",
    "we discussed",
    "we have discussed",
    "we talked about",
    "we've talked",
    "courses you mentioned",
    "courses we discussed",
    "mentioned so far",
    "discussed so far",
    "talked about so far",
    "list all courses",
    "list all the courses",
    "list the courses",
    "all the courses you",
    "what did we talk",
    "what did we discuss",
    "what did i ask",
    "previously discussed",
    "earlier in this conversation",
    "based on the current conversation",
    "上面提到",
    "你提到的",
    "你提到过",
    "我们聊过",
    "我们讨论过",
    "之前推荐的",
    "之前提到",
    "目前为止",
    "列出所有课",
)

def _default_intent_copy() -> dict:
    return {
        "query_type": "general",
        "course_codes": [],
        "keywords": [],
        "department": None,
        "department_terms": [],
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": None,
        "term": None,
        "comparison_targets": [],
        "suitability": None,
        "department_defaulted_from": None,
        "original_question": ""
    }


# ============================================================
# 2. 多语言关键词映射（中文/西语 → 英文）
# ============================================================

ZH_MAPPINGS = {
    # 系别
    "航空": "aerospace", "航天": "aerospace", "土木": "civil engineering",
    "计算机": "computer science", "机械": "mechanical engineering",
    "电子": "electrical engineering", "电气": "electrical engineering",
    "数学": "mathematics", "物理": "physics", "统计": "statistics",
    "化学": "chemical engineering", "化工": "chemical engineering",
    "材料": "materials science", "生物医学": "biomedical",
    "环境": "environmental engineering", "工业工程": "industrial engineering",
    "运筹": "operations research",
    # 时间
    "周一": "Monday", "周二": "Tuesday", "周三": "Wednesday",
    "周四": "Thursday", "周五": "Friday", "周六": "Saturday", "周日": "Sunday",
    "星期一": "Monday", "星期二": "Tuesday", "星期三": "Wednesday",
    "星期四": "Thursday", "星期五": "Friday",
    "上午": "morning", "下午": "afternoon", "晚上": "evening",
    "早上": "morning",
    # 操作
    "推荐": "recommend", "比较": "compare", "对比": "compare",
    "建议": "recommend suggest", "推荐一下": "recommend",
    "给我": "give me", "帮我找": "help me find",
    "有什么": "what are", "有哪些": "what are",
    "有什么好的": "good courses", "有哪些好的": "good courses",
    "介绍": "recommend", "推荐课程": "recommend courses",
    "课程建议": "course recommendations", "课程": "courses", "好的": "good",
    "什么时候": "when schedule", "学分": "credits",
    "先修课": "prerequisites", "教授": "professor",
    "想学": "want to study", "想上": "want to take",
    "想了解": "interested", "感兴趣": "interested",
    # 常见课程主题
    "机器人": "robotics robot", "人工智能": "artificial intelligence ai",
    "机器学习": "machine learning", "数据科学": "data science",
    "结构": "structural", "设计": "design", "编程": "programming",
    "算法": "algorithms", "力学": "mechanics", "热力学": "thermodynamics",
    "信号处理": "signal processing", "控制": "control systems",
    # 补充常见主题，避免中文查询频繁掉进 LLM fallback
    "量子": "quantum", "量子计算": "quantum computing",
    "深度学习": "deep learning", "神经网络": "neural networks",
    "计算机视觉": "computer vision", "自然语言": "natural language",
    "数据库": "database", "操作系统": "operating systems",
    "网络": "networks", "安全": "security", "密码": "cryptography",
    "优化": "optimization", "概率": "probability", "线性代数": "linear algebra",
    "微积分": "calculus", "流体": "fluid mechanics", "有限元": "finite element",
    "复合材料": "composite materials", "纳米": "nanotechnology",
    "能源": "energy", "可持续": "sustainability", "气候": "climate",
    "金融工程": "financial engineering", "供应链": "supply chain",
    "生物": "biology bioengineering", "医学影像": "medical imaging",
    "嵌入式": "embedded systems", "电路": "circuits",
    "航空航天": "aerospace", "推进": "propulsion", "空气动力": "aerodynamics",
    # “入门/基础”描述学生适合度，不是课程标题关键词。它们由规则层
    # 单独解析为 suitability，不能机械追加 introduction/fundamentals。
    "高级": "advanced",
    "毕业设计": "capstone design", "实习": "internship",
}

ES_MAPPINGS = {
    "ingeniería civil": "civil engineering",
    "ingeniería mecánica": "mechanical engineering",
    "ciencias de la computación": "computer science",
    "aeroespacial": "aerospace",
    "lunes": "Monday", "martes": "Tuesday", "miércoles": "Wednesday",
    "jueves": "Thursday", "viernes": "Friday",
    "mañana": "morning", "tarde": "afternoon", "noche": "evening",
    "recomendar": "recommend", "comparar": "compare",
    "recomiéndame": "recommend", "recomendame": "recommend",
    "sugiéreme": "suggest", "sugiereme": "suggest",
    "dame": "give me", "ayúdame a encontrar": "help me find",
    "qué cursos": "what courses", "cuáles cursos": "what courses",
    "cursos buenos": "good courses",
    "sugerencias": "suggestions", "recomendaciones": "recommendations",
    "computación": "computer science", "matemáticas": "mathematics",
    "buenos cursos": "good courses", "mejores cursos": "best courses",
    "créditos": "credits", "profesor": "professor",
    # 学科主题（缺失会导致检索静默返回无关课程）
    "robótica": "robotics robot", "robotica": "robotics robot",
    "aprendizaje automático": "machine learning",
    "aprendizaje automatico": "machine learning",
    "aprendizaje profundo": "deep learning",
    "inteligencia artificial": "artificial intelligence ai",
    "ciencia de datos": "data science",
    "estructural": "structural", "estructuras": "structural",
    "termodinámica": "thermodynamics", "termodinamica": "thermodynamics",
    "mecánica": "mechanics", "mecanica": "mechanics",
    "algoritmos": "algorithms", "programación": "programming",
    "programacion": "programming", "diseño": "design", "diseno": "design",
    "señales": "signal processing", "senales": "signal processing",
    "control": "control systems", "energía": "energy", "energia": "energy",
    "medio ambiente": "environmental engineering",
    "biomédica": "biomedical", "biomedica": "biomedical",
    "materiales": "materials science",
    "aeroespacial": "aerospace", "química": "chemical engineering",
    "quimica": "chemical engineering",
    "por la mañana": "morning", "por la tarde": "afternoon",
}

FR_MAPPINGS = {
    "recommande": "recommend",
    "recommandez": "recommend",
    "recommande-moi": "recommend me",
    "recommandez-moi": "recommend me",
    "suggère": "suggest",
    "suggerez": "suggest",
    "suggère-moi": "suggest me",
    "suggerez-moi": "suggest me",
    "donne-moi": "give me",
    "donnez-moi": "give me",
    "aide-moi à trouver": "help me find",
    "quels cours": "what courses",
    "quelles cours": "what courses",
    "informatique": "computer science",
    "suggestions": "suggestions",
    "recommandations": "recommendations",
    "bons cours": "good courses",
    "meilleurs cours": "best courses",
    # 系别
    "génie civil": "civil engineering", "genie civil": "civil engineering",
    "génie mécanique": "mechanical engineering",
    "genie mecanique": "mechanical engineering",
    "génie électrique": "electrical engineering",
    "genie electrique": "electrical engineering",
    "génie chimique": "chemical engineering",
    "genie chimique": "chemical engineering",
    "aérospatiale": "aerospace", "aerospatiale": "aerospace",
    "mathématiques": "mathematics", "mathematiques": "mathematics",
    "physique": "physics", "statistique": "statistics",
    "biomédical": "biomedical", "biomedical": "biomedical",
    "matériaux": "materials science", "materiaux": "materials science",
    "environnement": "environmental engineering",
    # 学科主题
    "robotique": "robotics robot",
    "apprentissage automatique": "machine learning",
    "apprentissage profond": "deep learning",
    "intelligence artificielle": "artificial intelligence ai",
    "science des données": "data science",
    "science des donnees": "data science",
    "algorithmes": "algorithms", "programmation": "programming",
    "structures": "structural", "structurel": "structural",
    "thermodynamique": "thermodynamics", "mécanique": "mechanics",
    "conception": "design", "traitement du signal": "signal processing",
    "énergie": "energy", "energie": "energy",
    # 时间/学分/称谓
    "lundi": "Monday", "mardi": "Tuesday", "mercredi": "Wednesday",
    "jeudi": "Thursday", "vendredi": "Friday",
    "matin": "morning", "après-midi": "afternoon", "apres-midi": "afternoon",
    "soir": "evening", "crédits": "credits",
    "professeur": "professor",
}


def normalize_question(question: str) -> str:
    """将非英文关键词映射为英文等价物，方便规则引擎处理。
    
    保留原文不删除，只是在末尾追加英文翻译，
    这样既能让规则引擎匹配，又不丢失原始信息。
    """
    additions = []
    q = question
    q_lower = q.lower()
    for source, target in {**ZH_MAPPINGS, **ES_MAPPINGS, **FR_MAPPINGS}.items():
        if source.lower() in q_lower:
            additions.append(target)
    if additions:
        return q + " " + " ".join(additions)
    return q


# ============================================================
# 3. 规则引擎（核心）
# ============================================================

# 短语级映射（优先级最高，先匹配）
DEPT_PHRASE_MAP = {
    # Civil Engineering
    "civil engineering": "CIEN",
    "civil eng": "CIEN",
    # Electrical Engineering
    "electrical engineering": "ELEN",
    "electrical eng": "ELEN",
    "ee courses": "ELEN",
    # Mechanical Engineering
    "mechanical engineering": "MECE",
    "mechanical eng": "MECE",
    "mech eng": "MECE",
    # Biomedical Engineering
    "biomedical engineering": "BMEN",
    "biomed eng": "BMEN",
    "biomedical eng": "BMEN",
    # Chemical Engineering
    "chemical engineering": "CHEN",
    "chem eng": "CHEN",
    # Industrial Engineering (IEOR)
    "industrial engineering": "IEOR",
    "operations research": "IEOR",
    # Applied Mathematics
    "applied mathematics": "APMA",
    "applied math": "APMA",
    # Applied physics
    "applied physics": "APPH",
    # Computer Science
    "computer science": "COMS",
    "comp sci": "COMS",
    # Earth and Environmental Engineering
    "earth and environmental": "EAEE",
    "environmental engineering": "EAEE",
    # Materials Science
    "materials science": "MSAE",
    # Nuclear engineering (if exists)
    "nuclear engineering": "NUCL",
}

# 短语匹配按长度降序，避免短词抢先命中（如 "applied math" vs "math"）
_SORTED_PHRASE_MAP = sorted(DEPT_PHRASE_MAP.items(), key=lambda x: -len(x[0]))

# 单词级系别匹配时，只有这些「纯系别名」可以从关键词中剔除。
# 像 robotics / structural / climate / data 虽然也路由到某个系，
# 但它们本身是主题词，剔除后会丢失检索信号
# （"recommend robotics courses" 会退化成「随便给几门 MECE 课」）。
PURE_DEPT_WORDS = frozenset(
    {
        "cs", "compsci", "computer", "ieor", "aero", "aerospace",
        "civil", "mechanical", "mech", "electrical", "ee",
        "biomedical", "biomed", "environmental", "materials",
        "statistics", "stats", "physics", "chemical", "chem",
        "bioinformatics", "industrial",
    }
    | {prefix.lower() for prefix in DEPARTMENT_NAMES}
)

# 系别名称 → 前缀 的反向映射（从 DEPARTMENT_NAMES 自动构建）
DEPT_KEYWORD_MAP: dict[str, str] = {}
for _prefix, _names in DEPARTMENT_NAMES.items():
    for _word in _names.split():
        if _word not in DEPT_KEYWORD_MAP:  # 先到先得，避免覆盖
            DEPT_KEYWORD_MAP[_word] = _prefix
    # An explicit department code (for example APMA or COMS) is a stronger
    # anchor than a later generic-mathematics default.
    DEPT_KEYWORD_MAP.setdefault(_prefix.lower(), _prefix)

# 补充常见简写和口语化表达
DEPT_KEYWORD_MAP.update({
    "cs": "COMS", "compsci": "COMS", "computer": "COMS",
    "aerospace": "AERO", "aero": "AERO",
    "civil": "CIEN", "structural": "CIEN",
    "mechanical": "MECE", "mech": "MECE",
    "electrical": "ELEN", "ee": "ELEN",
    "ieor": "IEOR",
    "biomedical": "BMEN", "biomed": "BMEN",
    "physics": "APPH",
    "stats": "STAT", "statistics": "STAT",
    "materials": "MSAE",
    "environmental": "EAEE", "climate": "EAEE",
    "robotics": "MECE", "robot": "MECE",
    "data": "COMS",
})

# Python's ``\b`` follows Unicode word semantics, so there is no boundary
# between the ASCII ``s`` and the CJK ``课`` in ``cs课``.  Only genuine ASCII
# aliases/codes use ASCII-aware lookarounds; ordinary department words retain
# their existing Unicode word-boundary behavior.
ASCII_DEPARTMENT_ALIASES = frozenset(
    {
        "cs", "compsci", "ieor", "aero", "mech", "ee", "biomed",
        "stats", "chem",
    }
    | {prefix.lower() for prefix in DEPARTMENT_NAMES}
)


def contains_ascii_alias(text: str, alias: str) -> bool:
    """Match one standalone ASCII alias even when adjacent to CJK text."""

    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])",
            text,
            flags=re.IGNORECASE,
        )
    )

# 移除易冲突通用词，避免误匹配到错误系别
for _generic_word in ("engineering", "applied", "science", "math", "mathematics"):
    DEPT_KEYWORD_MAP.pop(_generic_word, None)

COMPARE_RE = re.compile(r'\b(compare|comparison|difference|differ|vs\.?|versus)\b', re.I)
RECOMMEND_RE = re.compile(
    r'\b('
    r'recommend|suggest|interested|advice|advise|recommend me|'
    r'want to (?:learn|study|take)|'
    r'should i take|looking for|looking to take|'
    r'help me find|give me|show me|'
    r'good(?:\s+\w+){0,4}\s+(?:courses?|classes?)|'
    r'best(?:\s+\w+){0,4}\s+(?:courses?|classes?)|'
    r'what (?:courses?|classes?) (?:should|would|can|do)|'
    r'suggestions?|recommendations?|course recommendations?|'
    r'recomiéndame|recomendame|sugiéreme|sugiereme|dame|'
    r'recommande|recommandez|recommande-moi|recommandez-moi|'
    r'suggère|suggerez|suggère-moi|suggerez-moi|donne-moi|donnez-moi'
    r')\b',
    re.I
)
INSTRUCTOR_RE = re.compile(r'\b(professor|prof\.?|instructor|taught by|teach(?:es|ing)?)\b', re.I)

# --- 教授名提取 ---
# 历史 bug：整条正则加 re.I 会让 [A-Z] 也匹配小写字母，导致名字后面的动词被一起吞掉
# （"Professor Panayotidi teach" → instructor="Panayotidi teach" → 0 结果）。
# 修复方式：职称部分用作用域内联标志 (?i:...) 忽略大小写，名字部分保持大小写敏感。
_PROF_TITLE = r"(?i:\bprofessors?\b|\bprof\b\.?|\binstructors?\b|\bdr\b\.?)"

# 严格版：名字必须首字母大写，天然排除后面的小写动词。
PROF_STRICT_RE = re.compile(
    _PROF_TITLE + r"\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)*)"
)
# 宽松版：用户全小写输入时兜底，靠 noise token 过滤动词。
PROF_LOOSE_RE = re.compile(
    _PROF_TITLE + r"\s+([a-zA-Z'\-]{2,}(?:\s+[a-zA-Z'\-]{2,}){0,2})", re.I
)
# "taught by X" / "taught by Professor X"
TAUGHT_BY_RE = re.compile(
    r"\btaught\s+by\s+(?:professor\s+|prof\.?\s+|dr\.?\s+)?"
    r"([a-zA-Z'\-]{2,}(?:\s+[a-zA-Z'\-]{2,}){0,2})",
    re.I,
)

# 跟在教授名前后、需要剥掉的噪声词。
INSTRUCTOR_NOISE_TOKENS = frozenset({
    "teach", "teaches", "teaching", "taught",
    "offer", "offers", "offering", "offered", "give", "gives",
    "course", "courses", "class", "classes", "section", "sections",
    "lecture", "lectures", "seminar",
    "this", "next", "last", "the", "a", "an", "any", "all",
    "spring", "fall", "summer", "winter", "semester", "term", "year",
    "is", "are", "was", "were", "does", "do", "did", "has", "have", "had",
    "in", "on", "at", "for", "with", "about", "by", "from", "of", "and", "or",
    "what", "which", "who", "whom", "whose", "when", "where", "how",
    "show", "list", "find", "tell", "me", "my", "i", "you",
    "available", "offered", "still", "also",
})


def _clean_instructor_name(raw: str) -> str:
    """剥掉教授名前后的噪声词（动词、疑问词、学期名等）。"""
    tokens = [tok for tok in re.split(r"\s+", (raw or "").strip()) if tok]
    stripped = [tok.strip(".,;:?!\"'") for tok in tokens]
    stripped = [tok for tok in stripped if tok]

    while stripped and stripped[-1].lower() in INSTRUCTOR_NOISE_TOKENS:
        stripped.pop()
    while stripped and stripped[0].lower() in INSTRUCTOR_NOISE_TOKENS:
        stripped.pop(0)

    # 名字最多 3 个词，超出说明多吞了句子成分。
    return " ".join(stripped[:3]).strip()


def extract_instructor(question: str) -> str | None:
    """从问题中提取教授名，提取不到返回 None。"""
    for regex in (PROF_STRICT_RE, TAUGHT_BY_RE, PROF_LOOSE_RE):
        match = regex.search(question)
        if not match:
            continue
        name = _clean_instructor_name(match.group(1))
        if len(name) >= 2:
            return name

    # 没有职称词但出现 "taught by / teaches" 等信号时，退回找首字母大写的候选名。
    if INSTRUCTOR_RE.search(question):
        for candidate in re.findall(r"\b([A-Z][a-zA-Z'\-]{2,})\b", question):
            if candidate.lower() not in INSTRUCTOR_NOISE_TOKENS:
                return candidate
    return None

TIME_KEYWORDS = {"morning": "morning", "afternoon": "afternoon", "evening": "evening"}

DAY_KEYWORDS = {
    "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday",
    "thursday": "Thursday", "friday": "Friday", "saturday": "Saturday",
    "sunday": "Sunday",
    "mon": "Monday", "tue": "Tuesday", "tues": "Tuesday",
    "wed": "Wednesday", "thu": "Thursday", "thur": "Thursday",
    "thurs": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
}

BEGINNER_SUITABILITY_RE = re.compile(
    r"\b(?:beginners?|novices?|entry[- ]level|zero[- ]background|"
    r"no\s+prior\s+(?:background|experience|knowledge)|"
    r"without\s+prior\s+(?:background|experience|knowledge))\b|"
    r"零基础|无基础|没有基础|初学者|新手|入门|基础(?:课|课程)",
    re.IGNORECASE,
)

GENERIC_MATHEMATICS_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:math|mathematics)(?![A-Za-z0-9_])|数学",
    re.IGNORECASE,
)

# 停用词（用于 keyword 提取）
STOP_WORDS = frozenset({
    'what', 'are', 'the', 'is', 'a', 'an', 'in', 'on', 'for', 'to', 'of',
    'and', 'or', 'can', 'you', 'i', 'me', 'my', 'do', 'does', 'any', 'some',
    'all', 'courses', 'course', 'class', 'classes', 'about', 'tell', 'show',
    'list', 'find', 'get', 'have', 'has', 'with', 'from', 'at', 'by', 'this',
    'that', 'there', 'available', 'offered', 'offer', 'take', 'want', 'like',
    'please', 'could', 'would', 'should', 'also', 'more', 'much', 'many',
    'how', 'which', 'where', 'when', 'who', 'whom', 'why', 'been', 'being',
    'not', 'but', 'if', 'then', 'than', 'too', 'very', 'just', 'only',
    'recommend', 'suggest', 'compare', 'need', 'interested', 'looking',
    'want', 'wants', 'wanted', 'good', 'best', 'easy', 'easiest', 'hard',
    'beginner', 'beginners', 'novice', 'novices', 'entry', 'level', 'zero',
    'prior', 'background', 'experience', 'knowledge', 'without',
    'give', 'other', 'another', 'more', 'additional', 'else',
    'based', 'current', 'conversation', 'mentioned', 'them', 'their',
    'department', 'departments',
    # Output-shape words are not course topics.  Course identifiers themselves
    # are extracted separately before keyword ranking.
    'code', 'codes',
    # Requested result counts describe the query shape, not a course topic.
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty',
    # --- 西班牙语功能词（已折叠变音符号后的形式）---
    'que', 'cual', 'cuales', 'cursos', 'curso', 'clase', 'clases',
    'los', 'las', 'una', 'unos', 'unas', 'del', 'para', 'por', 'con',
    'sobre', 'hay', 'tiene', 'tienen', 'quiero', 'busco', 'dame',
    'recomiendame', 'recomendame', 'sugiereme', 'ayudame', 'encontrar',
    'buenos', 'buenas', 'mejores', 'mejor', 'algunos', 'algunas',
    'disponibles', 'disponible', 'ciencias',
    'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho',
    'nueve', 'diez',
    # --- 法语功能词 ---
    'cours', 'quels', 'quelles', 'quel', 'quelle', 'sont', 'est',
    'des', 'les', 'une', 'sur', 'dans', 'pour', 'avec', 'moi', 'toi',
    'recommande', 'recommandez', 'suggere', 'suggerez', 'donne', 'donnez',
    'aide', 'trouver', 'bons', 'bonnes', 'meilleurs', 'meilleur',
    'disponible', 'disponibles', 'veux', 'cherche',
    'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit',
    'neuf', 'dix',
})


def rule_based_extract(question: str) -> dict | None:
    """
    用规则提取意图。成功返回 intent dict，无法确定返回 None（fallback 到 LLM）。
    """
    q_lower = question.lower()
    intent = _default_intent_copy()
    intent["original_question"] = question
    if BEGINNER_SUITABILITY_RE.search(question):
        intent["suitability"] = "beginner"

    if any(pattern in q_lower for pattern in STATS_QUERY_PATTERNS):
        intent["query_type"] = "stats"
        intent["keywords"] = []
        return intent
    if any(pattern in q_lower for pattern in RECALL_QUERY_PATTERNS):
        intent["query_type"] = "search"
        intent["keywords"] = []
        return intent

    # --- 1. 提取课程代码 ---
    # Shared parser covers current single-letter levels B/C/E/W and two-letter
    # levels UN/GU/GR, plus compact/hyphen/underscore spelling variants.
    codes = extract_course_codes(question)
    intent["course_codes"] = codes

    # --- 2. 判断 query_type ---
    if len(codes) >= 2 and COMPARE_RE.search(question):
        intent["query_type"] = "compare"
        intent["comparison_targets"] = list(codes)
    elif COMPARE_RE.search(question) and len(codes) >= 2:
        intent["query_type"] = "compare"
        intent["comparison_targets"] = list(codes)
    elif len(codes) == 1:
        intent["query_type"] = "detail"
    elif COMPARE_RE.search(question):
        intent["query_type"] = "compare"
    elif RECOMMEND_RE.search(question):
        intent["query_type"] = "recommend"
    else:
        intent["query_type"] = "search"

    # --- 3. 提取系别 ---
    # 阶段 1：短语级优先匹配
    department = None
    department_terms: list[str] = []
    for phrase, dept in _SORTED_PHRASE_MAP:
        if phrase in q_lower:
            department = dept
            # 记下用户实际用来指代系别的词，供 Stage-2 打分时排除。
            department_terms = [w for w in re.findall(r"[a-z]+", phrase) if len(w) > 2]
            break

    # 阶段 2：短语未命中时，再做单词级兜底
    if not department:
        sorted_dept_keywords = sorted(DEPT_KEYWORD_MAP.keys(), key=len, reverse=True)
        for word in sorted_dept_keywords:
            matches_word = (
                contains_ascii_alias(q_lower, word)
                if word in ASCII_DEPARTMENT_ALIASES
                else bool(re.search(r'\b' + re.escape(word) + r'\b', q_lower))
            )
            if matches_word:
                # DEPARTMENT_NAMES intentionally enriches searchable text with
                # topical words.  Those words must not become a hard department
                # filter (e.g. robotics also appears outside MECE).
                if word not in PURE_DEPT_WORDS:
                    continue
                department = DEPT_KEYWORD_MAP[word]
                department_terms = [word]
                break

    # “数学/math” is a product default for the current Engineering catalog,
    # not an explicit department name.  Apply it only after course-code and
    # explicit department resolution so it can never conflict with either.
    if not codes and not department and GENERIC_MATHEMATICS_RE.search(question):
        department = "APMA"
        intent["department_defaulted_from"] = "mathematics"
        department_terms = [
            word
            for word in ("math", "mathematics")
            if re.search(
                rf"(?<![A-Za-z0-9_]){word}(?![A-Za-z0-9_])", q_lower
            )
        ] or ["mathematics"]
    if department:
        # normalize_question appends an English department phrase but keeps the
        # original Spanish/French wording.  Mark both as structural terms so
        # e.g. "ciencias de la computación" does not become a mandatory topic
        # keyword after COMS has already been selected.
        for source, target in {**ES_MAPPINGS, **FR_MAPPINGS}.items():
            if source.lower() not in q_lower:
                continue
            mapped_department = DEPT_PHRASE_MAP.get(target.lower())
            if mapped_department != department:
                continue
            for word in re.findall(r"[a-z]+", fold_accents(source.lower())):
                if len(word) > 2 and word not in department_terms:
                    department_terms.append(word)
    intent["department"] = department
    intent["department_terms"] = department_terms

    # --- 4. 提取教授名 ---
    intent["instructor"] = extract_instructor(question)

    # --- 5. 提取时间偏好 ---
    for key, val in TIME_KEYWORDS.items():
        if key in q_lower:
            intent["time_preference"] = val
            break

    # --- 6. 提取星期偏好 ---
    # 注意 s? ：用户写 "Tuesdays"（复数）非常常见，
    # 旧版只匹配 \btuesday\b，导致时间条件被静默丢弃且不提示用户。
    days = []
    for key, val in DAY_KEYWORDS.items():
        if re.search(r'\b' + re.escape(key) + r's?\b', q_lower):
            if val not in days:
                days.append(val)
    intent["day_preference"] = days

    # --- 7. 提取学分 ---
    intent["points_range"] = parse_points_range(question)

    # --- 8. 提取学期 ---
    term_match = re.search(r'(spring|fall|summer|winter)\s+(\d{4})', q_lower)
    if term_match:
        intent["term"] = f"{term_match.group(1).capitalize()} {term_match.group(2)}"

    # --- 9. 提取搜索关键词 ---
    # 先折叠变音符号，否则 "computación"/"robótica" 会被切碎成无意义片段。
    words = re.findall(r'[a-z]+', fold_accents(q_lower))
    seen_kw: set[str] = set()
    keywords: list[str] = []
    for w in words:
        if w in STOP_WORDS or len(w) <= 2 or w in seen_kw:
            continue
        seen_kw.add(w)
        keywords.append(w)
    intent["keywords"] = keywords[:5]

    # --- 10. 判断是否有足够信号 ---
    has_signal = (
        intent["course_codes"] or intent["department"] or
        intent["instructor"] or intent["time_preference"] or
        intent["day_preference"] or intent["points_range"] or
        intent["term"] or
        intent["suitability"] or
        # recommend/compare 由正则明确触发，本身就是强信号
        intent["query_type"] in ("recommend", "compare") or
        # 对于 keywords，至少要有 1 个有意义的词
        len([k for k in intent["keywords"] if k not in {'engineering', 'science'}]) > 0
    )

    # --- 11. 检测 general（非课程相关）问题 ---
    GENERAL_PATTERNS = re.compile(
        r'\b(how (?:do|can|to) i (?:register|enroll|sign up|apply|drop|withdraw|audit)|'
        r'what(?:\'s| is) (?:the deadline|my gpa|the tuition|the fee)|'
        r'where is (?:the|my)|office hours|academic calendar|'
        r'who (?:is|are) (?:the|my) (?:advisor|dean))\b', re.I
    )
    if GENERAL_PATTERNS.search(question) and not intent["course_codes"]:
        intent["query_type"] = "general"
        return intent

    if not has_signal:
        return None  # 让 LLM 处理

    return intent


# ============================================================
# 4. LLM Fallback（精简版 prompt）
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """Extract query intent as JSON only. No other text. /no_think

Format:
{"query_type":"search|compare|recommend|detail|schedule|general|stats","course_codes":[],"keywords":[],"department":null,"instructor":null,"time_preference":null,"day_preference":[],"points_range":null,"term":null,"comparison_targets":[],"original_question":""}

Department codes: AERO=aerospace, CIEN=civil, COMS=computer science, MECE=mechanical, IEOR=industrial/operations, ELEN=electrical, APMA=applied math, EAEE=environmental, BMEN=biomedical, CHEN=chemical, MSAE=materials, ENME=mechanics.
time_preference: "morning"|"afternoon"|"evening". day_preference: ["Monday","Tuesday",...].
Use "general" ONLY for non-course questions (e.g. "how do I register").

Q: "What 3-credit courses on Tuesdays?"
A: {"query_type":"search","course_codes":[],"keywords":[],"department":null,"instructor":null,"time_preference":null,"day_preference":["Tuesday"],"points_range":[3,3],"term":null,"comparison_targets":[],"original_question":"What 3-credit courses on Tuesdays?"}

Q: "Compare CIEN E3125 and ENME E3113"
A: {"query_type":"compare","course_codes":["CIEN E3125","ENME E3113"],"keywords":[],"department":null,"instructor":null,"time_preference":null,"day_preference":[],"points_range":null,"term":null,"comparison_targets":["CIEN E3125","ENME E3113"],"original_question":"Compare CIEN E3125 and ENME E3113"}

Q: "I'm interested in aerospace"
A: {"query_type":"recommend","course_codes":[],"keywords":["aerospace"],"department":"AERO","instructor":null,"time_preference":null,"day_preference":[],"points_range":null,"term":null,"comparison_targets":[],"original_question":"I'm interested in aerospace"}"""


# ============================================================
# 5. JSON 解析容错
# ============================================================

def _try_parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_by_json_block(text: str) -> dict | None:
    for match in re.finditer(r'\{.*?\}', text, flags=re.DOTALL):
        parsed = _try_parse_json(match.group(0))
        if parsed is not None:
            return parsed
    return None


def _parse_from_fenced_json(text: str) -> dict | None:
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return _try_parse_json(match.group(1))


def _remove_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


ALLOWED_QUERY_TYPES = frozenset(
    {"search", "compare", "recommend", "detail", "schedule", "general", "stats"}
)
ALLOWED_TIME_PREFERENCES = frozenset({"morning", "afternoon", "evening"})
ALLOWED_SUITABILITY = frozenset({"beginner"})
ALLOWED_DAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_DAY_BY_LOWER = {day.lower(): day for day in ALLOWED_DAYS}
_DEPARTMENT_CODE_RE = re.compile(r"^[A-Z]{2,6}$")
_TERM_VALUE_RE = re.compile(r"^(Spring|Summer|Fall|Winter)\s+\d{4}$", re.IGNORECASE)


def _validated_string_list(value: object, field: str, *, lower: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise IntentValidationError(f"{field} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise IntentValidationError(f"{field} entries must be strings")
        cleaned = item.strip()
        if not cleaned:
            continue
        result.append(cleaned.lower() if lower else cleaned)
    return result


def validate_intent_schema(parsed: dict) -> dict:
    """Validate and normalize one model's JSON without borrowing fields from another model.

    Missing optional fields are filled from ``DEFAULT_INTENT`` only after the payload has
    passed validation.  In particular, a syntactically valid ``general`` intent remains
    distinguishable from malformed JSON or an empty object.
    """
    if not isinstance(parsed, dict):
        raise IntentValidationError("intent must be a JSON object")

    query_type = parsed.get("query_type")
    if not isinstance(query_type, str) or query_type not in ALLOWED_QUERY_TYPES:
        raise IntentValidationError("query_type is missing or invalid")

    merged = _default_intent_copy()
    merged["query_type"] = query_type

    for field in ("course_codes", "comparison_targets"):
        raw = parsed.get(field, [])
        normalized_codes: list[str] = []
        for item in _validated_string_list(raw, field):
            normalized = normalize_course_code(item)
            if normalized is None:
                raise IntentValidationError(f"{field} contains an invalid course code")
            if normalized not in normalized_codes:
                normalized_codes.append(normalized)
        merged[field] = normalized_codes

    course_codes = merged["course_codes"]
    comparison_targets = merged["comparison_targets"]
    if query_type == "compare":
        if course_codes and comparison_targets and set(course_codes) != set(
            comparison_targets
        ):
            raise IntentValidationError(
                "comparison_targets conflict with course_codes"
            )
        if comparison_targets and not course_codes:
            # Models occasionally populate only the compare-specific field.
            # Canonicalize it to the retrieval field without combining
            # conflicting candidate sets from one uncertain response.
            merged["course_codes"] = list(comparison_targets)
    elif comparison_targets:
        raise IntentValidationError(
            "comparison_targets are only valid for compare intents"
        )

    merged["keywords"] = _validated_string_list(
        parsed.get("keywords", []), "keywords", lower=True
    )
    merged["department_terms"] = _validated_string_list(
        parsed.get("department_terms", []), "department_terms", lower=True
    )

    department = parsed.get("department")
    if department is not None:
        if not isinstance(department, str):
            raise IntentValidationError("department must be a string or null")
        department = department.strip().upper()
        if (
            not department
            or not _DEPARTMENT_CODE_RE.fullmatch(department)
            or department not in DEPARTMENT_NAMES
        ):
            raise IntentValidationError("department must be a known department code")
    merged["department"] = department

    suitability = parsed.get("suitability")
    if suitability is not None:
        if not isinstance(suitability, str):
            raise IntentValidationError("suitability must be a string or null")
        suitability = suitability.strip().lower()
        if suitability not in ALLOWED_SUITABILITY:
            raise IntentValidationError("suitability is invalid")
    merged["suitability"] = suitability

    defaulted_from = parsed.get("department_defaulted_from")
    if defaulted_from is not None:
        if not isinstance(defaulted_from, str):
            raise IntentValidationError(
                "department_defaulted_from must be a string or null"
            )
        defaulted_from = defaulted_from.strip().lower()
        if defaulted_from != "mathematics" or department != "APMA":
            raise IntentValidationError("department_defaulted_from is invalid")
    merged["department_defaulted_from"] = defaulted_from

    instructor = parsed.get("instructor")
    if instructor is not None:
        if not isinstance(instructor, str):
            raise IntentValidationError("instructor must be a string or null")
        instructor = instructor.strip() or None
    merged["instructor"] = instructor

    time_preference = parsed.get("time_preference")
    if time_preference is not None:
        if not isinstance(time_preference, str):
            raise IntentValidationError("time_preference must be a string or null")
        time_preference = time_preference.strip().lower()
        if time_preference not in ALLOWED_TIME_PREFERENCES:
            raise IntentValidationError("time_preference is invalid")
    merged["time_preference"] = time_preference

    raw_days = parsed.get("day_preference", [])
    days = _validated_string_list(raw_days, "day_preference")
    normalized_days: list[str] = []
    for day in days:
        normalized = _DAY_BY_LOWER.get(day.lower())
        if normalized is None:
            raise IntentValidationError(f"invalid day_preference: {day}")
        if normalized not in normalized_days:
            normalized_days.append(normalized)
    merged["day_preference"] = normalized_days

    points = parsed.get("points_range")
    if points is not None:
        if not isinstance(points, (list, tuple)) or len(points) != 2:
            raise IntentValidationError("points_range must contain exactly two numbers")
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, (int, float)))
            for value in points
        ):
            raise IntentValidationError("points_range values must be numeric or null")
        min_points = float(points[0]) if points[0] is not None else None
        max_points = float(points[1]) if points[1] is not None else None
        if min_points is None and max_points is None:
            raise IntentValidationError("points_range needs at least one bound")
        if any(
            value is not None and not math.isfinite(value)
            for value in (min_points, max_points)
        ):
            raise IntentValidationError("points_range values must be finite")
        if (
            (min_points is not None and min_points < 0)
            or (max_points is not None and max_points > 30)
            or (
                min_points is not None
                and max_points is not None
                and min_points > max_points
            )
        ):
            raise IntentValidationError("points_range is outside the valid range")
        merged["points_range"] = [min_points, max_points]

    term = parsed.get("term")
    if term is not None:
        if not isinstance(term, str):
            raise IntentValidationError("term must be a string or null")
        term = term.strip()
        if not term or len(term) > 64 or not _TERM_VALUE_RE.fullmatch(term):
            raise IntentValidationError("term is invalid")
    merged["term"] = term

    original_question = parsed.get("original_question", "")
    if not isinstance(original_question, str):
        raise IntentValidationError("original_question must be a string")
    merged["original_question"] = original_question
    return merged


def parse_extraction_response(raw_text: str) -> dict:
    """Extract and validate one model response.

    Invalid output raises an internal exception instead of being converted to the default
    ``general`` intent.  This is what allows the caller to accept a *valid* general intent
    while still falling back from malformed JSON or illegal fields.
    """
    text = raw_text or ""
    # Prefer the visible answer over JSON that may appear inside a Qwen <think> block.
    without_think = _remove_think_tags(text)
    parsed = _try_parse_json(without_think)
    if parsed is None:
        parsed = _parse_from_fenced_json(without_think)
    if parsed is None:
        parsed = _parse_by_json_block(without_think)
    if parsed is None:
        parsed = _try_parse_json(text)
        if parsed is None:
            parsed = _parse_from_fenced_json(text)
        if parsed is None:
            parsed = _parse_by_json_block(text)
    if parsed is None:
        raise IntentParseError("intent response did not contain valid JSON")
    return validate_intent_schema(parsed)


# ============================================================
# 6. 主入口
# ============================================================

def _intent_failure_reason(exc: Exception) -> str:
    if isinstance(exc, IntentParseError):
        return "invalid_json"
    if isinstance(exc, IntentValidationError):
        return "invalid_schema"
    return classify_provider_failure(exc)


async def _call_intent_model(
    client,
    messages: list[dict],
    *,
    model: str,
    timeout: float,
) -> str:
    kwargs = {
        "messages": messages,
        "system_prompt": EXTRACTION_SYSTEM_PROMPT,
        "max_tokens": config.INTENT_MAX_TOKENS,
    }
    if model:
        kwargs["model"] = model
    return await asyncio.wait_for(client.chat(**kwargs), timeout=timeout)


async def extract_query_intent_result(
    question: str,
    llm_client=None,
    model: str = "",
    *,
    primary_source: str = "ollama",
    fallback_client=None,
    fallback_model: str = "",
    fallback_source: str = "ollama",
    timeout: float | None = None,
) -> IntentExtractionResult:
    """
    Complete deterministic -> primary -> fallback -> minimal intent flow.

    1. 多语言关键词归一化
    2. 规则引擎尝试提取
    3. 成功 → 直接返回（不调用 LLM，0 延迟）
    4. 失败 → primary LLM；解析和语义校验失败才调用 fallback LLM
    5. 两个模型都失败 → deterministic minimal intent
    """
    normalized = normalize_question(question)
    rule_result = rule_based_extract(normalized)
    if rule_result is not None:
        rule_result["original_question"] = question  # 保留原始问题
        return IntentExtractionResult(intent=rule_result, source="rule")

    messages = [{"role": "user", "content": question}]
    request_timeout = float(timeout if timeout is not None else config.INTENT_TIMEOUT)
    primary_reason: str | None = None

    if llm_client is not None:
        try:
            raw_response = await _call_intent_model(
                llm_client,
                messages,
                model=model,
                timeout=request_timeout,
            )
            intent = parse_extraction_response(raw_response)
            intent["original_question"] = question
            return IntentExtractionResult(intent=intent, source=primary_source)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            primary_reason = _intent_failure_reason(exc)

    if fallback_client is not None:
        try:
            raw_response = await _call_intent_model(
                fallback_client,
                messages,
                model=fallback_model,
                timeout=request_timeout,
            )
            intent = parse_extraction_response(raw_response)
            intent["original_question"] = question
            return IntentExtractionResult(
                intent=intent,
                source=fallback_source,
                fallback_used=True,
                fallback_reason=primary_reason,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            fallback_reason = _intent_failure_reason(exc)
            if primary_reason:
                primary_reason = f"{primary_reason};{fallback_source}_{fallback_reason}"
            else:
                primary_reason = fallback_reason

    intent = _default_intent_copy()
    intent["original_question"] = question
    return IntentExtractionResult(
        intent=intent,
        source="minimal",
        fallback_used=fallback_client is not None,
        fallback_reason=primary_reason,
    )


async def extract_query_intent(
    question: str,
    llm_client,
    model: str = "",
    *,
    fallback_client=None,
    fallback_model: str = "",
    timeout: float | None = None,
) -> dict:
    """Backward-compatible dict-returning wrapper around the provenance-aware flow."""
    result = await extract_query_intent_result(
        question,
        llm_client,
        model=model,
        primary_source="groq" if model == config.GROQ_INTENT_MODEL else "ollama",
        fallback_client=fallback_client,
        fallback_model=fallback_model,
        fallback_source="ollama",
        timeout=timeout,
    )
    return result.intent

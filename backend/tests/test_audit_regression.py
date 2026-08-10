"""
针对 2026-08-10 代码审查发现问题的回归测试。

每个测试对应 BUG_REPORT.md 中的一个编号，防止修复被后续改动悄悄回退。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import server  # noqa: E402
from course_index import (  # noqa: E402
    course_quality_score,
    filter_by_fields,
    load_enriched_index,
)
from course_retriever import retrieve_courses  # noqa: E402
from query_parser import (  # noqa: E402
    DEFAULT_INTENT,
    extract_instructor,
    fold_accents,
    rule_based_extract,
    normalize_question,
)
from response_generator import build_answer_prompt  # noqa: E402


@pytest.fixture(scope="module")
def index() -> list[dict]:
    return load_enriched_index(str(config.ENRICHED_INDEX_PATH))


def _codes(courses: list[dict]) -> list[str]:
    return [c.get("course_code", "") for c in courses]


def _search(index: list[dict], question: str, max_results: int = 5) -> list[dict]:
    intent = rule_based_extract(normalize_question(question))
    assert intent is not None, f"rule engine returned None for: {question}"
    return retrieve_courses(index, intent, str(config.COURSES_DIR), max_results=max_results)


# ---------------------------------------------------------------- #1 教授检索
@pytest.mark.parametrize(
    "question,expected",
    [
        ("Which courses does Professor Panayotidi teach?", "Panayotidi"),
        ("What is Professor Panayotidi teaching this spring?", "Panayotidi"),
        ("courses taught by Panayotidi", "Panayotidi"),
        ("Professor Tom Panayotidi courses", "Tom Panayotidi"),
        ("prof. Panayotidi", "Panayotidi"),
    ],
)
def test_bug01_instructor_name_not_polluted_by_verbs(question, expected):
    """re.I 让 [A-Z] 匹配小写，把名字后面的动词一起吞进 instructor。"""
    assert extract_instructor(question) == expected


def test_bug01_instructor_query_returns_results(index):
    courses = _search(index, "Which courses does Professor Panayotidi teach?")
    assert courses, "按教授检索不应返回空"
    for course in courses:
        instructors = " ".join(
            (s.get("instructor") or "") for s in (course.get("sections") or [])
        )
        assert "panayotidi" in instructors.lower()


# ---------------------------------------------------------------- #2 排序退化
def test_bug02_department_query_not_alphabetical_placeholders(index):
    """
    "有哪些计算机课" 旧版返回 TUTORIAL / PROJECTS / TOPICS / SEMINAR IN COMPUTER SCIENCE
    —— 全系同分后按课号字母序取前 5。
    """
    codes = _codes(_search(index, "What computer science courses are available?"))
    assert codes, "系别检索不应为空"

    placeholders = {"COMS E6900", "COMS E6901", "COMS E6998", "COMS E9902"}
    assert not (set(codes) & placeholders), f"占位课程仍排在前列: {codes}"
    assert all(c.startswith("COMS") for c in codes)


def test_bug02_department_terms_stripped_from_scoring():
    intent = rule_based_extract(normalize_question("What computer science courses are available?"))
    assert intent["department"] == "COMS"
    assert set(intent["department_terms"]) == {"computer", "science"}


def test_bug02_topical_words_survive_department_routing():
    """robotics 是跨系主题，不能硬锁 MECE，也不能被剔除。"""
    intent = rule_based_extract(normalize_question("recommend some robotics courses"))
    assert intent["department"] is None
    assert intent["department_terms"] == []
    assert "robotics" in intent["keywords"]


def test_bug02_robotics_query_returns_actual_robotics_courses(index):
    titles = [c.get("title", "").lower() for c in _search(index, "recommend some robotics courses")]
    assert any("robot" in t for t in titles), f"没有一门是机器人课: {titles}"


def test_bug02_quality_score_penalizes_placeholder_courses():
    real = {
        "course_code": "COMS W1004",
        "title": "PROGRAMMING IN JAVA",
        "has_description": True,
        "sections_summary": [{"times": "M W 10:10am", "instructor": "X"}],
    }
    placeholder = {
        "course_code": "COMS E6901",
        "title": "PROJECTS IN COMPUTER SCIENCE",
        "has_description": False,
        "sections_summary": [],
    }
    assert course_quality_score(real) > course_quality_score(placeholder)


# ---------------------------------------------------------------- #3 重复课程记录保持独立
def test_bug03_same_code_records_are_not_merged():
    entries = [
        {
            "course_uid": "uid-a",
            "course_code": "ENGI E4300",
            "has_description": False,
            "sections_summary": [],
        },
        {
            "course_uid": "uid-b",
            "course_code": "ENGI E4300",
            "has_description": True,
            "sections_summary": [{"instructor": "A"}],
        },
    ]
    result = filter_by_fields(entries, {"course_codes": ["ENGI E4300"]})
    assert len(result) == 2
    assert [entry["course_uid"] for entry in result] == ["uid-a", "uid-b"]


def test_bug03_results_do_not_repeat_the_same_record(index):
    for question in [
        "Compare CIEN E3125 and ENME E3113",
        "I'm interested in machine learning",
        "computer science courses",
    ]:
        courses = _search(index, question, max_results=10)
        identities = [
            course.get("course_uid") or course.get("source_page_url")
            for course in courses
        ]
        nonempty = [identity for identity in identities if identity]
        assert len(nonempty) == len(set(nonempty)), f"同一记录被重复返回: {identities}"


# ---------------------------------------------------------------- #4 星期复数
@pytest.mark.parametrize(
    "question,expected",
    [
        ("What 3-credit courses are on Tuesdays?", ["Tuesday"]),
        ("on Tuesday", ["Tuesday"]),
        ("classes on Mondays and Wednesdays", ["Monday", "Wednesday"]),
        ("周二有什么课", ["Tuesday"]),
    ],
)
def test_bug04_plural_days_recognized(question, expected):
    intent = rule_based_extract(normalize_question(question))
    assert intent is not None
    assert sorted(intent["day_preference"]) == sorted(expected)


# ---------------------------------------------------------------- #7 token 上限
def test_bug07_token_limits_are_separate_and_large_enough():
    assert config.IMPORT_MAX_TOKENS >= 2048, "导入要输出完整课程 JSON"
    assert config.IMPORT_INPUT_MAX_CHARS >= 4000
    # 导入预算必须高于常规 5 门课的回答预算
    assert config.IMPORT_MAX_TOKENS > config.response_token_budget(5)


def test_bug07_response_budget_scales_with_course_count():
    """固定值在 5 门课时浪费、20 门课时不够，预算必须跟着课程数走。"""
    b5 = config.response_token_budget(5)
    b20 = config.response_token_budget(20)

    assert b5 >= 768, "5 门课详细对比约需 580 tok，预算要有余量"
    assert b20 >= 2000, "20 门课列表约需 2000 tok"
    assert b20 > b5, "预算必须随课程数增长"

    # 防跑飞：任何输入都不能超过绝对上限
    assert config.response_token_budget(9999) <= config.RESPONSE_MAX_TOKENS
    # 无课程的 follow-up 也要够写一段回答
    assert config.response_token_budget(0) >= 512


def test_bug07_local_context_window_not_exceeded():
    """本地 qwen3-nothink 的 Modelfile 写死 num_ctx=8192，是硬约束。"""
    LOCAL_CTX = 8192
    worst_prompt_tokens = 1865 + 400  # 20 门课上下文 + system prompt 实测值
    assert worst_prompt_tokens + config.response_token_budget(20) < LOCAL_CTX

    import_prompt_tokens = config.IMPORT_INPUT_MAX_CHARS // 4 + 400
    assert import_prompt_tokens + config.IMPORT_MAX_TOKENS < LOCAL_CTX


# ---------------------------------------------------------------- #8 max_results
def test_bug08_max_results_reaches_llm_prompt():
    courses = [
        {"course_code": f"TEST E{4000 + i}", "title": f"Course {i}", "sections": []}
        for i in range(12)
    ]
    intent = {"query_type": "search", "original_question": "q"}

    prompt_default, _ = build_answer_prompt(intent, courses, "en")
    assert prompt_default.count("[TEST E") == config.MAX_RETRIEVAL_RESULTS

    prompt_wide, _ = build_answer_prompt(intent, courses, "en", max_results=12)
    assert prompt_wide.count("[TEST E") == 12, "设置里调大 maxResults 后模型仍只看到 5 门"


# ---------------------------------------------------------------- #9 多语言
def test_bug09_accent_folding():
    assert fold_accents("computación") == "computacion"
    assert fold_accents("recomiéndame") == "recomiendame"
    assert fold_accents("robótique") == "robotique"


@pytest.mark.parametrize(
    "question",
    [
        "recomiéndame cursos de aprendizaje automático",
        "recommande-moi des cours de robotique",
        "¿Qué cursos de ciencias de la computación hay?",
    ],
)
def test_bug09_no_broken_word_fragments(question):
    intent = rule_based_extract(normalize_question(question))
    assert intent is not None
    broken = {"computaci", "recomi", "ndame", "autom", "rob", "tica"}
    assert not (set(intent["keywords"]) & broken), f"关键词被切碎: {intent['keywords']}"


def test_bug09_spanish_ml_query_finds_ml_courses(index):
    titles = [
        c.get("title", "").lower()
        for c in _search(index, "recomiéndame cursos de aprendizaje automático")
    ]
    assert any("machine learning" in t or "learning" in t for t in titles), titles


def test_bug09_french_robotics_query_finds_robotics_courses(index):
    titles = [
        c.get("title", "").lower()
        for c in _search(index, "recommande-moi des cours de robotique")
    ]
    assert any("robot" in t for t in titles), titles


# ---------------------------------------------------------------- #20 提前 return
def test_bug20_course_code_filter_respects_other_filters():
    index = [
        {
            "course_code": "CIEN E3125",
            "all_terms": ["Spring 2026"],
            "sections_summary": [{"term": "Spring 2026", "times": "M 10:00am"}],
            "all_instructors": [],
        },
        {
            "course_code": "CIEN E3125",
            "all_terms": ["Fall 2025"],
            "sections_summary": [{"term": "Fall 2025", "times": "M 10:00am"}],
            "all_instructors": [],
        },
    ]
    result = filter_by_fields(index, {"course_codes": ["CIEN E3125"], "term": "Spring 2026"})
    assert len(result) == 1
    assert result[0]["all_terms"] == ["Spring 2026"]


# ---------------------------------------------------------------- #21 学分区间
def test_bug21_variable_credit_course_matches_point_query():
    index = [
        {"course_code": "AAAA E1000", "points_min": 1.0, "points_max": 6.0,
         "sections_summary": [], "all_instructors": [], "all_terms": []},
        {"course_code": "BBBB E1000", "points_min": 4.0, "points_max": 4.0,
         "sections_summary": [], "all_instructors": [], "all_terms": []},
    ]
    result = filter_by_fields(index, {"points_min": 3.0, "points_max": 3.0})
    codes = [e["course_code"] for e in result]
    assert "AAAA E1000" in codes, "1.0-6.0 学分的课应能匹配「3 学分」查询"
    assert "BBBB E1000" not in codes


# ---------------------------------------------------------------- 多轮指代
def _intent(question: str) -> dict:
    """
    模拟生产路径：规则引擎拿不准时会 fallback 到 LLM，
    LLM 也失败则退回默认 intent（关键词为空）。
    这里用默认 intent 代表那条分支，保证判定逻辑在两条路径下都正确。
    """
    intent = rule_based_extract(normalize_question(question))
    if intent is None:
        intent = dict(DEFAULT_INTENT)
        intent["original_question"] = question
    return intent


@pytest.mark.parametrize(
    "question",
    [
        # 物主代词 —— 旧版 \bit\b 匹配不到 "its"
        "what are its prerequisites?",
        "who are their instructors?",
        # which one / any of —— 旧版完全没覆盖
        "which one has fewer prerequisites?",
        "which of those is easier?",
        "are any of them in the morning?",
        # 省略式追问
        "recommend one for a sophomore",
        "suggest another",
        # 纯属性提问，一个指代词都没有
        "what are the prerequisites?",
        "how many credits?",
        "when does it meet?",
        # 中文
        "哪个更简单？",
        "它的先修课是什么？",
        "再推荐一门",
    ],
)
def test_deep_followup_reuses_previous_courses(question):
    """
    这些追问旧版会被当成新查询，拿 ['prerequisites'] 之类的词去全库检索，
    把 AERO 的对话拽到一堆 ELEN 课上（实测 15 轮里有 4 轮发生课程漂移）。
    """
    intent = _intent(question)
    assert server._is_referential_followup(intent, question, is_followup=True), (
        f"未被识别为回指追问: {question} (keywords={intent['keywords']})"
    )


@pytest.mark.parametrize(
    "question",
    [
        "what about civil engineering instead?",   # 新系别
        "tell me about COMS W4111",                # 新课号
        "which courses does Professor Panayotidi teach?",  # 新教授
        "recommend some robotics courses",         # 新主题
        "I want to learn about thermodynamics",    # 新主题
    ],
)
def test_new_topic_not_treated_as_followup(question):
    """有新锚点/新主题时必须重新检索，不能粘在上一轮的课程上。"""
    intent = _intent(question)
    assert not server._is_referential_followup(intent, question, is_followup=True), (
        f"被误判为回指: {question}"
    )


def test_first_turn_never_referential():
    intent = _intent("what are its prerequisites?")
    assert not server._is_referential_followup(intent, "what are its prerequisites?", is_followup=False)


# ---------------------------------------------------------------- 第二轮审查发现
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3.00 points", (3.0, 3.0)),
        ("3 points", (3.0, 3.0)),
        ("1.5-6 points", (1.5, 6.0)),
        ("1 to 4 points", (1.0, 4.0)),
        ("4.5", (4.5, 4.5)),
        ("", None),
        ("no numbers here", None),
    ],
)
def test_bug26_points_raw_parsing(raw, expected):
    from file_importer import parse_points_raw
    assert parse_points_raw(raw) == expected


def test_bug26_manual_import_points_are_searchable():
    """
    手动录入表单只收 points_raw，旧版把 points_min/max 留成 0，
    这门课就永远无法被「3 学分的课」检索到。
    """
    from file_importer import backfill_points
    filled = backfill_points({"course_code": "TEST W1234", "title": "T", "points_raw": "3.00 points"})
    assert filled["points_min"] == 3.0
    assert filled["points_max"] == 3.0

    entry = {
        "course_code": "TEST W1234", "points_min": filled["points_min"],
        "points_max": filled["points_max"], "sections_summary": [],
        "all_instructors": [], "all_terms": [],
    }
    assert filter_by_fields([entry], {"points_min": 3.0, "points_max": 3.0})


def test_bug26_backfill_does_not_clobber_existing():
    from file_importer import backfill_points
    filled = backfill_points({"points_raw": "3 points", "points_min": 1.0, "points_max": 6.0})
    assert (filled["points_min"], filled["points_max"]) == (1.0, 6.0)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("COMS W4111", ["COMS W4111"]),
        ("COMS-W4111", ["COMS W4111"]),
        ("COMSW4111", ["COMS W4111"]),
        ("coms w4111", ["COMS W4111"]),
        ("COMS  W4111", ["COMS W4111"]),
        ("COMS_W4111", ["COMS W4111"]),
    ],
)
def test_bug30_course_code_variants(question, expected):
    """用户实际会写 COMS-W4111 / COMSW4111，旧版只认单一空格分隔。"""
    intent = _intent(question)
    assert intent["course_codes"] == expected


def test_bug28_health_degraded_when_no_courses(monkeypatch):
    """0 门课时 status 必须是 degraded —— 否则用户只会看到「找不到课程」。"""
    import server as srv
    from fastapi.testclient import TestClient

    monkeypatch.setattr(config, "WARMUP_ON_STARTUP", False)
    with TestClient(srv.app) as client:
        client.app.state.enriched_index = []
        body = client.get("/api/health").json()
        assert body["status"] == "degraded"
        assert body["usable"] is False
        assert any("No courses loaded" in r for r in body["reasons"])


def test_bug29_chat_emits_meta_with_history_turns(monkeypatch):
    """
    后端历史存在内存里，--reload 时每次存盘都会清空。
    前端靠 meta.history_turns 判断上下文是否断掉并提示用户。
    """
    import server as srv
    from fastapi.testclient import TestClient

    class Fake:
        async def is_available(self, force=False): return True
        async def chat(self, m, system_prompt="", max_tokens=512, model=""): return "{}"
        async def chat_stream(self, m, system_prompt="", max_tokens=512, model=""):
            yield "ok"

    monkeypatch.setattr(config, "WARMUP_ON_STARTUP", False)
    monkeypatch.setattr(config, "INFERENCE_MODE", "local")

    def parse(resp):
        return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]

    with TestClient(srv.app) as client:
        client.app.state.ollama = Fake()
        body = {"message": "aerospace courses", "conversation_id": "meta-t", "language": "en"}
        events = parse(client.post("/api/chat", json=body))
        meta = next(e for e in events if e["type"] == "meta")
        assert meta["history_turns"] == 0, "首轮应为 0"

        events = parse(client.post("/api/chat", json=body))
        meta = next(e for e in events if e["type"] == "meta")
        assert meta["history_turns"] == 1, "第二轮应能看到 1 轮历史"

        # 模拟后端重启
        client.app.state.conversations.clear()
        events = parse(client.post("/api/chat", json=body))
        meta = next(e for e in events if e["type"] == "meta")
        assert meta["history_turns"] == 0, "重启后应回到 0，前端据此提示上下文已重置"


def test_bug33_oversized_payloads_rejected(monkeypatch):
    import server as srv
    from fastapi.testclient import TestClient

    monkeypatch.setattr(config, "WARMUP_ON_STARTUP", False)
    with TestClient(srv.app) as client:
        r = client.post("/api/chat", json={
            "message": "hi", "conversation_id": "z" * 500, "language": "en"})
        assert r.status_code == 422, "conversation_id 应限长"

        r = client.post("/api/export", json={
            "messages": [{"role": "user", "content": "x"}] * 5000, "format": "json"})
        assert r.status_code == 422, "导出条数应有上限"


# ---------------------------------------------------------------- 综合：结果相关性
def test_no_irrelevant_results_for_unanchored_queries(index):
    """没有任何锚点的问题不应硬凑课程出来。"""
    for question in ["hello", "what is the meaning of life"]:
        intent = rule_based_extract(normalize_question(question))
        if intent is None:
            continue
        courses = retrieve_courses(index, intent, str(config.COURSES_DIR), max_results=5)
        assert courses == [], f"{question} 不应返回课程，实际: {_codes(courses)}"

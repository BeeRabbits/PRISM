"""
Evaluation Test Suite: curated prompts for testing model capability.

Two categories:
  1. General capability — instruction following, structured output, reasoning.
     These must NOT degrade after any training run.
  2. Personal recall — tests whether the knowledge graph / memory injection
     provides accurate personal context.

Each test case defines:
  - prompt: the user message
  - category: "general" or "personal"
  - checks: list of quality signals to verify in the response
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TestCase:
    """A single evaluation test case."""
    id: str
    prompt: str
    category: str  # "general" or "personal"
    checks: List[str] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# General capability tests — the model MUST pass these after any training
# ---------------------------------------------------------------------------

GENERAL_TESTS: List[TestCase] = [
    TestCase(
        id="gen_structured_workout",
        prompt="Give me a 3-day workout plan for building muscle. Include exercises, sets, and reps.",
        category="general",
        checks=["has_structure", "no_repetition", "no_gibberish", "min_length_200"],
        description="Structured output with markdown formatting",
    ),
    TestCase(
        id="gen_explain_recursion",
        prompt="Explain recursion in programming to someone who knows basic Python.",
        category="general",
        checks=["has_structure", "no_repetition", "no_gibberish", "min_length_100"],
        description="Technical explanation with clarity",
    ),
    TestCase(
        id="gen_pros_cons",
        prompt="What are the pros and cons of remote work vs. office work?",
        category="general",
        checks=["has_structure", "no_repetition", "no_gibberish", "min_length_150"],
        description="Balanced reasoning with multiple viewpoints",
    ),
    TestCase(
        id="gen_recipe",
        prompt="Give me a simple recipe for chicken stir-fry with vegetables.",
        category="general",
        checks=["has_structure", "no_repetition", "no_gibberish", "min_length_100"],
        description="Step-by-step instructions",
    ),
    TestCase(
        id="gen_email",
        prompt="Write a professional email to my manager requesting a day off next Friday.",
        category="general",
        checks=["no_repetition", "no_gibberish", "min_length_50", "no_chinese"],
        description="Professional writing tone and format",
    ),
    TestCase(
        id="gen_debug",
        prompt="This Python code gives an IndexError. Why?\n\ndef get_last(items):\n    return items[len(items)]",
        category="general",
        checks=["no_repetition", "no_gibberish", "min_length_50"],
        description="Code debugging and explanation",
    ),
    TestCase(
        id="gen_compare",
        prompt="Compare AWS Lambda and EC2. When would you use each?",
        category="general",
        checks=["has_structure", "no_repetition", "no_gibberish", "min_length_100"],
        description="Technical comparison with use cases",
    ),
    TestCase(
        id="gen_story",
        prompt="Write a very short story (3-4 paragraphs) about someone who discovers they can hear plants talking.",
        category="general",
        checks=["no_repetition", "no_gibberish", "min_length_100", "no_chinese"],
        description="Creative writing coherence",
    ),
    TestCase(
        id="gen_budget",
        prompt="How should someone making $55k/year set up a basic budget? Be specific with percentages.",
        category="general",
        checks=["has_structure", "no_repetition", "no_gibberish", "min_length_100"],
        description="Practical financial advice with specifics",
    ),
    TestCase(
        id="gen_long_response",
        prompt="Explain the differences between SQL and NoSQL databases. Cover types, use cases, scaling, and when to choose each.",
        category="general",
        checks=["has_structure", "no_repetition", "no_gibberish", "min_length_200", "no_chinese"],
        description="Long-form structured response (tests tail-end coherence)",
    ),
]

# ---------------------------------------------------------------------------
# Personal recall tests — tests knowledge graph / memory injection
# ---------------------------------------------------------------------------

PERSONAL_TESTS: List[TestCase] = [
    TestCase(
        id="per_son_name",
        prompt="What's my son's name?",
        category="personal",
        checks=["contains_landen"],
        description="Direct personal fact recall",
    ),
    TestCase(
        id="per_job",
        prompt="Where do I work and what do I do there?",
        category="personal",
        checks=["contains_aws"],
        description="Career fact recall",
    ),
    TestCase(
        id="per_school",
        prompt="What school am I going to?",
        category="personal",
        checks=["contains_wgu"],
        description="Education fact recall",
    ),
    TestCase(
        id="per_project",
        prompt="What's the AI project I'm building?",
        category="personal",
        checks=["contains_prism"],
        description="Project knowledge recall",
    ),
    TestCase(
        id="per_context_workout",
        prompt="I want to work out after my shift. What time would that be for me?",
        category="personal",
        checks=["contains_2pm_or_afternoon"],
        description="Contextual reasoning from personal facts",
    ),
    TestCase(
        id="per_girlfriend",
        prompt="What's my girlfriend's name?",
        category="personal",
        checks=["contains_sam"],
        description="Relationship fact recall",
    ),
    TestCase(
        id="per_career_goal",
        prompt="What career field am I trying to transition into?",
        category="personal",
        checks=["contains_ai_ml"],
        description="Goal recall",
    ),
    TestCase(
        id="per_story_project",
        prompt="Tell me about the story I'm working on.",
        category="personal",
        checks=["contains_archangel_or_michael"],
        description="Creative project recall",
    ),
    TestCase(
        id="per_birthday",
        prompt="When is my son's birthday?",
        category="personal",
        checks=["contains_april"],
        description="Date fact recall",
    ),
    TestCase(
        id="per_hobby",
        prompt="What anime have I been watching?",
        category="personal",
        checks=["contains_solo_leveling"],
        description="Interest/hobby recall",
    ),
]


ALL_TESTS: List[TestCase] = GENERAL_TESTS + PERSONAL_TESTS

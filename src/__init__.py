"""
Healthcare AI Research Agent - Source Package

This package contains the core implementation of the Healthcare AI Research Agent,
including configuration, prompts, chains, caching, and utilities.
"""

__version__ = "1.0.0"
__author__ = "Healthcare AI Research Team"
__description__ = "A production-ready LLM/RAG application for healthcare market intelligence"

from .config import Settings, CacheType
from .prompts import (
    SYSTEM_PROMPT,
    APPROVAL_GATE_PROMPT,
    REQUEST_ANALYSIS_PROMPT,
    RESEARCH_PLANNER_PROMPT,
    EVIDENCE_ANALYST_PROMPT,
    REPORT_GENERATOR_PROMPT,
    QUALITY_CONTROL_PROMPT
)
from .chains import MarketMindAgent
from .cache_manager import CacheManager, InMemoryCache, SQLiteCache
from .utils import (
    display_safety_notice,
    display_medical_disclaimer,
    create_urgency_badge,
    format_evidence_card,
    format_structured_output,
    validate_json_schema,
    get_timestamp,
    generate_run_id,
    sanitize_input,
    extract_entities
)

__all__ = [
    # Config
    'Settings',
    'CacheType',

    # Prompts
    'SYSTEM_PROMPT',
    'APPROVAL_GATE_PROMPT',
    'REQUEST_ANALYSIS_PROMPT',
    'RESEARCH_PLANNER_PROMPT',
    'EVIDENCE_ANALYST_PROMPT',
    'REPORT_GENERATOR_PROMPT',
    'QUALITY_CONTROL_PROMPT',

    # Chains
    'MarketMindAgent',

    # Cache
    'CacheManager',
    'InMemoryCache',
    'SQLiteCache',

    # Utilities
    'display_safety_notice',
    'display_medical_disclaimer',
    'create_urgency_badge',
    'format_evidence_card',
    'format_structured_output',
    'validate_json_schema',
    'get_timestamp',
    'generate_run_id',
    'sanitize_input',
    'extract_entities'
]
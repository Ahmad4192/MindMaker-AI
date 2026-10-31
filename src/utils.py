"""
Utility functions for Healthcare AI Research Agent
"""

import os
import re
import json
import uuid
import html
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import hashlib

import streamlit as st
from jsonschema import validate, ValidationError
from langchain_core.messages import BaseMessage

# ============================================================================
# SAFETY AND DISCLAIMER FUNCTIONS
# ============================================================================

def display_safety_notice():
    """Display safety notice in Streamlit"""
    st.markdown("""
    <div class="disclaimer-box">
        <h4>🚨 SAFETY NOTICE</h4>
        <p><strong>This is an AI research assistant for healthcare market intelligence.</strong></p>
        <ul>
            <li>All outputs are for <strong>research and educational purposes only</strong>.</li>
            <li>Do <strong>NOT</strong> use this system for medical diagnosis or treatment decisions.</li>
            <li>Always verify information with <strong>authoritative medical sources</strong>.</li>
            <li>Consult with <strong>qualified healthcare professionals</strong> for medical advice.</li>
            <li>This system may produce <strong>inaccurate or incomplete</strong> information.</li>
            <li>Human oversight and verification are <strong>always required</strong>.</li>
        </ul>
        <p><strong>You are responsible for validating all outputs before use.</strong></p>
    </div>
    """, unsafe_allow_html=True)

def display_medical_disclaimer():
    """Display medical disclaimer in Streamlit"""
    st.markdown("""
    <div class="disclaimer-box" style="background-color: #FFCDD2; border-left: 4px solid #F44336;">
        <h4>⚕️ MEDICAL DISCLAIMER</h4>
        <p><strong>IMPORTANT: This system does not provide medical advice, diagnosis, or treatment.</strong></p>
        <ul>
            <li>Information provided is for <strong>research purposes only</strong>.</li>
            <li>AI-generated content may contain <strong>errors or inaccuracies</strong>.</li>
            <li>Never rely solely on AI outputs for <strong>healthcare decisions</strong>.</li>
            <li>Always consult with <strong>licensed healthcare professionals</strong>.</li>
            <li>Verify all information with <strong>peer-reviewed medical literature</strong>.</li>
            <li>This system is <strong>not a substitute</strong> for professional medical judgment.</li>
        </ul>
        <p><strong>By using this system, you agree to hold the developers harmless from any consequences.</strong></p>
    </div>
    """, unsafe_allow_html=True)

def get_medical_disclaimer_text() -> str:
    """Get medical disclaimer as plain text"""
    return """
    ⚠️  MEDICAL DISCLAIMER

    IMPORTANT: This system does not provide medical advice, diagnosis, or treatment.

    - Information provided is for research purposes only.
    - AI-generated content may contain errors or inaccuracies.
    - Never rely solely on AI outputs for healthcare decisions.
    - Always consult with licensed healthcare professionals.
    - Verify all information with peer-reviewed medical literature.
    - This system is not a substitute for professional medical judgment.

    By using this system, you agree to hold the developers harmless from any consequences.
    """

# ============================================================================
# URGENCY BADGE FUNCTIONS
# ============================================================================

def create_urgency_badge(urgency: str, as_html: bool = True) -> str:
    """
    Create an urgency badge for display

    Args:
        urgency: Urgency level (LOW, MEDIUM, HIGH, EMERGENCY)
        as_html: Return as HTML (True) or plain text (False)

    Returns:
        Formatted urgency badge
    """
    urgency = urgency.upper()

    if as_html:
        if urgency == "LOW":
            return '<span class="urgency-badge urgency-low">LOW</span>'
        elif urgency == "MEDIUM":
            return '<span class="urgency-badge urgency-medium">MEDIUM</span>'
        elif urgency == "HIGH":
            return '<span class="urgency-badge urgency-high">HIGH</span>'
        elif urgency == "EMERGENCY":
            return '<span class="urgency-badge urgency-emergency">EMERGENCY</span>'
        else:
            return '<span class="urgency-badge">UNKNOWN</span>'
    else:
        if urgency == "LOW":
            return "[LOW]"
        elif urgency == "MEDIUM":
            return "[MEDIUM]"
        elif urgency == "HIGH":
            return "[HIGH]"
        elif urgency == "EMERGENCY":
            return "[EMERGENCY]"
        else:
            return "[UNKNOWN]"

def get_urgency_color(urgency: str) -> str:
    """
    Get color for urgency level

    Args:
        urgency: Urgency level

    Returns:
        Color hex code
    """
    urgency = urgency.upper()
    colors = {
        "LOW": "#4CAF50",      # Green
        "MEDIUM": "#FF9800",   # Orange
        "HIGH": "#F44336",     # Red
        "EMERGENCY": "#9C27B0" # Purple
    }
    return colors.get(urgency, "#888888")  # Gray for unknown

# ============================================================================
# FORMATTING FUNCTIONS
# ============================================================================

def format_evidence_card(evidence: Dict[str, Any], index: int = 1) -> str:
    """
    Format evidence as a card for display

    Args:
        evidence: Evidence record
        index: Card index

    Returns:
        Formatted evidence card HTML
    """
    urgency_badge = create_urgency_badge(evidence.get("urgency", "MEDIUM"))
    claim_type_color = get_claim_type_color(evidence.get("claim_type", "fact"))

    return f"""
    <div class="evidence-card">
        <h4>Evidence #{index} {urgency_badge}</h4>
        <p><strong>Claim:</strong> {evidence.get('claim', 'N/A')}</p>
        <p><strong>Type:</strong> <span style="color: {claim_type_color}; font-weight: bold;">
            {evidence.get('claim_type', 'N/A').upper()}
        </span></p>
        <p><strong>Confidence:</strong> {evidence.get('confidence', 'N/A').upper()}</p>
        <p><strong>Source:</strong> {evidence.get('source_detail', {{}}).get('title', 'N/A')}</p>
        <p><strong>Source ID:</strong> {evidence.get('source_ref', 'N/A')}</p>
        {f'<p><strong>Notes:</strong> {evidence.get("analyst_notes", "N/A")}</p>' if evidence.get("analyst_notes") else ''}
    </div>
    """

def get_claim_type_color(claim_type: str) -> str:
    """
    Get color for claim type

    Args:
        claim_type: Claim type (fact, inference, recommendation, uncertainty)

    Returns:
        Color hex code
    """
    colors = {
        "fact": "#4CAF50",      # Green
        "inference": "#FF9800", # Orange
        "recommendation": "#2196F3", # Blue
        "uncertainty": "#F44336"  # Red
    }
    return colors.get(claim_type, "#888888")  # Gray for unknown

def format_structured_output(data: Any, indent: int = 2) -> str:
    """
    Format structured output (dict/list) as pretty JSON

    Args:
        data: Data to format
        indent: Indentation level

    Returns:
        Pretty JSON string
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except:
        return str(data)

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Validate data against JSON schema

    Args:
        data: Data to validate
        schema: JSON schema

    Returns:
        List of validation errors
    """
    errors = []

    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        errors.append(f"Validation error: {e.message}")
        if e.path:
            errors.append(f"Path: {'.'.join(map(str, e.path))}")
        if e.validator:
            errors.append(f"Validator: {e.validator}")
    except Exception as e:
        errors.append(f"Schema validation failed: {str(e)}")

    return errors

def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """
    Validate that all required fields are present

    Args:
        data: Data to validate
        required_fields: List of required field names

    Returns:
        List of missing fields
    """
    missing = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing.append(field)
    return missing

# ============================================================================
# TIME AND ID FUNCTIONS
# ============================================================================

def get_timestamp() -> str:
    """Get current timestamp in ISO8601 format"""
    return datetime.now().isoformat()

def generate_run_id() -> str:
    """Generate a unique run ID"""
    return f"run_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"

def generate_evidence_id(run_id: str, index: int) -> str:
    """Generate a unique evidence ID"""
    return f"evidence_{index}_{run_id[:8]}"

# ============================================================================
# INPUT PROCESSING FUNCTIONS
# ============================================================================

def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize input text for safety

    Args:
        text: Input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Trim to max length
    text = text[:max_length]

    # Remove potentially harmful characters
    # Keep basic punctuation and alphanumeric
    text = re.sub(r'[<>]', '', text)  # Remove angle brackets

    # Escape HTML
    text = html.escape(text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_entities(text: str) -> Dict[str, str]:
    """
    Extract entities from text (simple implementation)

    Args:
        text: Input text

    Returns:
        Dictionary of extracted entities
    """
    entities = {
        "condition": None,
        "treatment": None,
        "technology": None,
        "market": None,
        "specialty": None,
        "drug": None,
        "company": None
    }

    # Simple keyword matching
    text_lower = text.lower()

    # Conditions
    conditions = ["cancer", "diabetes", "heart disease", "alzheimer", "covid", "depression"]
    for condition in conditions:
        if condition in text_lower:
            entities["condition"] = condition.title()
            break

    # Treatments
    treatments = ["chemotherapy", "radiation", "surgery", "therapy", "vaccine", "medication"]
    for treatment in treatments:
        if treatment in text_lower:
            entities["treatment"] = treatment.title()
            break

    # Technologies
    technologies = ["ai", "machine learning", "deep learning", "neural network", "computer vision"]
    for tech in technologies:
        if tech in text_lower:
            entities["technology"] = tech.title()
            break

    # Markets
    markets = ["healthcare", "pharmaceutical", "biotech", "medical devices", "diagnostics"]
    for market in markets:
        if market in text_lower:
            entities["market"] = market.title()
            break

    # Specialties
    specialties = ["oncology", "cardiology", "neurology", "radiology", "pathology"]
    for specialty in specialties:
        if specialty in text_lower:
            entities["specialty"] = specialty.title()
            break

    return entities

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ============================================================================
# FILE AND PATH FUNCTIONS
# ============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_json_file(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Read JSON file

    Args:
        path: File path

    Returns:
        Parsed JSON data or None
    """
    try:
        path = Path(path)
        if not path.exists():
            return None

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

def write_json_file(path: Union[str, Path], data: Any) -> bool:
    """
    Write JSON file

    Args:
        path: File path
        data: Data to write

    Returns:
        True if successful
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        return False

# ============================================================================
# HASHING AND ENCRYPTION FUNCTIONS
# ============================================================================

def generate_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Generate hash of text

    Args:
        text: Text to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)

    Returns:
        Hash string
    """
    if algorithm == "md5":
        return hashlib.md5(text.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(text.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(text.encode()).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(text.encode()).hexdigest()
    else:
        raise ValueError(f"Unknown hash algorithm: {algorithm}")

def anonymize_text(text: str) -> str:
    """
    Anonymize text by replacing potential PII

    Args:
        text: Input text

    Returns:
        Anonymized text
    """
    # Replace common PII patterns
    patterns = [
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),           # Social Security Number
        (r'\b\d{4}-\d{4}-\d{4}-\d{4}\b', '[CC]'),      # Credit Card
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
        (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]'),         # Phone number
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]')  # IP address
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)

    return text

# ============================================================================
# STREAMLIT UTILITY FUNCTIONS
# ============================================================================

def create_streamlit_metric(label: str, value: Any, delta: Optional[str] = None, icon: Optional[str] = None):
    """
    Create a Streamlit metric with optional icon

    Args:
        label: Metric label
        value: Metric value
        delta: Delta value
        icon: Optional icon emoji

    Returns:
        Streamlit metric
    """
    if icon:
        label = f"{icon} {label}"

    if delta:
        return st.metric(label, value, delta)
    else:
        return st.metric(label, value)

def create_streamlit_card(title: str, content: str, icon: Optional[str] = None):
    """
    Create a Streamlit card with title and content

    Args:
        title: Card title
        content: Card content
        icon: Optional icon emoji

    Returns:
        Streamlit container with card styling
    """
    if icon:
        title = f"{icon} {title}"

    with st.container():
        st.markdown(f"### {title}")
        st.markdown(content)
        st.markdown("---")

def create_streamlit_section(title: str, icon: Optional[str] = None):
    """
    Create a Streamlit section header

    Args:
        title: Section title
        icon: Optional icon emoji
    """
    if icon:
        title = f"{icon} {title}"

    st.markdown(f"## {title}")
    st.markdown("---")

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator between keys

    Returns:
        Flattened dictionary
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result

def filter_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Filter dictionary to include only specified keys

    Args:
        d: Dictionary to filter
        keys: Keys to include

    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k in keys}

# ============================================================================
# ERROR HANDLING FUNCTIONS
# ============================================================================

def safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary values

    Args:
        d: Dictionary
        *keys: Keys to traverse
        default: Default value if key not found

    Returns:
        Value or default
    """
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def safe_call(func, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """
    Safely call a function with error handling

    Args:
        func: Function to call
        *args: Positional arguments
        default: Default value if error occurs
        **kwargs: Keyword arguments

    Returns:
        Function result or default
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")
        return default

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def log_message(message: str, level: str = "INFO", timestamp: Optional[str] = None):
    """
    Log a message with timestamp and level

    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, DEBUG)
        timestamp: Optional timestamp
    """
    timestamp = timestamp or get_timestamp()
    print(f"[{timestamp}] [{level}] {message}")

def log_function_call(func_name: str, args: Optional[Dict[str, Any]] = None, result: Optional[Any] = None):
    """
    Log a function call with arguments and result

    Args:
        func_name: Function name
        args: Function arguments
        result: Function result
    """
    timestamp = get_timestamp()
    log_message(f"Function call: {func_name}", level="DEBUG", timestamp=timestamp)

    if args:
        # Truncate args for logging
        args_str = str(args)[:200]
        log_message(f"  Args: {args_str}", level="DEBUG", timestamp=timestamp)

    if result is not None:
        # Truncate result for logging
        result_str = str(result)[:200]
        log_message(f"  Result: {result_str}", level="DEBUG", timestamp=timestamp)

# ============================================================================
# TYPE CONVERSION FUNCTIONS
# ============================================================================

def to_snake_case(text: str) -> str:
    """
    Convert text to snake_case

    Args:
        text: Input text

    Returns:
        Snake case text
    """
    # Replace spaces and hyphens with underscores
    text = re.sub(r'[\s-]', '_', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-z0-9_]', '', text)
    # Remove duplicate underscores
    text = re.sub(r'_+', '_', text)
    # Remove leading/trailing underscores
    text = text.strip('_')
    return text

def to_camel_case(text: str) -> str:
    """
    Convert text to camelCase

    Args:
        text: Input text

    Returns:
        Camel case text
    """
    # Split into words
    words = re.split(r'[\s_-]', text)
    # Capitalize each word except first
    words = [words[0].lower()] + [word.capitalize() for word in words[1:]]
    # Join
    return ''.join(words)

def to_pascal_case(text: str) -> str:
    """
    Convert text to PascalCase

    Args:
        text: Input text

    Returns:
        Pascal case text
    """
    # Split into words
    words = re.split(r'[\s_-]', text)
    # Capitalize each word
    words = [word.capitalize() for word in words]
    # Join
    return ''.join(words)

def to_kebab_case(text: str) -> str:
    """
    Convert text to kebab-case

    Args:
        text: Input text

    Returns:
        Kebab case text
    """
    # Replace underscores with hyphens
    text = re.sub(r'_', '-', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-z0-9-]', '', text)
    # Remove duplicate hyphens
    text = re.sub(r'-+', '-', text)
    # Remove leading/trailing hyphens
    text = text.strip('-')
    return text

# ============================================================================
# EXPORT ALL FUNCTIONS
# ============================================================================

__all__ = [
    # Safety and Disclaimers
    'display_safety_notice',
    'display_medical_disclaimer',
    'get_medical_disclaimer_text',

    # Urgency and Formatting
    'create_urgency_badge',
    'get_urgency_color',
    'format_evidence_card',
    'get_claim_type_color',
    'format_structured_output',

    # Validation
    'validate_json_schema',
    'validate_required_fields',

    # Time and ID
    'get_timestamp',
    'generate_run_id',
    'generate_evidence_id',

    # Input Processing
    'sanitize_input',
    'extract_entities',
    'normalize_text',

    # File and Path
    'ensure_directory',
    'read_json_file',
    'write_json_file',

    # Hashing
    'generate_hash',
    'anonymize_text',

    # Streamlit
    'create_streamlit_metric',
    'create_streamlit_card',
    'create_streamlit_section',

    # Data Processing
    'flatten_dict',
    'chunk_list',
    'merge_dicts',
    'filter_dict',

    # Error Handling
    'safe_get',
    'safe_call',

    # Logging
    'log_message',
    'log_function_call',

    # Type Conversion
    'to_snake_case',
    'to_camel_case',
    'to_pascal_case',
    'to_kebab_case'
]
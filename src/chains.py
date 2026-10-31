"""
LangChain chains and agent implementation for Healthcare AI Research Agent
"""

import os
import json
import uuid
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import asyncio
from pathlib import Path
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.tools import Tool
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler
from langchain_openai import ChatOpenAI
from .config import Settings, CacheType
from .prompts import (
    SYSTEM_PROMPT,
    REQUEST_ANALYSIS_PROMPT,
    RESEARCH_PLANNER_PROMPT,
    EVIDENCE_ANALYST_PROMPT,
    REPORT_GENERATOR_PROMPT,
    QUALITY_CONTROL_PROMPT,
    get_schema_by_name
)
from .cache_manager import CacheManager
from .utils import (
    validate_json_schema,
    generate_run_id,
    get_timestamp,
    sanitize_input,
    extract_entities
)


class HealthcareAICallbackHandler(BaseCallbackHandler):
    """Custom callback handler for Healthcare AI Research Agent"""

    def __init__(self):
        super().__init__()
        self.token_count = 0
        self.start_time = None
        self.end_time = None
        self.tool_calls = []
        self.model_calls = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Called when LLM starts"""
        self.start_time = time.time()
        self.model_calls.append({
            "timestamp": get_timestamp(),
            "model": serialized.get("name", "unknown"),
            "prompt_length": sum(len(p) for p in prompts)
        })

    def on_llm_end(self, response: Any, **kwargs):
        """Called when LLM ends"""
        self.end_time = time.time()

    def on_llm_new_token(self, token: str, **kwargs):
        """Called for each new token"""
        self.token_count += 1

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Called when tool starts"""
        self.tool_calls.append({
            "timestamp": get_timestamp(),
            "tool": serialized.get("name", "unknown"),
            "input": input_str
        })

    def on_tool_end(self, output: str, **kwargs):
        """Called when tool ends"""
        if self.tool_calls:
            self.tool_calls[-1]["output"] = output

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        return {
            "token_count": self.token_count,
            "duration_seconds": duration,
            "tool_calls": len(self.tool_calls),
            "model_calls": len(self.model_calls),
            "tool_call_details": self.tool_calls,
            "model_call_details": self.model_calls
        }


class MarketMindAgent:
    """
    Main agent class for Healthcare AI Research Agent

    Implements the complete research workflow:
    1. Request Analysis
    2. Research Planning
    3. Evidence Collection
    4. Report Generation
    5. Quality Control
    6. Approval Gate
    """

    def __init__(self, settings: Settings, cache_manager: CacheManager):
        """
        Initialize the MarketMind Agent

        Args:
            settings: Application settings
            cache_manager: Cache manager for dual caching
        """
        self.settings = settings
        self.cache_manager = cache_manager
        self.run_id = None
        self.callback_handler = HealthcareAICallbackHandler()

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Initialize tools
        self.tools = self._initialize_tools()

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create run directory
        self.runs_dir = Path("runs")
        self.runs_dir.mkdir(exist_ok=True)

        # State tracking
        self.state = {
            "run_id": None,
            "research_plan": None,
            "evidence_store": [],
            "tool_call_history": [],
            "iteration_count": 0,
            "cumulative_tokens": 0,
            "cumulative_cost": 0.0,
            "defect_list": [],
            "current_stage": "initialized"
        }

    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the ChatOpenAI LLM with safety fallback attributes"""
        api_key = getattr(self.settings, "openai_api_key", None) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY.")

        model_name = getattr(self.settings, "model_name", "gpt-4o-mini")
        temperature = getattr(self.settings, "temperature", 0.2)

        # Settings کے محفوظ پیرامیٹرز حاصل کریں
        model_cfg = getattr(self.settings, "model", None)
        max_tokens = getattr(model_cfg, "max_tokens", 2000) if model_cfg else getattr(self.settings, "max_tokens", 2000)
        top_p = getattr(model_cfg, "top_p", 1.0) if model_cfg else 1.0
        freq_penalty = getattr(model_cfg, "frequency_penalty", 0.0) if model_cfg else 0.0
        pres_penalty = getattr(model_cfg, "presence_penalty", 0.0) if model_cfg else 0.0

        return ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=freq_penalty,
            presence_penalty=pres_penalty,
            timeout=60,
            max_retries=3,
            callback_manager=CallbackManager([self.callback_handler])
        )

    def _initialize_tools(self) -> List[Tool]:
        """Initialize all available tools"""
        tools = []

        # Search Information Tool
        tools.append(Tool(
            name="search_information",
            description="""
            Search for healthcare-related information from approved sources.
            Use this when you need to find documents, articles, or data about:
            - Medical treatments and procedures
            - Healthcare technologies and AI applications
            - Market data and trends
            - Clinical studies and trials
            - Regulatory information

            Args:
                query (str): The search query
                source_type (str): Type of source (medical_journals, market_reports, clinical_trials, news, all)
                max_results (int): Maximum number of results (1-20, default: 10)
                recency_window (str): Time window (1_month, 3_months, 6_months, 1_year, 2_years, all)

            Returns:
                List of search results with id, title, source, date, and snippet
            """,
            func=self.search_information
        ))

        # Retrieve Document Tool
        tools.append(Tool(
            name="retrieve_document",
            description="""
            Retrieve the full content of a specific document by its ID.
            Use this when you need to read the complete content of a document
            that was found through search_information.

            Args:
                document_id (str): The ID of the document to retrieve
                section (str, optional): Specific section to retrieve

            Returns:
                Full document content with metadata
            """,
            func=self.retrieve_document
        ))

        # Calculate Metric Tool
        tools.append(Tool(
            name="calculate_metric",
            description="""
            Perform calculations on numerical data.
            Use this for quantitative analysis like:
            - Growth rates
            - Market share calculations
            - Statistical analysis
            - Financial projections

            Args:
                operation (str): Type of calculation (growth_rate, cagr, share, ratio, average, sum)
                values (List[float]): Array of numerical values
                period (int, optional): Time period for calculations

            Returns:
                Calculation result with formula and inputs
            """,
            func=self.calculate_metric
        ))

        # Compare Companies Tool
        tools.append(Tool(
            name="compare_companies",
            description="""
            Compare healthcare companies or organizations in a structured matrix.
            Use this when you need to compare entities across multiple attributes.

            Args:
                entities (List[str]): List of company/organization names
                attributes (List[str]): List of attributes to compare
                    (revenue, market_share, fda_approvals, clinical_trials, pricing, technology, etc.)

            Returns:
                Structured comparison matrix with entity x attribute values
            """,
            func=self.compare_companies
        ))

        # Save Research Tool
        tools.append(Tool(
            name="save_research",
            description="""
            Save evidence and findings to the evidence store.
            Use this to record important information with proper provenance.

            Args:
                question_id (str): The ID of the research question this addresses
                claim (str): The statement or finding to save
                claim_type (str): Type of claim (fact, inference, recommendation, uncertainty)
                source_ref (str): Reference to the source tool result
                source_kind (str): Type of source (retrieved_document, search_result, calculation, model_generated)
                confidence (str): Confidence level (high, medium, low)
                source_detail (Dict, optional): Additional source metadata

            Returns:
                Saved evidence record with ID
            """,
            func=self.save_research
        ))

        # Generate Report Tool
        tools.append(Tool(
            name="generate_report",
            description="""
            Generate the final research report from all gathered evidence.
            Use this at the end of the research process to compile all findings.

            Args:
                run_id (str): The run ID for this research session
                sections (List[str], optional): Specific sections to include

            Returns:
                Complete research report in structured format
            """,
            func=self.generate_report
        ))

        return tools

    def _get_tool_by_name(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def reset_state(self, run_id: Optional[str] = None):
        """Reset agent state for a new run"""
        self.run_id = run_id or generate_run_id()
        self.state = {
            "run_id": self.run_id,
            "research_plan": None,
            "evidence_store": [],
            "tool_call_history": [],
            "iteration_count": 0,
            "cumulative_tokens": 0,
            "cumulative_cost": 0.0,
            "defect_list": [],
            "current_stage": "initialized"
        }
        self.callback_handler = HealthcareAICallbackHandler()
        return self.run_id

    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO8601 format"""
        return datetime.now().isoformat()

    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================

    def search_information(
        self,
        query: str,
        source_type: str = "all",
        max_results: int = 10,
        recency_window: str = "1_year"
    ) -> List[Dict[str, Any]]:
        """Search for healthcare-related information"""
        if max_results < 1 or max_results > 50:
            raise ValueError("max_results must be between 1 and 50")

        valid_source_types = ["medical_journals", "market_reports", "clinical_trials", "news", "all"]
        if source_type not in valid_source_types:
            raise ValueError(f"source_type must be one of {valid_source_types}")

        valid_recency_windows = ["1_month", "3_months", "6_months", "1_year", "2_years", "all"]
        if recency_window not in valid_recency_windows:
            raise ValueError(f"recency_window must be one of {valid_recency_windows}")

        query = sanitize_input(query)

        cache_key = f"search:{query}:{source_type}:{max_results}:{recency_window}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        search_results = self._mock_search(query, source_type, max_results, recency_window)

        self.cache_manager.set(cache_key, search_results)

        tool_call = {
            "timestamp": self.get_current_timestamp(),
            "tool": "search_information",
            "input": {"query": query, "source_type": source_type, "max_results": max_results, "recency_window": recency_window},
            "output": search_results,
            "run_id": self.state["run_id"]
        }
        self.state["tool_call_history"].append(tool_call)

        return search_results

    def retrieve_document(
        self,
        document_id: str,
        section: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve full document content by ID"""
        if not document_id:
            raise ValueError("document_id is required")

        cache_key = f"document:{document_id}:{section or 'full'}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        document = self._mock_retrieve_document(document_id, section)

        if not document:
            raise ValueError(f"Document with ID {document_id} not found")

        self.cache_manager.set(cache_key, document)

        tool_call = {
            "timestamp": self.get_current_timestamp(),
            "tool": "retrieve_document",
            "input": {"document_id": document_id, "section": section},
            "output": document,
            "run_id": self.state["run_id"]
        }
        self.state["tool_call_history"].append(tool_call)

        return document

    def calculate_metric(
        self,
        operation: str,
        values: List[float],
        period: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform calculations on numerical data"""
        if not values:
            raise ValueError("values array cannot be empty")

        valid_operations = ["growth_rate", "cagr", "share", "ratio", "average", "sum"]
        if operation not in valid_operations:
            raise ValueError(f"operation must be one of {valid_operations}")

        cache_key = f"calculation:{operation}:{json.dumps(values)}:{period or 0}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        result = self._perform_calculation(operation, values, period)

        self.cache_manager.set(cache_key, result)

        tool_call = {
            "timestamp": self.get_current_timestamp(),
            "tool": "calculate_metric",
            "input": {"operation": operation, "values": values, "period": period},
            "output": result,
            "run_id": self.state["run_id"]
        }
        self.state["tool_call_history"].append(tool_call)

        return result

    def compare_companies(
        self,
        entities: List[str],
        attributes: List[str]
    ) -> Dict[str, Any]:
        """Compare companies in a structured matrix"""
        if not entities:
            raise ValueError("entities list cannot be empty")

        if not attributes:
            raise ValueError("attributes list cannot be empty")

        if len(entities) > 20:
            raise ValueError("Cannot compare more than 20 entities")

        if len(attributes) > 15:
            raise ValueError("Cannot compare more than 15 attributes")

        cache_key = f"comparison:{json.dumps(entities)}:{json.dumps(attributes)}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        comparison = self._mock_compare_companies(entities, attributes)

        self.cache_manager.set(cache_key, comparison)

        tool_call = {
            "timestamp": self.get_current_timestamp(),
            "tool": "compare_companies",
            "input": {"entities": entities, "attributes": attributes},
            "output": comparison,
            "run_id": self.state["run_id"]
        }
        self.state["tool_call_history"].append(tool_call)

        return comparison

    def save_research(
        self,
        question_id: str,
        claim: str,
        claim_type: str,
        source_ref: str,
        source_kind: str,
        confidence: str,
        source_detail: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Save evidence and findings to the evidence store"""
        valid_claim_types = ["fact", "inference", "recommendation", "uncertainty"]
        if claim_type not in valid_claim_types:
            raise ValueError(f"claim_type must be one of {valid_claim_types}")

        valid_source_kinds = ["retrieved_document", "search_result", "calculation", "model_generated"]
        if source_kind not in valid_source_kinds:
            raise ValueError(f"source_kind must be one of {valid_source_kinds}")

        valid_confidence_levels = ["high", "medium", "low"]
        if confidence not in valid_confidence_levels:
            raise ValueError(f"confidence must be one of {valid_confidence_levels}")

        if source_ref:
            source_found = any(
                tc.get("output", {}).get("id") == source_ref or
                tc.get("output", {}).get("document_id") == source_ref or
                tc.get("input", {}).get("document_id") == source_ref
                for tc in self.state["tool_call_history"]
            )
            if not source_found and not source_ref.startswith("calculation_") and not source_ref.startswith("comparison_"):
                pass

        evidence_id = f"evidence_{len(self.state['evidence_store']) + 1}_{str(self.state['run_id'])[:8]}"

        evidence_record = {
            "evidence_id": evidence_id,
            "question_id": question_id,
            "claim": claim,
            "claim_type": claim_type,
            "source_ref": source_ref,
            "source_kind": source_kind,
            "source_detail": source_detail or {},
            "credibility": "medium",
            "recency": "current",
            "relevance": "high",
            "corroboration": [],
            "confidence": confidence,
            "analyst_notes": "Saved via save_research tool",
            "urgency": "MEDIUM",
            "timestamp": self.get_current_timestamp()
        }

        self.state["evidence_store"].append(evidence_record)

        tool_call = {
            "timestamp": self.get_current_timestamp(),
            "tool": "save_research",
            "input": {
                "question_id": question_id,
                "claim": claim,
                "claim_type": claim_type,
                "source_ref": source_ref,
                "source_kind": source_kind,
                "confidence": confidence
            },
            "output": evidence_record,
            "run_id": self.state["run_id"]
        }
        self.state["tool_call_history"].append(tool_call)

        return evidence_record

    def generate_report(
        self,
        run_id: str,
        sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate the final research report"""
        if run_id != self.state["run_id"]:
            raise ValueError(f"Invalid run_id. Current run_id is {self.state['run_id']}")

        if not self.state["research_plan"]:
            raise ValueError("Research plan not found. Please run research execution first.")

        if not self.state["evidence_store"]:
            raise ValueError("No evidence collected. Please run research execution first.")

        cache_key = f"report:{run_id}:{json.dumps(sections or [])}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        report = self._generate_report_content(sections)

        self.cache_manager.set(cache_key, report)

        tool_call = {
            "timestamp": self.get_current_timestamp(),
            "tool": "generate_report",
            "input": {"run_id": run_id, "sections": sections},
            "output": report,
            "run_id": self.state["run_id"]
        }
        self.state["tool_call_history"].append(tool_call)

        return report

    # =========================================================================
    # MOCK IMPLEMENTATIONS
    # =========================================================================

    def _mock_search(
        self,
        query: str,
        source_type: str,
        max_results: int,
        recency_window: str
    ) -> List[Dict[str, Any]]:
        """Mock search implementation"""
        entities = extract_entities(query)
        mock_results = []
        i = 1

        medical_journals = [
            {"id": f"mj_{i}", "title": f"Study on {entities.get('condition', 'AI')} in {entities.get('specialty', 'Healthcare')}", 
             "source": "New England Journal of Medicine", "date": "2024-03-15", 
             "snippet": f"Comprehensive study examining {query[:100]}...", "type": "medical_journal"},
            {"id": f"mj_{i+1}", "title": f"Clinical Trial Results for {entities.get('treatment', 'New Therapy')}", 
             "source": "The Lancet", "date": "2024-01-20", 
             "snippet": f"Phase III trial results showing {query[:100]}...", "type": "medical_journal"},
        ]

        market_reports = [
            {"id": f"mr_{i}", "title": f"Global {entities.get('market', 'Healthcare AI')} Market Analysis", 
             "source": "McKinsey & Company", "date": "2024-02-28", 
             "snippet": f"Market analysis covering {query[:100]}...", "type": "market_report"},
            {"id": f"mr_{i+1}", "title": f"{entities.get('technology', 'AI')} Adoption in Healthcare", 
             "source": "Deloitte", "date": "2024-01-10", 
             "snippet": f"Adoption trends and projections for {query[:100]}...", "type": "market_report"},
        ]

        clinical_trials = [
            {"id": f"ct_{i}", "title": f"Phase II Trial of {entities.get('drug', 'New Drug')} for {entities.get('condition', 'Disease')}", 
             "source": "ClinicalTrials.gov", "date": "2024-04-01", 
             "snippet": f"Ongoing trial investigating {query[:100]}...", "type": "clinical_trial"},
        ]

        news = [
            {"id": f"news_{i}", "title": f"FDA Approves {entities.get('product', 'New AI Tool')} for {entities.get('use_case', 'Diagnostics')}", 
             "source": "Healthcare IT News", "date": "2024-03-01", 
             "snippet": f"Regulatory approval announced for {query[:100]}...", "type": "news"},
        ]

        all_results = []
        if source_type in ["medical_journals", "all"]:
            all_results.extend(medical_journals)
        if source_type in ["market_reports", "all"]:
            all_results.extend(market_reports)
        if source_type in ["clinical_trials", "all"]:
            all_results.extend(clinical_trials)
        if source_type in ["news", "all"]:
            all_results.extend(news)

        return all_results[:max_results]

    def _mock_retrieve_document(
        self,
        document_id: str,
        section: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mock document retrieval"""
        mock_documents = {
            "mj_1": {
                "document_id": "mj_1",
                "title": "Study on AI Applications in Clinical Diagnostics",
                "source": "New England Journal of Medicine",
                "date": "2024-03-15",
                "authors": ["Dr. Smith", "Dr. Johnson"],
                "content": {
                    "abstract": "This study examines the effectiveness of AI applications in clinical diagnostics...",
                    "introduction": "Artificial intelligence has transformed clinical diagnostics...",
                    "methodology": "We conducted a comprehensive analysis of AI diagnostic tools...",
                    "results": "Our findings show that AI tools achieved 95% accuracy...",
                    "conclusion": "AI applications in clinical diagnostics show great promise..."
                },
                "type": "medical_journal"
            },
            "mr_1": {
                "document_id": "mr_1",
                "title": "Global Healthcare AI Market Analysis",
                "source": "McKinsey & Company",
                "date": "2024-02-28",
                "authors": ["McKinsey Healthcare Team"],
                "content": {
                    "executive_summary": "The global healthcare AI market is projected to reach $45.2B by 2026...",
                    "market_size": "Current market size is $14.6B with CAGR of 28.5%...",
                    "key_players": ["IBM Watson Health", "Google Health", "Microsoft Healthcare"],
                    "trends": ["Increased adoption of predictive analytics", "Growth in telemedicine AI"],
                    "challenges": ["Regulatory hurdles", "Data privacy concerns"]
                },
                "type": "market_report"
            },
            "ct_1": {
                "document_id": "ct_1",
                "title": "Phase II Trial of AI Diagnostic Tool",
                "source": "ClinicalTrials.gov",
                "date": "2024-04-01",
                "sponsor": "MedTech Innovations",
                "content": {
                    "summary": "Phase II trial evaluating AI diagnostic tool for early disease detection...",
                    "design": "Randomized, double-blind study with 500 participants...",
                    "primary_outcome": "Diagnostic accuracy compared to traditional methods...",
                    "status": "Recruiting",
                    "locations": ["Boston, MA", "New York, NY", "San Francisco, CA"]
                },
                "type": "clinical_trial"
            }
        }

        document = mock_documents.get(document_id)
        if not document:
            return {
                "error": f"Document with ID {document_id} not found",
                "available_documents": list(mock_documents.keys())
            }

        if section and section in document["content"]:
            return {
                **document,
                "content": {section: document["content"][section]},
                "retrieved_section": section
            }
        else:
            return document

    def _perform_calculation(
        self,
        operation: str,
        values: List[float],
        period: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform actual calculations"""
        try:
            if operation == "sum":
                result = sum(values)
                formula = f"sum({', '.join(map(str, values))})"
            elif operation == "average":
                result = sum(values) / len(values)
                formula = f"average({', '.join(map(str, values))})"
            elif operation == "growth_rate":
                if len(values) != 2:
                    raise ValueError("Growth rate requires exactly 2 values")
                result = ((values[1] - values[0]) / values[0]) * 100
                formula = f"((({values[1]} - {values[0]}) / {values[0]}) * 100)"
            elif operation == "cagr":
                if len(values) != 2 or not period:
                    raise ValueError("CAGR requires exactly 2 values and a period")
                result = ((values[1] / values[0]) ** (1 / period) - 1) * 100
                formula = f"((({values[1]} / {values[0]}) ^ (1 / {period})) - 1) * 100"
            elif operation == "share":
                total = sum(values)
                result = [(v / total) * 100 for v in values]
                formula = f"[value / {total} * 100 for value in {values}]"
            elif operation == "ratio":
                if len(values) != 2:
                    raise ValueError("Ratio requires exactly 2 values")
                result = values[0] / values[1]
                formula = f"({values[0]} / {values[1]})"
            else:
                raise ValueError(f"Unknown operation: {operation}")

            return {
                "operation": operation,
                "values": values,
                "period": period,
                "result": result,
                "formula": formula,
                "calculated_at": self.get_current_timestamp()
            }
        except Exception as e:
            return {
                "error": str(e),
                "operation": operation,
                "values": values,
                "period": period
            }

    def _mock_compare_companies(
        self,
        entities: List[str],
        attributes: List[str]
    ) -> Dict[str, Any]:
        """Mock company comparison"""
        company_data = {
            "IBM Watson Health": {
                "revenue": 1500000000,
                "market_share": 12.5,
                "fda_approvals": 5,
                "clinical_trials": 25,
                "pricing": "Custom",
                "technology": "Cognitive Computing",
                "founded": 2015
            },
            "Google Health": {
                "revenue": 2000000000,
                "market_share": 18.3,
                "fda_approvals": 8,
                "clinical_trials": 40,
                "pricing": "Subscription",
                "technology": "Deep Learning",
                "founded": 2018
            },
            "Microsoft Healthcare": {
                "revenue": 1200000000,
                "market_share": 10.2,
                "fda_approvals": 3,
                "clinical_trials": 15,
                "pricing": "Pay-per-use",
                "technology": "Cloud AI",
                "founded": 2017
            },
            "MedTech Innovations": {
                "revenue": 500000000,
                "market_share": 4.1,
                "fda_approvals": 2,
                "clinical_trials": 8,
                "pricing": "License",
                "technology": "Computer Vision",
                "founded": 2020
            }
        }

        matrix = {}
        for entity in entities:
            if entity in company_data:
                matrix[entity] = {}
                for attr in attributes:
                    if attr in company_data[entity]:
                        matrix[entity][attr] = company_data[entity][attr]
                    else:
                        matrix[entity][attr] = "NOT_ESTABLISHED"
            else:
                matrix[entity] = {attr: "NOT_ESTABLISHED" for attr in attributes}

        comparison = {
            "entities": entities,
            "attributes": attributes,
            "matrix": matrix,
            "generated_at": self.get_current_timestamp(),
            "notes": "Mock data for demonstration purposes"
        }

        return comparison

    # =========================================================================
    # CORE AGENT FUNCTIONS
    # =========================================================================

    def analyze_request(self, request: str) -> Dict[str, Any]:
        """Analyze a research request and generate a normalized objective"""
        if not request:
            raise ValueError("Request cannot be empty")

        request = sanitize_input(request)

        cache_key = f"analysis:{request}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        try:
            chain = REQUEST_ANALYSIS_PROMPT | self.llm
            response = chain.invoke({"request": request})

            if isinstance(response, AIMessage):
                response_text = response.content
            else:
                response_text = str(response)

            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {
                        "normalized_objective": f"Research: {request}",
                        "entity_type": "other",
                        "segment": "unspecified",
                        "geography": "unspecified",
                        "time_horizon": "current",
                        "deliverable_type": "report",
                        "ambiguities": [
                            {
                                "question": "Request is ambiguous",
                                "suggestion": "Please provide more specific details"
                            }
                        ],
                        "assumptions": [
                            {
                                "assumption": f"Research on: {request}",
                                "rationale": "No specific parameters provided"
                            }
                        ]
                    }

            schema = get_schema_by_name("research_plan")
            if schema:
                validation_errors = validate_json_schema(result, schema)
                if validation_errors:
                    result = self._fix_analysis_result(result, validation_errors)

            self.cache_manager.set(cache_key, result)
            return result

        except Exception as e:
            return {
                "normalized_objective": f"Research: {request}",
                "entity_type": "other",
                "segment": "unspecified",
                "geography": "unspecified",
                "time_horizon": "current",
                "deliverable_type": "report",
                "ambiguities": [
                    {
                        "question": f"Error analyzing request: {str(e)}",
                        "suggestion": "Please rephrase your request"
                    }
                ],
                "assumptions": []
            }

    def _fix_analysis_result(
        self,
        result: Dict[str, Any],
        errors: List[str]
    ) -> Dict[str, Any]:
        """Fix common validation errors in analysis result"""
        required_fields = ["normalized_objective", "entity_type", "segment", "geography", "time_horizon", "deliverable_type"]
        for field in required_fields:
            if field not in result:
                result[field] = "unspecified" if field != "normalized_objective" else f"Research: {result.get('request', '')}"

        if "ambiguities" not in result or not isinstance(result["ambiguities"], list):
            result["ambiguities"] = []

        if "assumptions" not in result or not isinstance(result["assumptions"], list):
            result["assumptions"] = []

        return result

    def generate_research_plan(
        self,
        normalized_objective: str,
        analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive research plan"""
        if not normalized_objective:
            raise ValueError("Normalized objective is required")

        cache_key = f"plan:{normalized_objective}:{json.dumps(analysis or {})}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        try:
            input_data = {
                "normalized_objective": normalized_objective,
                "analysis": json.dumps(analysis or {})
            }

            chain = RESEARCH_PLANNER_PROMPT | self.llm
            response = chain.invoke(input_data)

            if isinstance(response, AIMessage):
                response_text = response.content
            else:
                response_text = str(response)

            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = self._create_basic_plan(normalized_objective)

            result = self._validate_and_fix_plan(result)
            self.state["research_plan"] = result
            self.cache_manager.set(cache_key, result)

            return result

        except Exception:
            return self._create_basic_plan(normalized_objective)

    def _create_basic_plan(self, objective: str) -> Dict[str, Any]:
        """Create a basic research plan as fallback"""
        return {
            "research_objective": objective,
            "objectives": [
                {
                    "id": "obj_1",
                    "title": "Market Overview",
                    "description": f"Understand the current state of {objective}",
                    "sub_questions": [
                        {
                            "id": "q_1_1",
                            "text": f"What is the current market size for {objective}?",
                            "evidence_type": "market",
                            "tools": ["search_information", "retrieve_document"],
                            "urgency": "MEDIUM",
                            "expected_outcome": "Market size and growth data"
                        },
                        {
                            "id": "q_1_2",
                            "text": f"Who are the key players in {objective}?",
                            "evidence_type": "market",
                            "tools": ["search_information", "compare_companies"],
                            "urgency": "MEDIUM",
                            "expected_outcome": "List of major companies"
                        }
                    ],
                    "success_criteria": "Comprehensive market understanding"
                },
                {
                    "id": "obj_2",
                    "title": "Technology Analysis",
                    "description": f"Analyze the technology behind {objective}",
                    "sub_questions": [
                        {
                            "id": "q_2_1",
                            "text": f"What technologies are used in {objective}?",
                            "evidence_type": "technical",
                            "tools": ["search_information", "retrieve_document"],
                            "urgency": "HIGH",
                            "expected_outcome": "Technology overview"
                        }
                    ],
                    "success_criteria": "Technical understanding achieved"
                }
            ],
            "validation": {
                "min_objectives": 2,
                "max_objectives": 2,
                "min_questions_per_objective": 1,
                "max_questions_per_objective": 2,
                "total_questions": 3
            }
        }

    def _validate_and_fix_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix common issues in research plan"""
        if "research_objective" not in plan:
            plan["research_objective"] = "Unspecified"

        if "objectives" not in plan or not isinstance(plan["objectives"], list):
            plan["objectives"] = []

        for i, obj in enumerate(plan["objectives"]):
            if "id" not in obj:
                obj["id"] = f"obj_{i+1}"
            if "title" not in obj:
                obj["title"] = f"Objective {i+1}"
            if "description" not in obj:
                obj["description"] = ""
            if "sub_questions" not in obj or not isinstance(obj["sub_questions"], list):
                obj["sub_questions"] = []

            for j, q in enumerate(obj["sub_questions"]):
                if "id" not in q:
                    q["id"] = f"q_{i+1}_{j+1}"
                if "text" not in q:
                    q["text"] = f"Question {j+1}"
                if "evidence_type" not in q:
                    q["evidence_type"] = "other"
                if "tools" not in q or not isinstance(q["tools"], list):
                    q["tools"] = ["search_information"]
                if "urgency" not in q:
                    q["urgency"] = "MEDIUM"
                if "expected_outcome" not in q:
                    q["expected_outcome"] = ""

        if "validation" not in plan:
            plan["validation"] = {
                "min_objectives": len(plan["objectives"]),
                "max_objectives": len(plan["objectives"]),
                "min_questions_per_objective": 1,
                "max_questions_per_objective": max(len(obj["sub_questions"]) for obj in plan["objectives"]) if plan["objectives"] else 0,
                "total_questions": sum(len(obj["sub_questions"]) for obj in plan["objectives"])
            }

        return plan

    def validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a research plan"""
        issues = []

        if len(plan.get("objectives", [])) < 3:
            issues.append("Plan must have at least 3 objectives")

        if len(plan.get("objectives", [])) > 6:
            issues.append("Plan cannot have more than 6 objectives")

        for i, obj in enumerate(plan.get("objectives", []), 1):
            sub_questions = obj.get("sub_questions", [])
            if len(sub_questions) < 2:
                issues.append(f"Objective {i} must have at least 2 sub-questions")

            if len(sub_questions) > 5:
                issues.append(f"Objective {i} cannot have more than 5 sub-questions")

            question_texts = [q.get("text", "") for q in sub_questions]
            if len(question_texts) != len(set(question_texts)):
                issues.append(f"Objective {i} has duplicate sub-questions")

            for j, q in enumerate(sub_questions, 1):
                tools = q.get("tools", [])
                if not tools:
                    issues.append(f"Sub-question {i}.{j} has no tools assigned")

                valid_tools = [t.name for t in self.tools]
                for tool in tools:
                    if tool not in valid_tools:
                        issues.append(f"Sub-question {i}.{j} references invalid tool: {tool}")

        for i, obj in enumerate(plan.get("objectives", []), 1):
            for j, q in enumerate(obj.get("sub_questions", []), 1):
                question_text = q.get("text", "").lower()
                if any(word in question_text for word in ["everything", "all aspects", "comprehensive overview"]):
                    issues.append(f"Sub-question {i}.{j} is too broad and may not be answerable")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "plan": plan
        }

    def execute_research(
        self,
        research_plan: Dict[str, Any],
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the research loop to gather evidence"""
        if not research_plan:
            raise ValueError("Research plan is required")

        if run_id:
            self.reset_state(run_id)
        else:
            run_id = self.reset_state()

        self.state["research_plan"] = research_plan
        self.state["current_stage"] = "research_execution"
        self.state["evidence_store"] = []
        self.state["tool_call_history"] = []
        self.state["iteration_count"] = 0

        result = self._execute_research_loop(research_plan)
        report = self._generate_report_content()
        self._save_run_artifacts(run_id, research_plan, report)

        return {
            "run_id": run_id,
            "research_plan": research_plan,
            "evidence_store": self.state["evidence_store"],
            "tool_call_history": self.state["tool_call_history"],
            "report": report,
            "metrics": self.callback_handler.get_metrics()
        }

    def _execute_research_loop(
        self,
        research_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the bounded research loop"""
        all_questions = []
        for obj in research_plan.get("objectives", []):
            for q in obj.get("sub_questions", []):
                all_questions.append({
                    **q,
                    "objective_id": obj.get("id"),
                    "objective_title": obj.get("title")
                })

        question_status = {q["id"]: "open" for q in all_questions}

        max_iterations = getattr(getattr(self.settings, "budget", None), "max_iterations", 10)
        max_tool_calls = getattr(getattr(self.settings, "budget", None), "max_tool_calls", 25)

        while self._should_continue_research(question_status):
            self.state["iteration_count"] += 1

            if self.state["iteration_count"] > max_iterations:
                print(f"⚠️ Max iterations ({max_iterations}) reached")
                break

            if len(self.state["tool_call_history"]) >= max_tool_calls:
                print(f"⚠️ Max tool calls ({max_tool_calls}) reached")
                break

            next_question = self._select_next_question(all_questions, question_status)
            if not next_question:
                print("✅ All questions answered")
                break

            result = self._research_question(next_question)

            if result.get("status") == "answered":
                question_status[next_question["id"]] = "answered"
            elif result.get("status") == "unanswerable":
                question_status[next_question["id"]] = "unanswerable"

            self.state["evidence_store"].extend(result.get("evidence", []))
            self.state["tool_call_history"].extend(result.get("tool_calls", []))

        open_questions = [q for q in all_questions if question_status[q["id"]] == "open"]
        if open_questions:
            print(f"⚠️ {len(open_questions)} questions remain unanswered")

        return {
            "status": "completed" if not open_questions else "partial",
            "question_status": question_status,
            "answered_count": sum(1 for s in question_status.values() if s == "answered"),
            "unanswerable_count": sum(1 for s in question_status.values() if s == "unanswerable"),
            "open_count": sum(1 for s in question_status.values() if s == "open")
        }

    def _should_continue_research(self, question_status: Dict[str, str]) -> bool:
        """Determine if research should continue"""
        if "open" in question_status.values():
            return True

        max_iterations = getattr(getattr(self.settings, "budget", None), "max_iterations", 10)
        max_tool_calls = getattr(getattr(self.settings, "budget", None), "max_tool_calls", 25)

        if (self.state["iteration_count"] < max_iterations and
            len(self.state["tool_call_history"]) < max_tool_calls):
            return True

        return False

    def _select_next_question(
        self,
        all_questions: List[Dict[str, Any]],
        question_status: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Select the next question to research"""
        open_questions = [q for q in all_questions if question_status[q["id"]] == "open"]

        if not open_questions:
            return None

        urgency_order = {"EMERGENCY": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        open_questions.sort(key=lambda q: urgency_order.get(q.get("urgency", "MEDIUM"), 2))

        return open_questions[0]

    def _research_question(
        self,
        question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Research a specific question with proper dynamic tool binding"""
        question_id = question.get("id")
        question_text = question.get("text")
        tools = question.get("tools", [])

        print(f"🔍 Researching question {question_id}: {question_text}")

        context = self._prepare_question_context(question)
        available_tools = [t for t in self.tools if t.name in tools]

        try:
            # ٹولز کو رن ٹائم پر بائنڈ کریں اور پیور میسج ٹائپس بھیجیں
            bound_llm = self.llm.bind_tools(available_tools) if available_tools else self.llm
            response = bound_llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=context)
            ])

            evidence = []
            tool_calls = []

            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})

                    tool = self._get_tool_by_name(tool_name)
                    if tool:
                        try:
                            tool_result = tool.func(**tool_args)
                            tool_calls.append({
                                "timestamp": self.get_current_timestamp(),
                                "tool": tool_name,
                                "input": tool_args,
                                "output": tool_result,
                                "question_id": question_id
                            })
                            evidence_record = self._create_evidence_from_tool_result(
                                question_id, tool_name, tool_result
                            )
                            if evidence_record:
                                evidence.append(evidence_record)

                        except Exception as e:
                            tool_calls.append({
                                "timestamp": self.get_current_timestamp(),
                                "tool": tool_name,
                                "input": tool_args,
                                "output": {"error": str(e)},
                                "question_id": question_id
                            })
            else:
                response_text = response.content if isinstance(response, AIMessage) else str(response)

                evidence_record = {
                    "evidence_id": f"evidence_{len(self.state['evidence_store']) + 1}_{str(self.state['run_id'])[:8]}",
                    "question_id": question_id,
                    "claim": response_text,
                    "claim_type": "model_generated",
                    "source_ref": f"llm_response_{question_id}",
                    "source_kind": "model_generated",
                    "source_detail": {
                        "title": f"LLM Response for {question_id}",
                        "identifier": f"llm_response_{question_id}",
                        "date": self.get_current_timestamp()
                    },
                    "credibility": "medium",
                    "recency": "current",
                    "relevance": "high",
                    "corroboration": [],
                    "confidence": "medium",
                    "analyst_notes": "Generated by LLM without tool calls",
                    "urgency": question.get("urgency", "MEDIUM"),
                    "timestamp": self.get_current_timestamp()
                }
                evidence.append(evidence_record)

            return {
                "question_id": question_id,
                "status": "answered" if evidence else "unanswerable",
                "evidence": evidence,
                "tool_calls": tool_calls
            }

        except Exception as e:
            return {
                "question_id": question_id,
                "status": "error",
                "error": str(e),
                "evidence": [],
                "tool_calls": []
            }

    def _prepare_question_context(self, question: Dict[str, Any]) -> str:
        """Prepare context for researching a question"""
        context = f"""
        Research the following question as part of a healthcare market intelligence report:

        **Question ID:** {question.get('id')}
        **Question:** {question.get('text')}
        **Objective:** {question.get('objective_title', 'N/A')}
        **Evidence Type:** {question.get('evidence_type', 'other')}
        **Tools Available:** {', '.join(question.get('tools', []))}

        **Instructions:**
        1. Use the available tools to gather evidence
        2. Save all findings using the save_research tool
        3. Classify each finding by type (fact, inference, recommendation, uncertainty)
        4. Assign appropriate confidence levels
        5. Cite all sources properly

        **Previous Evidence:**
        {self._get_relevant_evidence(question.get('id'))}

        **Current Findings:**
        {self._get_current_findings_summary()}

        **Medical Disclaimer:**
        ⚠️ MEDICAL DISCLAIMER: This information is for research purposes only. It does not constitute medical advice. Always consult with qualified healthcare professionals.

        Now, research this question and provide evidence-based findings.
        """
        return context

    def _get_relevant_evidence(self, question_id: str) -> str:
        """Get evidence relevant to a specific question"""
        relevant_evidence = [
            e for e in self.state["evidence_store"]
            if e.get("question_id") == question_id
        ]

        if not relevant_evidence:
            return "No previous evidence for this question."

        evidence_text = "\n".join([
            f"- {e.get('claim', 'N/A')} (Type: {e.get('claim_type', 'N/A')}, Confidence: {e.get('confidence', 'N/A')})"
            for e in relevant_evidence
        ])

        return evidence_text

    def _get_current_findings_summary(self) -> str:
        """Get summary of current findings with syntax bug resolved"""
        if not self.state["evidence_store"]:
            return "No findings yet."

        fact_count = len([e for e in self.state["evidence_store"] if e.get("claim_type") == "fact"])
        inference_count = len([e for e in self.state["evidence_store"] if e.get("claim_type") == "inference"])
        recommendation_count = len([e for e in self.state["evidence_store"] if e.get("claim_type") == "recommendation"])
        uncertainty_count = len([e for e in self.state["evidence_store"] if e.get("claim_type") == "uncertainty"])

        return f"Current findings: {fact_count} facts, {inference_count} inferences, {recommendation_count} recommendations, {uncertainty_count} uncertainties."

    def _create_evidence_from_tool_result(
        self,
        question_id: str,
        tool_name: str,
        tool_result: Any
    ) -> Optional[Dict[str, Any]]:
        """Create evidence record from tool result"""
        try:
            if tool_name == "search_information":
                if isinstance(tool_result, list) and tool_result:
                    first_result = tool_result[0]
                    return {
                        "evidence_id": f"evidence_{len(self.state['evidence_store']) + 1}_{str(self.state['run_id'])[:8]}",
                        "question_id": question_id,
                        "claim": f"Search results found for query related to {question_id}",
                        "claim_type": "fact",
                        "source_ref": first_result.get("id", "unknown"),
                        "source_kind": "search_result",
                        "source_detail": {
                            "title": first_result.get("title", "Unknown"),
                            "identifier": first_result.get("id", "unknown"),
                            "date": first_result.get("date", "unknown"),
                            "source": first_result.get("source", "unknown")
                        },
                        "credibility": "medium",
                        "recency": "current",
                        "relevance": "high",
                        "corroboration": [],
                        "confidence": "medium",
                        "analyst_notes": f"Search result from {tool_name}",
                        "urgency": "MEDIUM",
                        "timestamp": self.get_current_timestamp()
                    }

            elif tool_name == "retrieve_document":
                if isinstance(tool_result, dict):
                    return {
                        "evidence_id": f"evidence_{len(self.state['evidence_store']) + 1}_{str(self.state['run_id'])[:8]}",
                        "question_id": question_id,
                        "claim": tool_result.get("title", "Document retrieved"),
                        "claim_type": "fact",
                        "source_ref": tool_result.get("document_id", "unknown"),
                        "source_kind": "retrieved_document",
                        "source_detail": {
                            "title": tool_result.get("title", "Unknown"),
                            "identifier": tool_result.get("document_id", "unknown"),
                            "date": tool_result.get("date", "unknown"),
                            "source": tool_result.get("source", "unknown"),
                            "author": tool_result.get("authors", "unknown")
                        },
                        "credibility": "high",
                        "recency": tool_result.get("date", "unknown"),
                        "relevance": "high",
                        "corroboration": [],
                        "confidence": "high",
                        "analyst_notes": f"Document retrieved via {tool_name}",
                        "urgency": "MEDIUM",
                        "timestamp": self.get_current_timestamp()
                    }

            elif tool_name == "calculate_metric":
                if isinstance(tool_result, dict):
                    return {
                        "evidence_id": f"evidence_{len(self.state['evidence_store']) + 1}_{str(self.state['run_id'])[:8]}",
                        "question_id": question_id,
                        "claim": f"Calculation: {tool_result.get('operation', 'unknown')} = {tool_result.get('result', 'N/A')}",
                        "claim_type": "fact",
                        "source_ref": f"calculation_{tool_result.get('operation', 'unknown')}_{self.get_current_timestamp()}",
                        "source_kind": "calculation",
                        "source_detail": {
                            "title": f"Calculation: {tool_result.get('operation', 'unknown')}",
                            "identifier": f"calc_{len(self.state['tool_call_history'])}",
                            "date": self.get_current_timestamp(),
                            "formula": tool_result.get("formula", "N/A")
                        },
                        "credibility": "high",
                        "recency": "current",
                        "relevance": "high",
                        "corroboration": [],
                        "confidence": "high",
                        "analyst_notes": f"Calculation performed via {tool_name}",
                        "urgency": "MEDIUM",
                        "timestamp": self.get_current_timestamp()
                    }

            elif tool_name == "compare_companies":
                if isinstance(tool_result, dict):
                    return {
                        "evidence_id": f"evidence_{len(self.state['evidence_store']) + 1}_{str(self.state['run_id'])[:8]}",
                        "question_id": question_id,
                        "claim": f"Company comparison: {len(tool_result.get('entities', []))} entities compared across {len(tool_result.get('attributes', []))} attributes",
                        "claim_type": "fact",
                        "source_ref": f"comparison_{len(self.state['tool_call_history'])}",
                        "source_kind": "calculation",
                        "source_detail": {
                            "title": "Company Comparison",
                            "identifier": f"comp_{len(self.state['tool_call_history'])}",
                            "date": self.get_current_timestamp()
                        },
                        "credibility": "medium",
                        "recency": "current",
                        "relevance": "high",
                        "corroboration": [],
                        "confidence": "medium",
                        "analyst_notes": f"Comparison performed via {tool_name}",
                        "urgency": "MEDIUM",
                        "timestamp": self.get_current_timestamp()
                    }

        except Exception as e:
            print(f"Error creating evidence from tool result: {e}")

        return None

    def _generate_report_content(self, sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate the final report content"""
        try:
            input_data = {
                "run_id": self.state["run_id"],
                "research_plan": json.dumps(self.state["research_plan"]),
                "evidence_store": json.dumps(self.state["evidence_store"]),
                "tool_history": json.dumps(self.state["tool_call_history"])
            }

            chain = REPORT_GENERATOR_PROMPT | self.llm
            response = chain.invoke(input_data)

            if isinstance(response, AIMessage):
                response_text = response.content
            else:
                response_text = str(response)

            try:
                report = json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    report = json.loads(json_match.group())
                else:
                    report = self._create_basic_report()

            report = self._validate_and_fix_report(report)

            report["report_id"] = self.state["run_id"]
            report["generated_at"] = self.get_current_timestamp()

            model_name = getattr(self.settings, "model_name", "gpt-4o-mini")
            report["model_versions"] = {
                "request_analysis": model_name,
                "research_planner": model_name,
                "research_execution": model_name,
                "report_generation": model_name
            }

            metrics = self.callback_handler.get_metrics()
            report["run_cost"] = self._calculate_run_cost(metrics)

            return report

        except Exception:
            return self._create_basic_report()

    def _create_basic_report(self) -> Dict[str, Any]:
        """Create a basic report as fallback"""
        research_objective = (self.state["research_plan"] or {}).get("research_objective", "Unspecified")
        model_name = getattr(self.settings, "model_name", "gpt-4o-mini")

        return {
            "report_id": self.state["run_id"],
            "generated_at": self.get_current_timestamp(),
            "model_versions": {
                "all": model_name
            },
            "run_cost": 0.0,
            "research_objective": research_objective,
            "ambiguities": (self.state["research_plan"] or {}).get("ambiguities", []),
            "executive_summary": f"Research report for: {research_objective}. This is a basic report generated as a fallback.",
            "market_overview": {
                "summary": "Market overview not fully established due to limited data."
            },
            "key_trends": [
                {
                    "statement": f"Research conducted on {research_objective}",
                    "claim_type": "fact",
                    "evidence_ids": [],
                    "confidence": "low",
                    "urgency": "MEDIUM"
                }
            ],
            "competitor_analysis": {},
            "opportunities": [],
            "risks": [],
            "recommendations": [],
            "evidence_appendix": self.state["evidence_store"],
            "limitations_and_gaps": [
                "Report generation encountered errors. Some sections may be incomplete.",
                "Please review all findings carefully."
            ],
            "sources": list(set([
                e.get("source_detail", {}).get("source", "Unknown")
                for e in self.state["evidence_store"]
                if e.get("source_detail", {}).get("source")
            ])),
            "confidence_level": {
                "level": "low",
                "rationale": "Report generated as fallback due to errors"
            }
        }

    def _validate_and_fix_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix common issues in report"""
        required_fields = [
            "report_id", "generated_at", "research_objective",
            "executive_summary", "key_trends", "limitations_and_gaps",
            "evidence_appendix", "confidence_level"
        ]

        for field in required_fields:
            if field not in report:
                if field == "report_id":
                    report[field] = self.state["run_id"]
                elif field == "generated_at":
                    report[field] = self.get_current_timestamp()
                elif field == "research_objective":
                    report[field] = (self.state["research_plan"] or {}).get("research_objective", "Unspecified")
                elif field == "executive_summary":
                    report[field] = f"Research report for: {report.get('research_objective', 'Unspecified')}"
                elif field == "key_trends":
                    report[field] = []
                elif field == "limitations_and_gaps":
                    report[field] = ["Report validation identified missing sections"]
                elif field == "evidence_appendix":
                    report[field] = self.state["evidence_store"]
                elif field == "confidence_level":
                    report[field] = {
                        "level": "low",
                        "rationale": "Report validation identified issues"
                    }

        if "evidence_appendix" in report:
            existing_ids = {e.get("evidence_id") for e in report["evidence_appendix"]}
            for evidence in self.state["evidence_store"]:
                if evidence.get("evidence_id") not in existing_ids:
                    report["evidence_appendix"].append(evidence)

        if "research_plan" in self.state and self.state["research_plan"]:
            all_questions = []
            for obj in self.state["research_plan"].get("objectives", []):
                all_questions.extend(obj.get("sub_questions", []))

            answered_questions = {e.get("question_id") for e in report.get("evidence_appendix", [])}
            unanswered = [q.get("id") for q in all_questions if q.get("id") not in answered_questions]

            if unanswered and "limitations_and_gaps" in report:
                if not report["limitations_and_gaps"]:
                    report["limitations_and_gaps"] = []
                report["limitations_and_gaps"].append(
                    f"Questions not addressed: {', '.join(unanswered)}"
                )

        self._validate_report_invariants(report)
        return report

    def _validate_report_invariants(self, report: Dict[str, Any]):
        """Validate cross-field invariants in the report"""
        issues = []
        evidence_ids_in_appendix = {e.get("evidence_id") for e in report.get("evidence_appendix", [])}

        for section in ["key_trends", "opportunities", "risks"]:
            if section in report:
                for item in report[section]:
                    for evidence_id in item.get("evidence_ids", []):
                        if evidence_id not in evidence_ids_in_appendix:
                            issues.append(f"Evidence ID {evidence_id} in {section} does not resolve")

        for item in report.get("key_trends", []):
            if item.get("claim_type") == "fact" and not item.get("evidence_ids"):
                issues.append(f"Fact claim without evidence: {item.get('statement', 'N/A')}")

        if issues and "limitations_and_gaps" in report:
            if not report["limitations_and_gaps"]:
                report["limitations_and_gaps"] = []
            report["limitations_and_gaps"].extend(issues)

    def _calculate_run_cost(self, metrics: Dict[str, Any]) -> float:
        """Calculate the estimated cost of the run"""
        pricing = {
            "gpt-4o-mini": 0.000003,
            "gpt-4o": 0.000015,
            "gpt-3.5-turbo": 0.0000015
        }

        model_name = getattr(self.settings, "model_name", "gpt-4o-mini")
        token_count = metrics.get("token_count", 0)
        price_per_token = pricing.get(model_name, 0.000003)
        estimated_cost = token_count * price_per_token * 2

        return round(estimated_cost, 4)

    def run_quality_control(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Run quality control on the report"""
        try:
            input_data = {
                "report": json.dumps(report)
            }

            chain = QUALITY_CONTROL_PROMPT | self.llm
            response = chain.invoke(input_data)

            if isinstance(response, AIMessage):
                response_text = response.content
            else:
                response_text = str(response)

            try:
                qc_result = json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    qc_result = json.loads(json_match.group())
                else:
                    qc_result = self._create_basic_qc_result(report)

            qc_result = self._perform_code_based_qc_checks(report, qc_result)
            return qc_result

        except Exception:
            return self._create_basic_qc_result(report)

    def _create_basic_qc_result(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic QC result as fallback"""
        return {
            "report_id": report.get("report_id", "unknown"),
            "inspection_timestamp": self.get_current_timestamp(),
            "overall_status": "PASS_WITH_WARNINGS",
            "defects": [
                {
                    "defect_id": "fallback_1",
                    "defect_type": "coverage_gap",
                    "severity": "MEDIUM",
                    "location": {"section": "report", "field": "all"},
                    "description": "QC generation encountered errors. Manual review recommended.",
                    "gap_question": "Please review the report manually",
                    "status": "NEW"
                }
            ],
            "summary": {
                "total_defects": 1,
                "by_type": {"coverage_gap": 1},
                "by_severity": {"MEDIUM": 1},
                "critical_defects": 0,
                "blocking_defects": 0
            },
            "recommendation": "REPAIR",
            "repair_questions": ["Please review the report manually"]
        }

    def _perform_code_based_qc_checks(
        self,
        report: Dict[str, Any],
        qc_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform code-based quality control checks"""
        defects = qc_result.get("defects", [])
        defect_ids = {d.get("defect_id") for d in defects}

        for section in ["key_trends", "opportunities", "risks"]:
            if section in report:
                for i, item in enumerate(report[section]):
                    if item.get("claim_type") == "fact" and not item.get("evidence_ids"):
                        defect_id = f"unsupported_claim_{section}_{i}"
                        if defect_id not in defect_ids:
                            defects.append({
                                "defect_id": defect_id,
                                "defect_type": "unsupported_claim",
                                "severity": "CRITICAL",
                                "location": {"section": section, "index": i},
                                "description": f"Fact claim without evidence: {item.get('statement', 'N/A')[:100]}...",
                                "gap_question": f"Find evidence to support: {item.get('statement', 'N/A')}",
                                "status": "NEW"
                            })
                            defect_ids.add(defect_id)

        evidence_appendix = report.get("evidence_appendix", [])
        tool_call_ids = {
            tc.get("output", {}).get("id") or
            tc.get("output", {}).get("document_id") or
            tc.get("input", {}).get("document_id") or
            tc.get("output", {}).get("source_ref")
            for tc in self.state["tool_call_history"]
        }

        for i, evidence in enumerate(evidence_appendix):
            source_ref = evidence.get("source_ref")
            if source_ref and source_ref not in tool_call_ids and not source_ref.startswith("llm_response_") and not source_ref.startswith("calculation_") and not source_ref.startswith("comparison_"):
                defect_id = f"missing_source_{i}"
                if defect_id not in defect_ids:
                    defects.append({
                        "defect_id": defect_id,
                        "defect_type": "missing_source",
                        "severity": "HIGH",
                        "location": {"section": "evidence_appendix", "index": i},
                        "description": f"Source reference {source_ref} does not resolve to a real tool result",
                        "gap_question": f"Verify or replace source reference: {source_ref}",
                        "status": "NEW"
                    })
                    defect_ids.add(defect_id)

        if "research_plan" in self.state and self.state["research_plan"]:
            all_questions = []
            for obj in self.state["research_plan"].get("objectives", []):
                all_questions.extend(obj.get("sub_questions", []))

            evidence_question_ids = {e.get("question_id") for e in evidence_appendix}
            report_question_ids = set()

            for section in ["key_trends", "opportunities", "risks"]:
                if section in report:
                    for item in report[section]:
                        for evidence_id in item.get("evidence_ids", []):
                            for evidence in evidence_appendix:
                                if evidence.get("evidence_id") == evidence_id:
                                    report_question_ids.add(evidence.get("question_id"))
                                    break

            for q in all_questions:
                q_id = q.get("id")
                if q_id not in evidence_question_ids and q_id not in report_question_ids:
                    defect_id = f"coverage_gap_{q_id}"
                    if defect_id not in defect_ids:
                        defects.append({
                            "defect_id": defect_id,
                            "defect_type": "coverage_gap",
                            "severity": "HIGH",
                            "location": {"section": "research_plan", "question_id": q_id},
                            "description": f"Planned question not addressed: {q.get('text', 'N/A')[:100]}...",
                            "gap_question": f"Research: {q.get('text', 'N/A')}",
                            "status": "NEW"
                        })
                        defect_ids.add(defect_id)

        qc_result["defects"] = defects

        by_type = {}
        by_severity = {}
        critical_count = 0
        blocking_count = 0

        for defect in defects:
            defect_type = defect.get("defect_type")
            severity = defect.get("severity")

            by_type[defect_type] = by_type.get(defect_type, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1

            if severity in ["CRITICAL", "HIGH"]:
                blocking_count += 1
            if severity == "CRITICAL":
                critical_count += 1

        qc_result["summary"] = {
            "total_defects": len(defects),
            "by_type": by_type,
            "by_severity": by_severity,
            "critical_defects": critical_count,
            "blocking_defects": blocking_count
        }

        if critical_count > 0:
            qc_result["overall_status"] = "FAIL"
            qc_result["recommendation"] = "REJECT"
        elif blocking_count > 0:
            qc_result["overall_status"] = "FAIL"
            qc_result["recommendation"] = "REPAIR"
        elif len(defects) > 0:
            qc_result["overall_status"] = "PASS_WITH_WARNINGS"
            qc_result["recommendation"] = "REPAIR"
        else:
            qc_result["overall_status"] = "PASS"
            qc_result["recommendation"] = "APPROVE"

        repair_questions = [d.get("gap_question") for d in defects if d.get("gap_question")]
        qc_result["repair_questions"] = list(set(repair_questions))

        return qc_result

    def run_repair_loop(
        self,
        report: Dict[str, Any],
        qc_result: Dict[str, Any],
        max_rounds: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run the repair loop to fix QC defects"""
        if max_rounds is None:
            max_rounds = getattr(getattr(self.settings, "budget", None), "max_repair_rounds", 3)

        repair_round = 0
        current_report = report
        current_qc = qc_result

        while repair_round < max_rounds:
            repair_round += 1
            print(f"🔧 Running repair round {repair_round}/{max_rounds}")

            repair_questions = current_qc.get("repair_questions", [])

            if not repair_questions:
                print("✅ No repair questions remaining")
                break

            repair_evidence = []
            for question in repair_questions:
                temp_question = {
                    "id": f"repair_{repair_round}_{len(repair_evidence) + 1}",
                    "text": question,
                    "evidence_type": "other",
                    "tools": ["search_information", "retrieve_document"],
                    "urgency": "HIGH",
                    "expected_outcome": f"Answer to: {question}"
                }

                result = self._research_question(temp_question)
                repair_evidence.extend(result.get("evidence", []))

            existing_ids = {e.get("evidence_id") for e in current_report.get("evidence_appendix", [])}
            for evidence in repair_evidence:
                if evidence.get("evidence_id") not in existing_ids:
                    current_report["evidence_appendix"].append(evidence)
                    existing_ids.add(evidence.get("evidence_id"))

            current_report = self._generate_report_content()
            current_qc = self.run_quality_control(current_report)

            if current_qc.get("overall_status") == "PASS":
                print("✅ Report passed QC after repair")
                break

        if repair_round >= max_rounds and current_qc.get("overall_status") != "PASS":
            print(f"⚠️ Exhausted {max_rounds} repair rounds")
            for defect in current_qc.get("defects", []):
                if defect.get("status") == "NEW":
                    defect["status"] = "UNRESOLVED"

            unresolved = [d.get("description") for d in current_qc.get("defects", [])
                         if d.get("status") == "UNRESOLVED"]
            if unresolved and "limitations_and_gaps" in current_report:
                current_report["limitations_and_gaps"].extend(unresolved)

        return current_report, current_qc

    def save_report(self, report: Dict[str, Any], run_id: str):
        """Save the final report"""
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)

        report_path = run_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        if self.state["research_plan"]:
            plan_path = run_dir / "research_plan.json"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(self.state["research_plan"], f, indent=2)

        evidence_path = run_dir / "evidence_store.json"
        with open(evidence_path, "w", encoding="utf-8") as f:
            json.dump(self.state["evidence_store"], f, indent=2)

        tools_path = run_dir / "tool_call_history.json"
        with open(tools_path, "w", encoding="utf-8") as f:
            json.dump(self.state["tool_call_history"], f, indent=2)

        if "qc_result" in self.state:
            qc_path = run_dir / "qc_verdict.json"
            with open(qc_path, "w", encoding="utf-8") as f:
                json.dump(self.state["qc_result"], f, indent=2)

        usage_summary = {
            "run_id": run_id,
            "timestamp": self.get_current_timestamp(),
            "model": getattr(self.settings, "model_name", "gpt-4o-mini"),
            "iterations": self.state["iteration_count"],
            "tool_calls": len(self.state["tool_call_history"]),
            "evidence_collected": len(self.state["evidence_store"]),
            "report_status": report.get("approval", {}).get("decision", "NOT_APPROVED"),
            "metrics": self.callback_handler.get_metrics()
        }

        usage_path = run_dir / "usage_summary.json"
        with open(usage_path, "w", encoding="utf-8") as f:
            json.dump(usage_summary, f, indent=2)

        print(f"✅ Report saved to: {run_dir}")

    def _save_run_artifacts(
        self,
        run_id: str,
        research_plan: Dict[str, Any],
        report: Dict[str, Any]
    ):
        """Save run artifacts to disk"""
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)

        plan_path = run_dir / "research_plan.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(research_plan, f, indent=2)

        evidence_path = run_dir / "evidence_store.json"
        with open(evidence_path, "w", encoding="utf-8") as f:
            json.dump(self.state["evidence_store"], f, indent=2)

        tools_path = run_dir / "tool_call_history.json"
        with open(tools_path, "w", encoding="utf-8") as f:
            json.dump(self.state["tool_call_history"], f, indent=2)

        usage_summary = {
            "run_id": run_id,
            "timestamp": self.get_current_timestamp(),
            "model": getattr(self.settings, "model_name", "gpt-4o-mini"),
            "iterations": self.state["iteration_count"],
            "tool_calls": len(self.state["tool_call_history"]),
            "evidence_collected": len(self.state["evidence_store"]),
            "metrics": self.callback_handler.get_metrics()
        }

        usage_path = run_dir / "usage_summary.json"
        with open(usage_path, "w", encoding="utf-8") as f:
            json.dump(usage_summary, f, indent=2)

        print(f"📁 Run artifacts saved to: {run_dir}")

    # =========================================================================
    # STREAMLIT INTEGRATION
    # =========================================================================

    def stream_analysis(self, request: str):
        """Stream the request analysis process"""
        if not request:
            yield "Please provide a research request."
            return

        self.reset_state()
        chain = REQUEST_ANALYSIS_PROMPT | self.llm

        try:
            for chunk in chain.stream({"request": request}):
                if isinstance(chunk, AIMessage):
                    if chunk.content:
                        yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"

    def stream_research_execution(self, research_plan: Dict[str, Any]):
        """Stream the research execution process"""
        if not research_plan:
            yield "Please provide a research plan."
            return

        self.reset_state()
        self.state["research_plan"] = research_plan

        yield f"🚀 Starting research execution for: {research_plan.get('research_objective', 'Unspecified')}\n\n"

        all_questions = []
        for obj in research_plan.get("objectives", []):
            for q in obj.get("sub_questions", []):
                all_questions.append({
                    **q,
                    "objective_id": obj.get("id"),
                    "objective_title": obj.get("title")
                })

        max_iterations = getattr(getattr(self.settings, "budget", None), "max_iterations", 10)
        max_tool_calls = getattr(getattr(self.settings, "budget", None), "max_tool_calls", 25)

        for i, question in enumerate(all_questions, 1):
            yield f"🔍 ({i}/{len(all_questions)}) Researching: {question.get('text', 'N/A')}\n"

            result = self._research_question(question)

            if result.get("status") == "answered":
                yield f"✅ Answered with {len(result.get('evidence', []))} evidence items\n"
            elif result.get("status") == "unanswerable":
                yield f"⚠️ Could not answer: {question.get('text', 'N/A')}\n"
            else:
                yield f"❌ Error researching: {question.get('text', 'N/A')}\n"

            self.state["evidence_store"].extend(result.get("evidence", []))
            self.state["tool_call_history"].extend(result.get("tool_calls", []))

            if (self.state["iteration_count"] >= max_iterations or
                len(self.state["tool_call_history"]) >= max_tool_calls):
                yield f"⚠️ Budget limits reached. Stopping research.\n"
                break

            yield "\n"

        yield "📝 Generating report...\n"
        report = self._generate_report_content()
        self.state["report"] = report

        yield f"✅ Research completed! Collected {len(self.state['evidence_store'])} evidence items.\n"
        yield f"Report ID: {report.get('report_id', 'N/A')}\n"

    def get_full_workflow(
        self,
        request: str,
        max_iterations: Optional[int] = None,
        max_tool_calls: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute the complete workflow from request to report"""
        budget_obj = getattr(self.settings, "budget", None)
        original_max_iterations = getattr(budget_obj, "max_iterations", 10) if budget_obj else 10
        original_max_tool_calls = getattr(budget_obj, "max_tool_calls", 25) if budget_obj else 25

        if budget_obj:
            if max_iterations is not None:
                self.settings.budget.max_iterations = max_iterations
            if max_tool_calls is not None:
                self.settings.budget.max_tool_calls = max_tool_calls

        try:
            analysis_result = self.analyze_request(request)
            research_plan = self.generate_research_plan(
                analysis_result["normalized_objective"],
                analysis_result
            )
            research_result = self.execute_research(research_plan)
            qc_result = self.run_quality_control(research_result["report"])

            final_report = research_result["report"]
            if qc_result.get("recommendation") == "REPAIR":
                final_report, qc_result = self.run_repair_loop(
                    research_result["report"],
                    qc_result
                )

            self.save_report(final_report, research_result["run_id"])

            return {
                "analysis": analysis_result,
                "research_plan": research_plan,
                "research_result": research_result,
                "report": final_report,
                "qc_result": qc_result,
                "run_id": research_result["run_id"]
            }

        finally:
            if budget_obj:
                self.settings.budget.max_iterations = original_max_iterations
                self.settings.budget.max_tool_calls = original_max_tool_calls
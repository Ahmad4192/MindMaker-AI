import os
import sys
import streamlit as st
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Settings, CacheType
from src.chains import MarketMindAgent
from src.cache_manager import CacheManager
from src.prompts import SYSTEM_PROMPT, APPROVAL_GATE_PROMPT
from src.utils import (
    display_safety_notice,
    display_medical_disclaimer,
    create_urgency_badge,
    format_evidence_card,
    format_structured_output,
    validate_json_schema
)

# Initialize settings
settings = Settings()

# Initialize cache manager
cache_manager = CacheManager(
    cache_type=settings.cache_type,
    memory_cache_size=settings.memory_cache_size,
    sqlite_cache_path=settings.sqlite_cache_path
)

# Initialize agent
agent = MarketMindAgent(
    settings=settings,
    cache_manager=cache_manager
)

def main():
    """Main Streamlit application for Healthcare AI Research Agent"""

    # Page configuration
    st.set_page_config(
        page_title="Healthcare AI Research Agent",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for dark theme and styling
    st.markdown("""
    <style>
        /* Dark Theme */
        :root {
            --primary-color: #2E86AB;
            --secondary-color: #A23B72;
            --accent-color: #F18F01;
            --background-color: #0E1117;
            --card-bg: #1E2127;
            --text-color: #FAFAFA;
            --border-color: #3D3D3D;
            --success-color: #4CAF50;
            --warning-color: #FF9800;
            --danger-color: #F44336;
            --info-color: #2196F3;
        }

        /* Main container styling */
        .main > div {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Card containers */
        .stCard {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }

        .stCard:hover {
            border-color: var(--primary-color);
            box-shadow: 0 6px 12px rgba(46, 134, 171, 0.2);
        }

        /* Glowing focus states */
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(46, 134, 171, 0.3);
        }

        /* Gradient action buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 32px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* Secondary buttons */
        .stButton > button:has(.stButton > button:contains("Secondary")) {
            background: linear-gradient(135deg, var(--accent-color), #F5A623);
        }

        /* Danger buttons */
        .stButton > button:has(.stButton > button:contains("Reject")) {
            background: linear-gradient(135deg, var(--danger-color), #D32F2F);
        }

        /* Success buttons */
        .stButton > button:has(.stButton > button:contains("Approve")) {
            background: linear-gradient(135deg, var(--success-color), #388E3C);
        }

        /* Urgency badges */
        .urgency-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 8px;
        }

        .urgency-low {
            background-color: var(--success-color);
            color: white;
        }

        .urgency-medium {
            background-color: var(--warning-color);
            color: white;
        }

        .urgency-high {
            background-color: var(--danger-color);
            color: white;
        }

        .urgency-emergency {
            background-color: #9C27B0;
            color: white;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        /* Evidence cards */
        .evidence-card {
            background-color: var(--card-bg);
            border-left: 4px solid var(--primary-color);
            padding: 16px;
            margin: 8px 0;
            border-radius: 8px;
        }

        /* Structured output styling */
        .structured-output {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 40px;
            padding: 0 24px;
            background-color: var(--card-bg);
            border-radius: 8px 8px 0 0;
            border: 1px solid var(--border-color);
            border-bottom: none;
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--primary-color);
            color: white;
            border-bottom: 1px solid var(--primary-color);
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--background-color);
        }

        /* Metric cards */
        .stMetric {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
        }

        /* Disclaimer styling */
        .disclaimer-box {
            background-color: #FFEB3B;
            color: #1E1E1E;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid var(--warning-color);
            font-size: 14px;
            line-height: 1.6;
        }

        /* Code blocks */
        pre {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            overflow-x: auto;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--background-color);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-color);
        }
    </style>
    """, unsafe_allow_html=True)

    # Display safety notices and disclaimers
    display_safety_notice()
    display_medical_disclaimer()

    # Sidebar
    with st.sidebar:
        st.title("⚕️ Healthcare AI Research")
        st.markdown("---")

        # Cache type selector
        cache_option = st.selectbox(
            "Cache Mechanism",
            options=[CacheType.IN_MEMORY.value, CacheType.SQLITE.value],
            index=0 if settings.cache_type == CacheType.IN_MEMORY else 1,
            help="Toggle between In-Memory and SQLite caching"
        )

        if cache_option != settings.cache_type.value:
            settings.cache_type = CacheType(cache_option)
            cache_manager.switch_cache_type(settings.cache_type)
            st.rerun()

        st.markdown("---")

        # API Key input
        if not settings.openai_api_key:
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key"
            )
            if openai_key:
                settings.openai_api_key = openai_key
                st.rerun()
        else:
            st.success("✅ API Key Configured")
            if st.button("Clear API Key"):
                settings.openai_api_key = None
                st.rerun()

        st.markdown("---")

        # Model selection
        model_option = st.selectbox(
            "LLM Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Select the OpenAI model to use"
        )

        if model_option != settings.model_name:
            settings.model_name = model_option
            st.rerun()

        st.markdown("---")

        # Temperature control
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=settings.temperature,
            step=0.1,
            help="Controls randomness in model responses"
        )

        if temperature != settings.temperature:
            settings.temperature = temperature
            st.rerun()

        st.markdown("---")

        # Budget controls
        st.subheader("💰 Budget Controls")
        max_tokens = st.number_input(
            "Max Tokens per Run",
            min_value=1000,
            max_value=100000,
            value=settings.max_tokens,
            step=1000
        )

        max_iterations = st.number_input(
            "Max Iterations",
            min_value=1,
            max_value=50,
            value=settings.max_iterations,
            step=1
        )

        if max_tokens != settings.max_tokens or max_iterations != settings.max_iterations:
            settings.max_tokens = max_tokens
            settings.max_iterations = max_iterations
            st.rerun()

        st.markdown("---")

        # Clear cache button
        if st.button("🗑️ Clear Cache"):
            cache_manager.clear_all_caches()
            st.success("All caches cleared!")

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Healthcare AI Research Agent**
        Version: 1.0.0
        Built with: LangChain, OpenAI, Streamlit

        This agent helps healthcare professionals
        research medical topics, treatments, and
        industry trends with full traceability.
        """)

    # Main content
    st.title("🏥 Healthcare AI Research Agent")
    st.markdown("""
    A production-ready LLM/RAG application for healthcare market intelligence.
    Generate comprehensive research reports with full traceability and validation.
    """)

    # Initialize session state
    if "run_id" not in st.session_state:
        st.session_state.run_id = None
    if "research_plan" not in st.session_state:
        st.session_state.research_plan = None
    if "evidence_store" not in st.session_state:
        st.session_state.evidence_store = []
    if "report" not in st.session_state:
        st.session_state.report = None
    if "approval_status" not in st.session_state:
        st.session_state.approval_status = None
    if "streaming_output" not in st.session_state:
        st.session_state.streaming_output = ""

    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Research Request",
        "🔍 Research Plan",
        "📊 Evidence & Analysis",
        "📄 Final Report",
        "✅ Approval Gate"
    ])

    with tab1:
        st.header("Research Request")

        if st.session_state.research_plan or st.session_state.report:
            st.warning("⚠️ Starting a new request will reset the current session.")
            if st.button("Start New Request", type="primary"):
                st.session_state.run_id = None
                st.session_state.research_plan = None
                st.session_state.evidence_store = []
                st.session_state.report = None
                st.session_state.approval_status = None
                st.session_state.streaming_output = ""
                st.rerun()

        research_request = st.text_area(
            "Enter your healthcare research request:",
            placeholder="Example: Research AI applications in clinical diagnostics for 2024-2025, focusing on FDA-approved solutions and market trends.",
            height=150
        )

        if st.button("Analyze Request & Generate Plan", type="primary", use_container_width=True):
          if not research_request.strip():
           st.warning("Please enter a research request first!")
        else:
            with st.spinner("Analyzing request and generating research plan..."):
                try:
                    # 1. تجزیہ حاصل کریں
                    analysis = agent.analyze_request(research_request)
                    st.session_state.analysis = analysis
                    # ambiguities اور assumptions کو بھی سیشن اسٹیٹ میں ڈال دیں
                    st.session_state.ambiguities = analysis.get("ambiguities", [])
                    st.session_state.assumptions = analysis.get("assumptions", [])
                
                   # 2. پلان حاصل کریں
                    obj = analysis.get("normalized_objective", research_request)
                    raw_plan = agent.generate_research_plan(obj, analysis)
                
                   # 3. محفوظ طریقے سے سیشن اسٹیٹ میں محفوظ کریں (KeyError کا خاتمہ)
                    if isinstance(raw_plan, dict) and "plan" in raw_plan:
                       st.session_state.research_plan = raw_plan["plan"]
                    else:
                       st.session_state.research_plan = raw_plan
                
                    # رن آئی ڈی بھی ساتھ سیٹ کر دیں
                    st.session_state.run_id = agent.run_id or agent.reset_state()
                
                    st.success("✅ Research plan generated successfully!")
                    time.sleep(0.5)
                    st.rerun()  # اگلے ٹیب پر خود لے جائے گا
                
                except Exception as e:
                 st.error(f"Error generating plan: {str(e)}")
    with tab2:
        st.header("Research Plan")

        if not st.session_state.research_plan:
            st.info("Please enter a research request in the first tab to generate a plan.")
        else:
            st.subheader(f"Run ID: {st.session_state.run_id}")

            # Display ambiguities if any
            if st.session_state.get("ambiguities", []):
                with st.expander("❓ Detected Ambiguities", expanded=True):
                    for i, ambiguity in enumerate(st.session_state.ambiguities, 1):
                        st.markdown(f"**{i}. {ambiguity['question']}**")
                        st.caption(f"Suggested clarification: {ambiguity.get('suggestion', 'N/A')}")

            # Display plan
            st.subheader("📋 Research Plan")

            for i, objective in enumerate(st.session_state.research_plan["objectives"], 1):
                with st.expander(f"🎯 Objective {i}: {objective['title']}", expanded=True):
                    st.markdown(f"**Description:** {objective['description']}")

                    if objective["sub_questions"]:
                        st.markdown("**Sub-Questions:**")
                        for j, question in enumerate(objective["sub_questions"], 1):
                            urgency_badge = create_urgency_badge(question.get("urgency", "MEDIUM"))
                            st.markdown(f"{j}. {question['text']} {urgency_badge}")

                            if question.get("tools"):
                                st.caption(f"Tools: {', '.join(question['tools'])}")

            # Plan validation
            st.markdown("---")
            st.subheader("✅ Plan Validation")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Objectives", len(st.session_state.research_plan["objectives"]))
            with col2:
                total_questions = sum(len(obj["sub_questions"]) for obj in st.session_state.research_plan["objectives"])
                st.metric("Total Questions", total_questions)
            with col3:
                validation_status = agent.validate_plan(st.session_state.research_plan)
                st.metric("Validation", "✅ Valid" if validation_status["valid"] else "❌ Invalid")

            if not validation_status["valid"]:
                with st.expander("Validation Issues"):
                    for issue in validation_status["issues"]:
                        st.error(f"- {issue}")

# Start research button
            # اگر validation_status ڈکشنری ہو یا بولین، دونوں صورتوں کو محفوظ طریقے سے ہینڈل کریں
            is_valid = validation_status.get("valid", False) if isinstance(validation_status, dict) else bool(validation_status)

            if is_valid:
                st.markdown("---")
                if st.button("🚀 Start Research Execution", type="primary", use_container_width=True):
                    with st.status("Executing Autonomous Research Loop...", expanded=True) as status_box:
                         try:
                            # 1. اسٹیٹ محفوظ طریقے سے ری سیٹ کریں
                            st.session_state.evidence_store = []
                            current_run_id = getattr(st.session_state, "run_id", None)

                            status_box.write("🔍 Running agent loop and dispatching tools...")

                            # 2. ریسرچ ایگزیکیوشن کال
                            research_result = agent.execute_research(
                                st.session_state.research_plan,
                                current_run_id
                            )

                            # 3. ڈیٹا کو محفوظ انداز میں نکال کر محفوظ کریں
                            st.session_state.evidence_store = research_result.get("evidence_store", [])
                            st.session_state.report = research_result.get("report", {})
                            if "run_id" in research_result:
                                st.session_state.run_id = research_result["run_id"]

                            status_box.update(label="✅ Research Execution Completed!", state="complete", expanded=False)
                            st.success("✅ Research execution completed successfully!")
                            time.sleep(0.8)
                            st.rerun()
                         except Exception as e:
                             status_box.update(label="❌ Execution Failed", state="error", expanded=True)
                             st.error(f"Error during execution: {str(e)}")
                # یہ بلاک بٹن کے بعد add کریں
                if getattr(st.session_state, "research_plan", None):
                    plan_data = st.session_state.research_plan
    
                    st.markdown("---")
                    st.subheader("📋 Generated Research Plan")
    
                    # 1. ریسرچ آبجیکٹو دکھائیں
                    st.markdown(f"**Research Objective:** {plan_data.get('research_objective', 'N/A')}")
    
                    # 2. پلان کے مقاصد اور ذیلی سوالات کے کارڈز
                    objectives = plan_data.get("objectives", [])
                    if objectives:
                        for idx, obj in enumerate(objectives, 1):
                            with st.expander(f"🎯 Objective {idx}: {obj.get('title', 'Objective')}", expanded=True):
                                st.write(obj.get("description", ""))
                                st.markdown("**Sub-questions:**")
                                for q in obj.get("sub_questions", []):
                                    urgency = q.get("urgency", "MEDIUM")
                                    st.markdown(f"- ❓ **{q.get('text', '')}** `[Urgency: {urgency}]`")
                else:
                   st.json(plan_data)
        
                st.markdown("---")
    
                # 3. باقاعدہ ریسرچ ایگزیکیوشن شروع کرنے کا بٹن
                if st.button("🚀 Start Research Execution", type="primary", use_container_width=True):
                   with st.status("Executing Autonomous Research Loop...", expanded=True) as status_box:
                       try:
                          st.session_state.evidence_store = []
                          status_box.write("🔍 Collecting evidence from sources and tools...")
                
                          research_result = agent.execute_research(
                              st.session_state.research_plan,
                              getattr(st.session_state, "run_id", None)
                          )
                
                          st.session_state.evidence_store = research_result.get("evidence_store", [])
                          st.session_state.report = research_result.get("report", {})
                
                          status_box.update(label="✅ Research Execution Completed!", state="complete")
                          st.success(f"✅ Success! Collected {len(st.session_state.evidence_store)} evidence records.")
                          st.rerun()
                
                       except Exception as e:
                           status_box.update(label="❌ Execution Failed", state="error")
                           st.error(f"Error during research: {str(e)}")
                       except Exception as e:
                            status_box.update(label="❌ Execution Failed", state="error")
                            st.error(f"❌ Error during research execution: {str(e)}")
            else:
                st.warning("⚠️ Please resolve plan validation issues before starting research.")

    with tab3:
        st.header("Evidence & Analysis")

        # محفوظ جانچ کہ آیا ایویڈینس موجود ہے
        evidence_list = getattr(st.session_state, "evidence_store", [])

        if not evidence_list:
            st.info("No evidence collected yet. Please run research execution in the Plan tab.")
        else:
            st.subheader(f"Evidence Store ({len(evidence_list)} items)")

            # ایویڈینس ریکارڈز کو خوبصورت کارڈز میں دکھائیں
            for idx, item in enumerate(evidence_list, 1):
                claim_type = item.get("claim_type", "fact").upper()
                confidence = item.get("confidence", "medium").upper()
                claim_text = item.get("claim", "No description provided.")
                source_ref = item.get("source_ref", "Unknown source")

                # ٹائپ کے مطابق رنگ کا انتخاب
                badge_color = "#10B981" if claim_type == "FACT" else "#F59E0B" if claim_type == "INFERENCE" else "#6366F1"

                with st.expander(f"📌 Evidence #{idx}: [{claim_type}] - {claim_text[:80]}..."):
                    st.markdown(f"**Claim:** {claim_text}")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(f"**Type:** <span style='color:{badge_color}; font-weight:bold;'>{claim_type}</span>", unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"**Confidence:** `{confidence}`")
                    with col_c:
                        st.markdown(f"**Source Ref:** `{source_ref}`")

                    # اضافی معلومات اگر موجود ہوں
                    if item.get("source_detail"):
                        st.json(item["source_detail"])

            # Evidence filters
            col1, col2, col3 = st.columns(3)
            with col1:
                claim_type_filter = st.selectbox(
                    "Filter by Claim Type",
                    options=["All", "fact", "inference", "recommendation", "uncertainty"],
                    index=0
                )
            with col2:
                confidence_filter = st.selectbox(
                    "Filter by Confidence",
                    options=["All", "high", "medium", "low"],
                    index=0
                )
            with col3:
                search_term = st.text_input("Search Evidence", placeholder="Enter search term")

            # Display evidence
            for i, evidence in enumerate(st.session_state.evidence_store, 1):
                # Apply filters
                if claim_type_filter != "All" and evidence.get("claim_type") != claim_type_filter:
                    continue
                if confidence_filter != "All" and evidence.get("confidence") != confidence_filter:
                    continue
                if search_term and search_term.lower() not in evidence.get("claim", "").lower():
                    continue

                # Create evidence card
                urgency_badge = create_urgency_badge(evidence.get("urgency", "MEDIUM"))
                evidence_card_html = f"""
                <div class="evidence-card">
                    <h4>Evidence #{i} {urgency_badge}</h4>
                    <p><strong>Claim:</strong> {evidence.get('claim', 'N/A')}</p>
                    <p><strong>Type:</strong> <span style="color: {'#4CAF50' if evidence.get('claim_type') == 'fact' else '#FF9800' if evidence.get('claim_type') == 'inference' else '#F44336' if evidence.get('claim_type') == 'uncertainty' else '#2196F3'}">
                        {evidence.get('claim_type', 'N/A')}
                    </span></p>
                    <p><strong>Confidence:</strong> {evidence.get('confidence', 'N/A')}</p>
                    <p><strong>Source:</strong> {evidence.get('source_detail', {}).get('title', 'N/A')}</p>
                    <p><strong>Source ID:</strong> {evidence.get('source_ref', 'N/A')}</p>
                </div>
                """
                st.markdown(evidence_card_html, unsafe_allow_html=True)

            # Analysis section
            st.markdown("---")
            st.subheader("📊 Analysis Summary")

            if st.session_state.report:
                report_data = st.session_state.report

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    fact_count = len([e for e in st.session_state.evidence_store if e.get("claim_type") == "fact"])
                    st.metric("Facts", fact_count, help="Directly sourced claims")
                with col2:
                    inference_count = len([e for e in st.session_state.evidence_store if e.get("claim_type") == "inference"])
                    st.metric("Inferences", inference_count, help="Derived claims")
                with col3:
                    high_confidence = len([e for e in st.session_state.evidence_store if e.get("confidence") == "high"])
                    st.metric("High Confidence", high_confidence)
                with col4:
                    low_confidence = len([e for e in st.session_state.evidence_store if e.get("confidence") == "low"])
                    st.metric("Low Confidence", low_confidence)

                # Display key findings
                if "key_trends" in report_data:
                    with st.expander("🔍 Key Trends & Findings", expanded=True):
                        for trend in report_data["key_trends"]:
                            urgency_badge = create_urgency_badge(trend.get("urgency", "MEDIUM"))
                            st.markdown(f"**{trend.get('statement', 'N/A')}** {urgency_badge}")
                            st.caption(f"Type: {trend.get('claim_type', 'N/A')} | Confidence: {trend.get('confidence', 'N/A')}")

                # Display competitor analysis if available
                if "competitor_analysis" in report_data:
                    with st.expander("🏢 Competitor Analysis", expanded=False):
                        st.dataframe(report_data["competitor_analysis"])
            else:
                st.info("Report data not available yet.")

    with tab4:
        st.header("Final Report")

        if not st.session_state.report:
            st.info("No report generated yet. Please complete research execution.")
        else:
            report = st.session_state.report

            # Display report metadata
            st.subheader("📋 Report Metadata")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Report ID", report.get("report_id", "N/A"))
            with col2:
                st.metric("Generated", report.get("generated_at", "N/A"))
            with col3:
                st.metric("Run Cost", f"${report.get('run_cost', 0):.2f}")

            st.markdown("---")

            # Executive Summary
            with st.expander("📝 Executive Summary", expanded=True):
                st.markdown(report.get("executive_summary", "No executive summary available."))

            st.markdown("---")

            # Market Overview
            with st.expander("🌍 Market Overview", expanded=False):
                if "market_overview" in report:
                    st.markdown(report["market_overview"])
                else:
                    st.info("Market overview not available.")

            st.markdown("---")

            # Key Trends
            with st.expander("📈 Key Trends", expanded=False):
                if "key_trends" in report:
                    for trend in report["key_trends"]:
                        urgency_badge = create_urgency_badge(trend.get("urgency", "MEDIUM"))
                        st.markdown(f"**{trend.get('statement', 'N/A')}** {urgency_badge}")
                        st.caption(f"Claim Type: {trend.get('claim_type', 'N/A')} | Confidence: {trend.get('confidence', 'N/A')}")
                else:
                    st.info("No key trends available.")

            st.markdown("---")

            # Opportunities
            with st.expander("🎯 Opportunities", expanded=False):
                if "opportunities" in report:
                    for opp in report["opportunities"]:
                        urgency_badge = create_urgency_badge(opp.get("urgency", "MEDIUM"))
                        st.markdown(f"**{opp.get('statement', 'N/A')}** {urgency_badge}")
                else:
                    st.info("No opportunities identified.")

            st.markdown("---")

            # Risks
            with st.expander("⚠️ Risks", expanded=False):
                if "risks" in report:
                    for risk in report["risks"]:
                        urgency_badge = create_urgency_badge(risk.get("urgency", "HIGH"))
                        st.markdown(f"**{risk.get('statement', 'N/A')}** {urgency_badge}")
                else:
                    st.info("No risks identified.")

            st.markdown("---")

            # Limitations and Gaps
            with st.expander("⚠️ Limitations & Gaps", expanded=True):
                if "limitations_and_gaps" in report and report["limitations_and_gaps"]:
                    for gap in report["limitations_and_gaps"]:
                        st.warning(f"- {gap}")
                else:
                    st.success("✅ No identified limitations or gaps")

            st.markdown("---")

            # Evidence Appendix
            with st.expander("📚 Evidence Appendix", expanded=False):
                if "evidence_appendix" in report:
                    for evidence in report["evidence_appendix"]:
                        with st.container():
                            st.markdown(f"**Evidence ID:** {evidence.get('evidence_id', 'N/A')}")
                            st.markdown(f"**Claim:** {evidence.get('claim', 'N/A')}")
                            st.markdown(f"**Type:** {evidence.get('claim_type', 'N/A')}")
                            st.markdown(f"**Source:** {evidence.get('source_detail', {}).get('title', 'N/A')}")
                            st.markdown("---")
                else:
                    st.info("No evidence appendix available.")

            # JSON Export
            st.markdown("---")
            if st.button("💾 Export Report as JSON"):
                import json
                json_str = json.dumps(report, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"healthcare_report_{report.get('report_id', 'unknown')}.json",
                    mime="application/json"
                )

    with tab5:
        st.header("Approval Gate")

        if not st.session_state.report:
            st.info("No report available for approval. Please generate a report first.")
        elif st.session_state.approval_status:
            st.success(f"✅ Report already {st.session_state.approval_status}")
            st.markdown("---")
            st.json(st.session_state.report.get("approval", {}))
        else:
            st.warning("⚠️ This report requires explicit approval before publication.")

            # Display approval information
            st.subheader("📋 Approval Summary")

            report = st.session_state.report

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Findings", len(report.get("key_trends", [])))
                st.metric("Facts", len([e for e in st.session_state.evidence_store if e.get("claim_type") == "fact"]))
            with col2:
                st.metric("Inferences", len([e for e in st.session_state.evidence_store if e.get("claim_type") == "inference"]))
                st.metric("Uncertainties", len([e for e in st.session_state.evidence_store if e.get("claim_type") == "uncertainty"]))

            st.markdown("---")

            # Confidence level
            confidence = report.get("confidence_level", {})
            st.subheader("🎯 Confidence Assessment")
            st.progress(confidence.get("value", 0.5))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence Level", confidence.get("level", "N/A"))
            with col2:
                st.metric("Basis", confidence.get("rationale", "N/A"))

            st.markdown("---")

            # Limitations
            limitations = report.get("limitations_and_gaps", [])
            if limitations:
                st.subheader("⚠️ Outstanding Limitations")
                for limitation in limitations:
                    st.warning(f"- {limitation}")
            else:
                st.success("✅ No outstanding limitations")

            st.markdown("---")

            # Approval options
            st.subheader("📝 Approval Decision")

            with st.form("approval_form"):
                approver_name = st.text_input("Your Name", placeholder="Enter your name")
                approver_role = st.text_input("Your Role", placeholder="Enter your role")
                decision_notes = st.text_area("Notes", placeholder="Any additional comments or notes")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    approve = st.form_submit_button("✅ Approve", type="primary", use_container_width=True)
                with col2:
                    reject = st.form_submit_button("❌ Reject", type="secondary", use_container_width=True)
                with col3:
                    more_research = st.form_submit_button("🔍 More Research", use_container_width=True)
                with col4:
                    modify_scope = st.form_submit_button("📝 Modify Scope", use_container_width=True)

                if approve:
                    if not approver_name:
                        st.error("Please enter your name")
                    else:
                        approval_data = {
                            "approver_identity": {
                                "name": approver_name,
                                "role": approver_role
                            },
                            "decision": "APPROVED",
                            "timestamp": agent.get_current_timestamp(),
                            "notes": decision_notes
                        }

                        report["approval"] = approval_data
                        st.session_state.report = report
                        st.session_state.approval_status = "APPROVED"

                        # Save final report
                        agent.save_report(report, st.session_state.run_id)

                        st.success("✅ Report approved and finalized!")
                        st.rerun()

                elif reject:
                    if not approver_name:
                        st.error("Please enter your name")
                    else:
                        approval_data = {
                            "approver_identity": {
                                "name": approver_name,
                                "role": approver_role
                            },
                            "decision": "REJECTED",
                            "timestamp": agent.get_current_timestamp(),
                            "notes": decision_notes
                        }

                        report["approval"] = approval_data
                        st.session_state.report = report
                        st.session_state.approval_status = "REJECTED"

                        st.success("✅ Report rejected and archived!")
                        st.rerun()

                elif more_research:
                    st.info("🔄 Redirecting to research request for additional questions...")
                    st.session_state.approval_status = "NEEDS_MORE_RESEARCH"
                    st.rerun()

                elif modify_scope:
                    st.info("📝 Redirecting to modify research scope...")
                    st.session_state.approval_status = "SCOPE_MODIFICATION_REQUESTED"
                    st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>🏥 Healthcare AI Research Agent | Built with ❤️ using LangChain & OpenAI</p>
        <p>© 2024 | All rights reserved | For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
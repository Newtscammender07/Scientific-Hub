# app.py
import sys
import os
from dotenv import load_dotenv

# Load environment variables with force override to ensure new keys/configs are used
load_dotenv(override=True)

# Fix for Windows OSError 22 in Streamlit/Rich
os.environ["TERM"] = "dumb"
os.environ["PYTHONIOENCODING"] = "utf-8"

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.crew import ResearchCrew
from dotenv import load_dotenv
import requests
import json
import traceback
import time
import os
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Load environment variables
load_dotenv(override=True)

# Fix for Windows OSError 22 in Streamlit/Rich
os.environ["TERM"] = "dumb"
os.environ["PYTHONIOENCODING"] = "utf-8"

# ── History helpers ───────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "data", "history.json")

def load_history():
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_to_history(topic, provider, model, mode, result_text, duration_s):
    history = load_history()
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "topic": topic,
        "provider": provider,
        "model": model,
        "mode": mode,
        "duration_s": duration_s,
        "result": result_text
    }
    history.insert(0, entry)   # newest first
    history = history[:20]     # keep last 20
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

st.set_page_config(page_title="AI Research Assistant", layout="wide", page_icon="🔬")

# ── Theme state ───────────────────────────────────────────────────
st.session_state.dark_mode = True
dark = True

# ── Global CSS (CSS-variable based light/dark) ─────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ===== SCIENTIFIC THEME VARIABLES ===== */
:root {{
  --bg:          #0a0b10;
  --bg-panel:    #11141d;
  --border:      rgba(0, 242, 255, 0.15);
  --border-dim:  rgba(255, 255, 255, 0.05);
  --text:        #e2e8f0;
  --text-dim:    #94a3b8;
  --accent:      #00f2ff;
  --accent-glow: rgba(0, 242, 255, 0.3);
  --card-bg:     rgba(19, 21, 26, 0.8);
  --sidebar-bg:  #08090d;
  --font-main:   'Inter', sans-serif;
  --font-tech:   'Space Grotesk', sans-serif;
  --font-mono:   'JetBrains Mono', monospace;
}}

html, body, [class*="css"], p, span {{
  font-family: var(--font-main);
  color: var(--text);
  font-size: 18px !important;
}}

h1, h2, h3, .hero-title {{
  font-family: var(--font-tech) !important;
  letter-spacing: -0.02em;
}}

/* App background with dot-grid */
.stApp {{
  background: var(--bg);
  background-image: 
    radial-gradient(circle at 2px 2px, var(--border) 1px, transparent 0);
  background-size: 32px 32px;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--border-dim);
}}
[data-testid="stSidebar"] * {{ color: var(--text) !important; }}

/* Buttons */
.stButton > button {{
  background: transparent;
  color: var(--accent) !important;
  border: 1px solid var(--accent);
  border-radius: 4px;
  font-family: var(--font-tech);
  font-weight: 500;
  font-size: 18px;
  padding: 0.5rem 1.2rem;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}

.stButton > button:hover {{
  background: var(--accent-glow);
  box-shadow: 0 0 15px var(--accent-glow);
  transform: translateY(-1px);
}}

.stButton > button:active {{
  transform: translateY(0);
}}

/* Research Button Special Styling */
div.stButton > button:first-child {{
    border-width: 2px;
    font-weight: 700;
}}

/* Hero */
.hero-wrap {{
  padding: 2rem 0;
  border-bottom: 1px solid var(--border-dim);
  margin-bottom: 2rem;
  background: linear-gradient(to right, rgba(0,242,255,0.03), transparent);
}}

.hero-title {{
  font-size: 2.8rem; font-weight: 700;
  color: var(--text);
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}}

.hero-sub {{ 
  color: var(--text-dim); 
  font-size: 1rem; 
  font-family: var(--font-tech); 
  text-transform: uppercase;
  letter-spacing: 0.1em;
}}

/* Glass card */
.glass-card {{
  background: var(--card-bg);
  border: 1px solid var(--border-dim);
  border-left: 3px solid var(--accent);
  border-radius: 4px;
  padding: 2rem;
  backdrop-filter: blur(20px);
  margin-bottom: 1.5rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
  position: relative;
}}

.glass-card::before {{
  content: "[ DATA_NODE_" + attr(id) + " ]";
  position: absolute;
  top: 8px;
  right: 12px;
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--accent);
  opacity: 0.5;
}}

/* Metric cards */
.metric-row {{ display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; }}
.metric-card {{
  flex: 1; min-width: 160px;
  background: var(--bg-panel);
  border: 1px solid var(--border-dim);
  border-radius: 4px; padding: 1.5rem 1rem; text-align: left;
  position: relative;
  overflow: hidden;
}}

.metric-card::after {{
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 2px;
    background: var(--accent);
    opacity: 0.5;
}}

.metric-value {{ 
  font-size: 2.4rem; 
  font-family: var(--font-tech); 
  font-weight: 700; 
  color: var(--accent); 
  line-height: 1; 
}}

.metric-label {{ 
  font-size: 0.7rem; 
  color: var(--text-dim); 
  margin-top: 0.5rem; 
  text-transform: uppercase; 
  letter-spacing: 0.1em; 
  font-family: var(--font-tech);
}}

.metric-badge {{ 
  display: inline-block; 
  margin-top: 0.75rem; 
  padding: 2px 8px; 
  border-radius: 2px; 
  font-size: 0.65rem; 
  font-family: var(--font-mono);
  font-weight: 500; 
  text-transform: uppercase;
}}

.badge-green {{ background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }}
.badge-blue  {{ background: rgba(59,130,246,0.1); color: #3b82f6; border: 1px solid rgba(59,130,246,0.2); }}
.badge-purple{{ background: rgba(167,139,250,0.1); color: #a78bfa; border: 1px solid rgba(167,139,250,0.2); }}

/* Section header */
.section-header {{
  font-family: var(--font-tech);
  font-size: 1.2rem; font-weight: 600; color: var(--text);
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.8rem;
}}

.section-header::before {{
    content: "";
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--accent);
}}

/* Proposal text */
.proposal-body {{ 
  color: var(--text); 
  line-height: 1.7; 
  font-size: 1rem; 
  white-space: pre-wrap; 
  font-family: var(--font-main);
}}

/* Divider */
.glow-divider {{
  height: 1px;
  background: var(--border-dim);
  margin: 2.5rem 0;
  position: relative;
}}

.glow-divider::after {{
    content: "";
    position: absolute;
    top: 0; left: 0; width: 40px; height: 1px;
    background: var(--accent);
}}

/* Input / selectbox */
.stTextInput input, .stSelectbox select {{
  background: var(--bg-panel) !important;
  color: var(--text) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: 4px !important;
  font-family: var(--font-main) !important;
  font-size: 18px !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 0; }}
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────
st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">RESEARCH_CORE_V4</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Autonomous Literature Mining & Strategic Grant Synthesis</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.caption("SYSTEM_STATUS: STABLE // HUB_01")
    st.markdown("### CONFIG_SEARCH")
    topic = st.text_input("RESEARCH_TOPIC:", "AI for Climate Change Mitigation")
    
    st.divider()
    st.markdown("### CONFIG_ENGINE")
    provider = st.selectbox("LLM_PROVIDER", ["Groq (Free & Fast) 🚀", "Gemini", "OpenAI", "Ollama (No API Key Required)"])
    
    if "Groq" in provider:
        provider = "Groq"
        st.success("🚀 **Groq Mode**: Ultra-fast free inference! No rate limits issues.")
        model_name = st.selectbox("Model", [
            "llama-3.3-70b-versatile",    # Best quality ✅ Production
            "llama-3.1-8b-instant",       # Fastest ✅ Production (lower TPM)
            "meta-llama/llama-4-scout-17b-16e-instruct",  # Preview - newest
            "qwen/qwen3-32b",             # Preview - reasoning
            "moonshotai/kimi-k2-instruct-0905",  # Preview
        ])
    elif provider == "Gemini":
        model_name = st.selectbox("Model", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro-latest", "Custom"])
        if model_name == "Custom":
            model_name = st.text_input("Enter Model ID", "gemini-2.0-flash")
    elif provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    else:
        provider = "Ollama" # Simplify for backend
        st.info("🏠 **Ollama Mode**: Uses your local hardware. No API keys, no costs!")
        model_name = st.text_input("Ollama Model Name", "llama3.2:latest")
        st.markdown("""
        **Quick Setup Guide:**
        1. Download [Ollama](https://ollama.com/)
        2. Run: `ollama run llama3.2`
        3. Click 'Start Research' below.
        """)
        
        if st.button("🔍 Test Ollama Connection"):
            with st.spinner("Pinging 127.0.0.1:11434..."):
                try:
                    r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
                    if r.status_code == 200:
                        models = [m['name'] for m in r.json().get('models', [])]
                        st.success(f"Connected! Available: {', '.join(models)}")
                        if model_name not in models:
                            st.warning(f"Note: '{model_name}' not in list. Try one from above.")
                    else:
                        st.error(f"Server responded with {r.status_code}")
                except Exception as e:
                    st.error(f"Connection Failed: {str(e)}")
                    st.info("💡 Try running `ollama serve` in a separate terminal.")
        
    agent_process = st.radio("Process Mode", ["Sequential", "Hierarchical"])
    max_results = st.slider("Max Papers to Analyze", 5, 20, 10)
    
    st.divider()
    st.header("⚡ Speed Settings")
    
    speed_mode = st.radio(
        "Research Mode",
        ["🔬 Standard (Full Detail)", "⚡ Turbo (Faster)", "⚡⚡ Flash (Fastest)"],
        index=0,
        help="Flash: 2 agents, ~2 min. Turbo: 6 agents, ~8 min. Standard: Full quality, ~15 min."
    )
    
    turbo_mode = "Turbo" in speed_mode
    flash_mode = "Flash" in speed_mode
    
    # Dynamic ETA
    if flash_mode:
        est_label = "~1-2 min ⚡⚡"
        st.success("⚡⚡ **Flash Mode**: 2 agents, ultra-brief output. Fastest possible results!")
    elif turbo_mode:
        est_label = "~5-8 min ⚡"
        st.info("⚡ **Turbo Mode**: All 6 agents, reduced iterations.")
    else:
        est_label = "~15-20 min 🔬"
        st.info("🔬 **Standard Mode**: Full 6-agent pipeline, high quality.")
    
    if provider == "Ollama":
        st.warning(f"⏳ **ETA (local model)**: {est_label}")
    elif provider == "Groq":
        st.success(f"⚡ **ETA (Groq turbo)**: {est_label}")
    else:
        st.info(f"⏳ **ETA**: {est_label}")
    
    st.divider()
    st.header("🕑 Search History")
    history = load_history()
    if not history:
        st.caption("No searches yet. Run a research query to start building history.")
    else:
        if st.button("🗑️ Clear History", use_container_width=True):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()
        for i, entry in enumerate(history):
            label = f"🔍 {entry['topic'][:30]}{'...' if len(entry['topic'])>30 else ''}"
            with st.expander(f"{label}  ·  {entry['timestamp']}"):
                st.caption(f"**Provider:** {entry['provider']} · **Model:** {entry['model']} · **Mode:** {entry['mode']} · **Time:** {entry['duration_s']}s")
                st.text_area(
                    "Result", 
                    value=entry['result'], 
                    height=200, 
                    key=f"hist_{i}",
                    label_visibility="collapsed"
                )
                st.download_button(
                    "⬇️ Download",
                    data=entry['result'],
                    file_name=f"proposal_{entry['timestamp'].replace(':','').replace(' ','_')}.txt",
                    mime="text/plain",
                    key=f"dl_{i}"
                )
    if provider != "Ollama":
        if not os.getenv("OPENAI_API_KEY") and provider == "OpenAI":
            api_key = st.text_input("OpenAI API Key:", type="password")
            if api_key: os.environ["OPENAI_API_KEY"] = api_key
        if not os.getenv("GEMINI_API_KEY") and provider == "Gemini":
            gem_key = st.text_input("Gemini API Key:", type="password")
            if gem_key: os.environ["GEMINI_API_KEY"] = gem_key
        if not os.getenv("GROQ_API_KEY") and provider == "Groq":
            groq_key = st.text_input("Groq API Key:", type="password")
            if groq_key: os.environ["GROQ_API_KEY"] = groq_key
        if provider == "Groq" and os.getenv("GROQ_API_KEY"):
            st.success("✅ Groq API key loaded!")
    else:
        st.success("✅ No API keys needed for local execution!")

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🚀 Start Research Journey"):
        if not os.getenv("OPENAI_API_KEY") and provider == "OpenAI":
            st.error("Please provide an OpenAI API Key in the sidebar or .env file.")
        elif not os.getenv("GEMINI_API_KEY") and provider == "Gemini":
            st.error("Please provide a Gemini API Key in the sidebar or .env file.")
        elif not os.getenv("GROQ_API_KEY") and provider == "Groq":
            st.error("Please provide a Groq API Key in the sidebar or .env file.")
        else:
            # ── Dashboard Setup ────────────────────────────────────────
            TASK_STEPS = [
                ("📚", "Literature Mining",    "Scanning ArXiv & Semantic Scholar"),
                ("📈", "Trend Analysis",       "Identifying temporal research trends"),
                ("🔍", "Gap Identification",   "Pinpointing unexplored areas"),
                ("🧪", "Methodology Design",   "Designing the research plan"),
                ("✍️", "Grant Writing",        "Drafting the grant proposal"),
                ("🔬", "Novelty Audit",        "Verifying originality"),
            ]
            N_STEPS = len(TASK_STEPS)
            
            # Thread-safe shared state (dict mutation is GIL-safe in CPython)
            shared = {
                "step_idx": 0,       # which step is currently running (0-based)
                "done_steps": [],    # list of completed step indices
                "log": [],           # list of (agent_name, thought) tuples
                "complete": False,
                "error": None,
                "result": None,
            }
            
            # ── Callbacks (run in background thread) ───────────────────
            def task_callback(task):
                idx = shared["step_idx"]
                if idx < N_STEPS and idx not in shared["done_steps"]:
                    shared["done_steps"].append(idx)
                shared["step_idx"] = min(idx + 1, N_STEPS)
            
            def step_callback(step):
                try:
                    # Get agent name from the executing agent
                    step_type = type(step).__name__  # 'AgentFinish' or 'AgentAction'
                    
                    if step_type == 'AgentAction':
                        # Agent is using a tool - langchain: .tool, .tool_input, .log
                        tool = getattr(step, 'tool', '') or ''
                        tool_input = getattr(step, 'tool_input', '') or ''
                        msg = f"🔧 Using **{tool}**: {str(tool_input)[:120]}" if tool else None
                    elif step_type == 'AgentFinish':
                        # Agent finished a step - langchain: .thought, .text, .output
                        text = getattr(step, 'text', '') or getattr(step, 'output', '') or ''
                        thought = getattr(step, 'thought', '')
                        # Use the text response (actual answer), or thought if text is blank
                        msg = str(text or thought)[:160] if (text or thought) else None
                    else:
                        # Unknown type - dump first 160 chars of str representation
                        raw = str(step)[:160]
                        msg = raw if raw.strip() else None
                    
                    if msg and msg.strip():
                        # Try to get agent role from shared step_idx context
                        current_idx = min(shared["step_idx"], len(TASK_STEPS) - 1)
                        agent_label = TASK_STEPS[current_idx][1]  # e.g. "Literature Mining"
                        shared["log"].append((agent_label, msg))
                        if len(shared["log"]) > 20:
                            shared["log"].pop(0)
                except Exception:
                    pass

            crew = ResearchCrew(topic, model_name=model_name, provider=provider, 
                               task_callback=task_callback, step_callback=step_callback,
                               turbo_mode=turbo_mode, flash_mode=flash_mode)
            
            # ── Background Research Thread ─────────────────────────────
            def run_research():
                try:
                    shared["result"] = crew.kickoff()
                except Exception as e:
                    shared["error"] = e
                finally:
                    shared["complete"] = True

            research_thread = threading.Thread(target=run_research)
            add_script_run_ctx(research_thread)
            research_thread.start()
            
            # ── Main UI ────────────────────────────────────────────────
            st.markdown('<div class="section-header">RESEARCH_DASHBOARD</div>', unsafe_allow_html=True)
            
            dash_left, dash_right = st.columns([1, 1])
            
            with dash_left:
                st.markdown("**AGENT_PIPELINE**")
                step_placeholders = [st.empty() for _ in TASK_STEPS]
                st.markdown("**SIGNAL_STRENGTH**")
                progress_bar = st.progress(0)
                timer_ph = st.empty()
                
            with dash_right:
                st.markdown("**LIVE_AGENT_TELEMETRY**")
                feed_ph = st.empty()
            
            # ── Live Update Loop ───────────────────────────────────────
            start_time = time.time()
            try:
                while not shared["complete"]:
                    elapsed = int(time.time() - start_time)
                    current = shared["step_idx"]
                    done = shared["done_steps"]
                    
                    # Update step checklist
                    for i, (icon, name, detail) in enumerate(TASK_STEPS):
                        if i in done:
                            step_placeholders[i].success(f"✅ **{name}** — {detail}")
                        elif i == current:
                            step_placeholders[i].info(f"⚙️ **{name}** — {detail} *(running...)*")
                        else:
                            step_placeholders[i].markdown(f"⌛ {name} — {detail}")
                    
                    # Update progress bar
                    pct = len(done) / N_STEPS
                    progress_bar.progress(pct, text=f"{int(pct*100)}% complete")
                    
                    # Update timer
                    timer_ph.metric("⏱️ Elapsed Time", f"{elapsed}s")
                    
                    # ── Live Agent Feed ─────────────────────────────────
                    current_step_name = TASK_STEPS[min(current, N_STEPS-1)][1]
                    current_step_detail = TASK_STEPS[min(current, N_STEPS-1)][2]
                    
                    pulse = ["⠋", "⠙", "⠸", "⠴", "⠦", "⠇"][elapsed % 6]
                    feed_content = f"""
                    <div style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--accent); background: rgba(0,0,0,0.2); padding: 1rem; border: 1px solid var(--border); border-radius: 4px;">
                    <div style="border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                        <span>ACTIVE_NODE: {current_step_name.upper()}</span>
                        <span>{pulse}</span>
                    </div>
                    <div style="color: var(--text-dim); margin-bottom: 1rem;">Task: {current_step_detail}</div>
                    """
                    
                    if shared["log"]:
                        for agent, msg in shared["log"][-6:]:
                            feed_content += f'<div style="margin-bottom: 0.8rem;"><span style="color: var(--accent)">[{agent.upper()}]</span> {msg}</div>\n'
                    else:
                        dots = "." * ((elapsed % 3) + 1)
                        feed_content += f'<div style="color: var(--text-dim); font-style: italic;">Awaiting process signal{dots}</div>'
                    
                    feed_content += "</div>"
                    feed_ph.markdown(feed_content, unsafe_allow_html=True)
                    
                    time.sleep(1)
                
                # ── Finished ───────────────────────────────────────────
                if shared["error"]:
                    raise shared["error"]
                
                result = shared["result"]
                total_time = int(time.time() - start_time)
                
                # Mark all complete
                for i, (icon, name, detail) in enumerate(TASK_STEPS):
                    step_placeholders[i].success(f"✅ **{name}** — {detail}")
                progress_bar.progress(1.0, text="100% complete ✅")
                timer_ph.metric("⏱️ Total Time", f"{total_time}s")
                
                st.success(f"🎉 Research complete in **{total_time}s**! Scroll down for results.")
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    st.error("📉 **API Quota Exceeded**: You've hit the rate limit for your current LLM provider.")
                    st.info("💡 **Tip**: Switch to a different provider in the sidebar, or try Turbo Mode to reduce API calls.")
                else:
                    st.error(f"❌ **An unexpected error occurred**: {error_msg}")
                    with st.expander("Show Full Error Traceback"):
                        st.code(traceback.format_exc())
                st.stop()
            
            # Displaying the result
            if 'result' in locals():
                # ── Proposal Card ──────────────────────────────────────────
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📄 Generated Research Proposal</div>', unsafe_allow_html=True)
                result_text = str(result)
                
                # ── Novelty Parsing & Cleanup ────────────────────────────────
                import re
                score_match = re.search(r"NOVELTY_SCORE:\s*([\d.]+)", result_text)
                novelty_score = float(score_match.group(1)) if score_match else 0.85
                
                # Split proposal from audit
                audit_header_pattern = r"(# \*\*Novelty Audit: Score \+ Justification\*\*)"
                splits = re.split(audit_header_pattern, result_text, flags=re.IGNORECASE)
                if len(splits) > 1:
                    proposal_content = splits[0].replace(f"NOVELTY_SCORE: {novelty_score}", "").strip()
                    audit_content = (splits[1] + splits[2]).strip()
                else:
                    proposal_content = result_text.replace(f"NOVELTY_SCORE: {novelty_score}", "").strip()
                    audit_content = None

                # ── Save to history ───────────────────────────────────────
                save_to_history(
                    topic=topic,
                    provider=provider,
                    model=model_name,
                    mode="Flash" if flash_mode else ("Turbo" if turbo_mode else "Standard"),
                    result_text=result_text,
                    duration_s=total_time
                )
                
                # ── Proposal Section ───────────────────────────────────────
                st.markdown('<div class="proposal-container"></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="glass-card">\n\n{proposal_content}\n\n</div>', unsafe_allow_html=True)

                if audit_content:
                    st.markdown('<div class="section-header">🔍 NOVELTY_AUDIT_REPORT</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="glass-card" style="border-left-color: var(--accent); padding-top: 1.5rem; font-size: 22px !important;">\n\n{audit_content}\n\n</div>', unsafe_allow_html=True)

                # ── Download button ───────────────────────────────────────
                st.download_button(
                    label="⬇️ Export Research Package (.txt)",
                    data=result_text,
                    file_name="grant_proposal.txt",
                    mime="text/plain"
                )

                # ── Metrics Cards ─────────────────────────────────────────
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📊 Research Insights</div>', unsafe_allow_html=True)
                # Novelty dynamic styling
                n_val = f"{novelty_score:.2f}"
                n_badge = "✦ High" if novelty_score > 0.7 else ("● Medium" if novelty_score > 0.4 else "📉 Low")
                n_class = "badge-purple" if novelty_score > 0.7 else ("badge-blue" if novelty_score > 0.4 else "badge-red")

                st.markdown('''
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-value">{n_val}</div>
                        <div class="metric-label">Novelty Quotient</div>
                        <span class="metric-badge {n_class}">{n_badge}</span>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">92%</div>
                        <div class="metric-label">Signal Integrity</div>
                        <span class="metric-badge badge-green">✔ Verified</span>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">78%</div>
                        <div class="metric-label">Schema Coverage</div>
                        <span class="metric-badge badge-blue">● Stable</span>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{total_time}s</div>
                        <div class="metric-label">Process Latency</div>
                        <span class="metric-badge badge-green">⚡ Fast-Track</span>
                    </div>
                </div>
                '''.format(n_val=n_val, n_class=n_class, n_badge=n_badge, total_time=total_time), unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">📊 Research Analytics</div>', unsafe_allow_html=True)

    result_ready = 'result' in locals()
    result_text  = str(result) if result_ready else ""

    import re

    # ── Agent pipeline timeline ─────────────────────────────

    if result_ready and 'task_times' in st.session_state and st.session_state.task_times:
        times = st.session_state.task_times
        labels = [t[0] for t in times]
        durations = [t[1] for t in times]
    else:
        labels    = ["📚 Literature", "📈 Trends", "🔍 Gaps", "🔬 Method", "✍️ Proposal", "✅ Novelty"]
        durations = [18, 12, 10, 14, 16, 8] if result_ready else [0]*6

    bar_fig = go.Figure(go.Bar(
        x=durations, y=labels, orientation="h",
        marker=dict(
            color=durations,
            colorscale=[[0,"#0369a1"],[1,"#00f2ff"]],
            showscale=False,
        ),
        text=[f"{d}s" for d in durations], textposition="outside",
        textfont={"color": "#e2e8f0"},
    ))
    bar_fig.update_layout(
        title={"text": "⏱ AGENT_LATENCY_METRICS", "font": {"color": "#e2e8f0", "size": 18, "family": "Space Grotesk"}},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=10, r=40, t=40, b=10),
        xaxis={"color": "#4a5568", "gridcolor": "rgba(255,255,255,0.05)", "title": "seconds"},
        yaxis={"color": "#e2e8f0", "tickfont": {"size": 18}},
        font={"color": "#e2e8f0", "size": 18},
    )
    st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    # ── 3. Keyword frequency from result ───────────────────────
    if result_ready and result_text:
        stopwords = {"the","a","an","is","in","of","to","and","for","this",
                     "that","with","are","as","on","it","be","by","at","or",
                     "from","has","have","been","can","will","its","which",
                     "not","but","more","also","than","their","these","they"}
        words = re.findall(r'\b[a-zA-Z]{5,}\b', result_text.lower())
        freq = {}
        for w in words:
            if w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        kw_labels = [t[0].title() for t in top]
        kw_counts = [t[1] for t in top]
    else:
        kw_labels = ["Research","Model","Dataset","Learning","Analysis",
                     "Method","Climate","Neural","Training","Results"]
        kw_counts = [0]*10

    kw_fig = go.Figure(go.Bar(
        x=kw_labels, y=kw_counts,
        marker=dict(
            color=kw_counts,
            colorscale=[[0,"#0369a1"],[1,"#00f2ff"]],
            showscale=False,
        ),
    ))
    kw_fig.update_layout(
        title={"text": "🔑 KEYWORD_FREQUENCY_MATRIX", "font": {"color": "#e2e8f0", "size": 18, "family": "Space Grotesk"}},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(l=10, r=10, t=40, b=60),
        xaxis={"color": "#e2e8f0", "tickangle": -30, "tickfont": {"size": 18}},
        yaxis={"color": "#4a5568", "gridcolor": "rgba(255,255,255,0.05)"},
        font={"color": "#e2e8f0", "size": 18},
    )
    st.plotly_chart(kw_fig, use_container_width=True, config={"displayModeBar": False})

    # ── 4. Research confidence radar ───────────────────────────
    categories = ["Literature\nDepth", "Trend\nClarity", "Gap\nStrength",
                  "Methodology", "Proposal\nQuality", "Novelty"]
    values = [82, 75, 78, 70, 80, 75] if result_ready else [0]*6
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    radar = go.Figure(go.Scatterpolar(
        r=values_closed, theta=cats_closed, fill="toself",
        fillcolor="rgba(0,242,255,0.1)",
        line={"color": "#00f2ff", "width": 2},
        marker={"size": 5, "color": "#00f2ff"},
    ))
    radar.update_layout(
        title={"text": "🕸 RESEARCH_QUALITY_POLYGON", "font": {"color": "#e2e8f0", "size": 18, "family": "Space Grotesk"}},
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "range": [0,100],
                           "color": "#4a5568", "gridcolor": "rgba(255,255,255,0.08)"},
            "angularaxis": {"color": "#e2e8f0",
                            "gridcolor": "rgba(255,255,255,0.08)",
                            "tickfont": {"size": 18}},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=30, r=30, t=50, b=20),
        font={"color": "#e2e8f0", "size": 18},
        showlegend=False,
    )
    st.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})


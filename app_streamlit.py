from __future__ import annotations
import streamlit as st
import json
import os
import time
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

from dotenv import load_dotenv
from core.config import RuntimeConfig
from core.extractors.url_fetcher import fetch_url
from core.extractors.file_ingest import extract_text_from_file
from core.normalizers.text_clean import normalize_text
from core.rules_engine import load_rules, run_rules, rules_score
from core.providers.openai_provider import OpenAIProvider
from core.providers.anthropic_provider import AnthropicProvider
from core.providers.gemini_provider import GeminiProvider
from core.scoring import aggregate_score, final_decision
from core.report import build_report, to_markdown
from core.friction import generate_friction
from core.telemetry import append_log, read_logs

# Import your HTML guide
from spot_guide_with_logo import SPOT_GUIDE_HTML

load_dotenv()
st.set_page_config(
    page_title="Stop the Slop", 
    layout="wide",
    page_icon="assets/sts.png"  
)
st.sidebar.image("assets/sts.png", width='stretch')

@st.cache_data(show_spinner=False)
def get_prompt_text() -> str:
    with open("prompts/auditor_system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()

def analyze(content: str, meta: dict, rules_path: str, frictions_path: str, providers: list[str], model_map: dict, cfg: RuntimeConfig):
    rule_cfg = load_rules(rules_path)
    findings = run_rules(content, rule_cfg["rules"])
    rscore = rules_score(findings)
    llm_results = []
    system_prompt = get_prompt_text()
    for prov in providers:
        with st.spinner(f"Querying {prov}"):
            if prov == "openai" and cfg.openai_key:
                p = OpenAIProvider(cfg.openai_key)
                out = p.audit(content, findings, system_prompt, model_map["openai"], timeout=cfg.timeout_sec)
                out["_provider"] = "openai"
                llm_results.append(out)
            elif prov == "anthropic" and cfg.anthropic_key:
                p = AnthropicProvider(cfg.anthropic_key)
                out = p.audit(content, findings, system_prompt, model_map["anthropic"], timeout=cfg.timeout_sec)
                out["_provider"] = "anthropic"
                llm_results.append(out)
            elif prov == "gemini" and cfg.google_key:
                p = GeminiProvider(cfg.google_key, model_map["gemini"])
                out = p.audit(content, findings, system_prompt, model_map["gemini"], timeout=cfg.timeout_sec)
                out["_provider"] = "gemini"
                llm_results.append(out)
    agg = aggregate_score(rscore, llm_results, rule_cfg["score"])
    frictions = generate_friction(content, findings, llm_results, agg["combined_score"], rule_cfg["score"]["slop_threshold"], frictions_path, cfg)
    report = build_report(content, meta, findings, rscore, llm_results, agg, rule_cfg["score"]["slop_threshold"], providers, model_map, rule_cfg["version"], frictions)

    try:
        entry = {
            "ts": int(time.time()),
            "source": meta.get("source", "unknown"),
            "content_sha256": report.get("content_sha256"),
            "combined_score": report.get("combined_score"),
            "decision_slop": report.get("decision_slop"),
            "confidence": report.get("confidence"),
            "rules_hit": [f["rule_id"] for f in report.get("rules_findings", []) if f.get("hit")],
            "frictions": [f.get("id") for f in report.get("friction", [])],
            "providers": report.get("providers_used", []),
            "models": report.get("models_used", {})
        }
        append_log(entry)
    except Exception:
        pass

    return report

def _load_logs_df() -> pd.DataFrame:
    rows = read_logs()
    if not rows:
        return pd.DataFrame(columns=["ts","source","combined_score","decision_slop","confidence","rules_hit","frictions"])
    df = pd.DataFrame(rows)
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"], unit="s")
    else:
        df["ts"] = pd.Timestamp.utcnow()
    df["run_id"] = range(1, len(df) + 1)
    return df

def _top_counts(series, topn=10):
    import pandas as pd
    if series is None:
        return pd.DataFrame(columns=["item","count"])
    if series.apply(lambda x: isinstance(x, list)).any():
        s = series.explode().dropna()
    else:
        s = series.dropna()
    if s.empty:
        return pd.DataFrame(columns=["item","count"])
    vc = s.value_counts().head(topn).reset_index()
    vc.columns = ["item", "count"]
    return vc

def main():
    cfg = RuntimeConfig()
    
    # Initialize session state for content persistence
    if 'content' not in st.session_state:
        st.session_state.content = ""
    if 'meta' not in st.session_state:
        st.session_state.meta = {}

    # Sidebar inputs
    st.sidebar.subheader("Input")
    input_mode = st.sidebar.radio("Source", ["URL", "File", "Paste text"], horizontal=True)
    with st.sidebar.expander("SLOP STOP Config"):
        rules_file = st.text_input("Rules file", value="rules/rules.example.json")
        frictions_file = st.text_input("Friction policies", value="rules/friction_policies.example.json")
        max_chars = st.number_input("Max chars", min_value=10000, max_value=1000000, value=cfg.max_chars, step=5000)
        use_selenium = st.checkbox("Enable headless Selenium", value=cfg.enable_selenium)
        block_private = st.checkbox("Block private IPs for URLs", value=cfg.block_private_ips)

        provider_opts = []
        if cfg.openai_key:
            provider_opts.append("openai")
        if cfg.anthropic_key:
            provider_opts.append("anthropic")
        if cfg.google_key:
            provider_opts.append("gemini")
        providers = st.multiselect("LLM providers", provider_opts, default=provider_opts)
        model_map = {
            "openai": st.text_input("OpenAI model", value=cfg.default_models["openai"]),
            "anthropic": st.text_input("Anthropic model", value=cfg.default_models["anthropic"]),
            "gemini": st.text_input("Gemini model", value=cfg.default_models["gemini"]),
        }

    # Handle input modes - SAVE TO SESSION STATE
    if input_mode == "URL":
        url = st.sidebar.text_input("Enter URL")
        if st.sidebar.button("Fetch"):
            with st.spinner("Fetching"):
                text, meta = fetch_url(url, timeout_sec=cfg.timeout_sec, block_private_ips=block_private, use_selenium=use_selenium)
                st.session_state.content = normalize_text(text, max_chars)
                st.session_state.meta = meta
                st.sidebar.success(f"Fetched {len(st.session_state.content)} chars")
    elif input_mode == "File":
        f = st.sidebar.file_uploader("Upload a file", type=["pdf", "docx", "html", "htm", "md", "txt", "log", "ioc"])
        if f is not None:
            temp_path = f"uploaded_{int(time.time())}_{f.name}"
            with open(temp_path, "wb") as out:
                out.write(f.read())
            text, meta = extract_text_from_file(temp_path)
            os.remove(temp_path)
            st.session_state.content = normalize_text(text, max_chars)
            st.session_state.meta = meta
            st.sidebar.success(f"Ingested {len(st.session_state.content)} chars")
    else:
        pasted = st.sidebar.text_area("Paste text here")
        if st.sidebar.button("Use pasted"):
            st.session_state.content = normalize_text(pasted, max_chars)
            st.session_state.meta = {"source": "pasted"}

    # Load logs data
    df_logs = _load_logs_df()
    total_runs = int(df_logs.shape[0])
    slop_runs = int(df_logs["decision_slop"].sum()) if total_runs else 0
    not_slop = total_runs - slop_runs
    uniq_sources = int(df_logs["source"].nunique()) if total_runs else 0
    rule_hit_count = int(df_logs["rules_hit"].explode().dropna().shape[0]) if total_runs else 0

    # SIDEBAR: Metrics row
    st.sidebar.divider()
    st.sidebar.subheader("Statistics")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Analyses", total_runs)
        st.metric("Slop", slop_runs)
        st.metric("Clean", not_slop)
    with col2:
        st.metric("Sources", uniq_sources)
        st.metric("Rule hits", rule_hit_count)

    # SIDEBAR: Top lists
    st.sidebar.divider()
    with st.sidebar.expander("Top slop observations", expanded=False):
        top_rules = _top_counts(df_logs["rules_hit"] if total_runs else None, topn=10)
        if top_rules.empty:
            st.write("No data yet")
        else:
            st.dataframe(top_rules, use_container_width=True, hide_index=True)
    
    with st.sidebar.expander("Top friction points", expanded=False):
        top_fric = _top_counts(df_logs["frictions"] if total_runs else None, topn=10)
        if top_fric.empty:
            st.write("No data yet")
        else:
            st.dataframe(top_fric, use_container_width=True, hide_index=True)

    # SIDEBAR: Log viewer
    st.sidebar.divider()
    with st.sidebar.expander("Run log", expanded=False):
        if total_runs:
            show = df_logs.copy()
            show = show[["run_id","ts","source","combined_score","decision_slop","confidence","rules_hit","frictions"]]
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.write("No runs yet")

    # MAIN AREA: Visuals
    with st.expander("Visuals", expanded=False):
        if total_runs:
            fig1 = px.line(df_logs, x="ts", y="combined_score", markers=True, title="Combined score over time", color=df_logs["decision_slop"].map({True:"slop", False:"clean"}), color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig1, use_container_width=True)

            rule_counts = df_logs["rules_hit"].explode().dropna().value_counts().reset_index()
            rule_counts.columns = ["rule_id","count"]
            rule_counts = rule_counts.head(15)
            if not rule_counts.empty:
                fig2 = px.bar(rule_counts, x="rule_id", y="count", title="Top rule hits", color="count", color_continuous_scale="Plasma")
                fig2.update_layout(xaxis_title="", yaxis_title="count")
                st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.histogram(df_logs, x="combined_score", nbins=20, title="Score distribution", color=df_logs["decision_slop"].map({True:"slop", False:"clean"}), color_discrete_sequence=px.colors.sequential.Magma)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Run at least one analysis to populate charts")

    # MAIN AREA: Analysis runner
    if st.session_state.content:
        st.subheader("Preview")
        st.text_area("Normalized content sample", st.session_state.content[:2000], height=160, key="preview_area")
        if st.button("Run slop detection"):
            report = analyze(st.session_state.content, st.session_state.meta, rules_file, frictions_file, providers, model_map, cfg)
            st.subheader("Decision")
            st.metric("Slop", str(report["decision_slop"]), delta=f"{report['combined_score']:.3f}")
            st.write(f"Confidence: {report['confidence']:.2f}")
            st.subheader("Reasons")
            for r in report["overall_reasons"]:
                st.write(f"- {r}")
            st.subheader("Rule findings")
            st.json(report["rules_findings"])
            st.subheader("LLM results")
            st.json(report["llm_results"])
            st.subheader("Friction plan")
            st.json(report["friction"])

            j = json.dumps(report, indent=2)
            st.download_button("Download JSON report", data=j, file_name="slopwatch_report.json", mime="application/json")
            md = to_markdown(report)
            st.download_button("Download Markdown report", data=md, file_name="slopwatch_report.md", mime="text/markdown")
    
    # MAIN AREA: Spot the Slop guide at the bottom
    components.html(SPOT_GUIDE_HTML, height=8000, scrolling=True)

if __name__ == "__main__":
    main()

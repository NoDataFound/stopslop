from __future__ import annotations
import streamlit as st
import json
import os
import time
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
import re
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

from spot_guide_with_logo import SPOT_GUIDE_HTML

load_dotenv()
st.set_page_config(
    page_title="Stop the Slop",
    layout="wide",
    page_icon="assets/sts.png"
)

st.sidebar.image("assets/sts.png", width='stretch')

with st.sidebar.expander("haKCer Academy", expanded=True):
    st.image("assets/hackeracademy.png", width='stretch')
    st.markdown(
        """
        **What:** [haKCer Academy Article](https://www.linkedin.com/pulse/thank-you-subscribing-hakcer-academy-corian-cory-kennedy-sguwc/?trackingId=XS6mJXBLQ8KL8dhXtTd%2FPw%3D%3D)  
        **Where:** [haKC.io](https://hakc.io/) | [SecKC.academy](https://secKC.academy)
        """,
        unsafe_allow_html=False,
    )

@st.cache_data(show_spinner=False)
def get_prompt_text() -> str:
    with open("prompts/auditor_system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()


def _annotate_reasons(reasons: list[str], local_findings: list[dict]) -> list[str]:
    """Annotate LLM reasons that reference rules local engine didn’t hit."""
    local_hits = {f["rule_id"] for f in local_findings if f.get("hit")}
    local_rules = {f["rule_id"] for f in local_findings}
    out = []
    for r in reasons:
        m = re.match(r'^\[(llm|local):([^\]]+)\]\s*(.*)$', r.strip())
        if not m:
            out.append(r)
            continue
        src, rid, rest = m.groups()
        if src == "llm":
            if rid in local_hits:
                out.append(f"[llm:{rid}] {rest}")
            elif rid in local_rules:
                out.append(f"[llm:{rid}] {rest}  (note: local rule did not hit)")
            else:
                out.append(f"[llm:{rid}] {rest}  (note: rule not in local ruleset)")
        else:
            out.append(r)
    return out


def analyze(content: str, meta: dict, rules_path: str, frictions_path: str,
            providers: list[str], model_map: dict, cfg: RuntimeConfig):
    """Main analysis pipeline."""
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
    frictions = generate_friction(content, findings, llm_results,
                                  agg["combined_score"],
                                  rule_cfg["score"]["slop_threshold"],
                                  frictions_path, cfg)

    report = build_report(content, meta, findings, rscore, llm_results, agg,
                          rule_cfg["score"]["slop_threshold"], providers,
                          model_map, rule_cfg["version"], frictions)

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
        return pd.DataFrame(columns=["ts", "source", "combined_score",
                                     "decision_slop", "confidence",
                                     "rules_hit", "frictions"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce").fillna(pd.Timestamp.utcnow())
    df["run_id"] = range(1, len(df) + 1)
    return df


def _top_counts(series, topn=10):
    if series is None:
        return pd.DataFrame(columns=["item", "count"])
    if series.apply(lambda x: isinstance(x, list)).any():
        s = series.explode().dropna()
    else:
        s = series.dropna()
    if s.empty:
        return pd.DataFrame(columns=["item", "count"])
    vc = s.value_counts().head(topn).reset_index()
    vc.columns = ["item", "count"]
    return vc


def main():
    cfg = RuntimeConfig()
    st.session_state.setdefault("rules_file", "rules/rules.example.json")
    st.session_state.setdefault("frictions_file", "rules/friction_policies.example.json")
    st.session_state.setdefault("max_chars", int(cfg.max_chars))
    st.session_state.setdefault("use_selenium", bool(cfg.enable_selenium))
    st.session_state.setdefault("block_private", bool(cfg.block_private_ips))


    if "content" not in st.session_state:
        st.session_state.content = ""
    if "meta" not in st.session_state:
        st.session_state.meta = {}


    # === Sidebar: Input expander ===
    with st.sidebar.expander("Input & Fetch Content", expanded=True):
        st.subheader("Input")
        input_mode = st.radio("Source", ["URL", "File", "Paste text"], horizontal=True)

        if input_mode == "URL":
            url = st.text_input("Enter URL")
            if st.button("Fetch", use_container_width=True):
                with st.spinner("Fetching"):
                    text, meta = fetch_url(
                        url,
                        timeout_sec=cfg.timeout_sec,
                        block_private_ips=st.session_state["block_private"],
                        use_selenium=st.session_state["use_selenium"]
                    )
                    st.session_state.content = normalize_text(text, st.session_state["max_chars"])
                    st.session_state.meta = meta
                    st.success(f"Fetched {len(st.session_state.content)} chars")

        elif input_mode == "File":
            f = st.file_uploader("Upload a file",
                                 type=["pdf", "docx", "html", "htm", "md", "txt", "log", "ioc"])
            if f is not None and st.button("Ingest File", use_container_width=True):
                temp_path = f"uploaded_{int(time.time())}_{f.name}"
                with open(temp_path, "wb") as out:
                    out.write(f.read())
                text, meta = extract_text_from_file(temp_path)
                os.remove(temp_path)
                st.session_state.content = normalize_text(text, st.session_state["max_chars"])
                st.session_state.meta = meta
                st.success(f"Ingested {len(st.session_state.content)} chars")

        else:
            pasted = st.text_area("Paste text here")
            if st.button("Use Pasted Text", use_container_width=True):
                st.session_state.content = normalize_text(pasted, st.session_state["max_chars"])
                st.session_state.meta = {"source": "pasted"}
                st.success(f"Loaded {len(st.session_state.content)} chars from pasted input")

    # === Sidebar: Stats ===
    df_logs = _load_logs_df()
    total_runs = len(df_logs)
    slop_runs = int(df_logs["decision_slop"].sum()) if total_runs else 0
    not_slop = total_runs - slop_runs
    uniq_sources = int(df_logs["source"].nunique()) if total_runs else 0
    rule_hit_count = int(df_logs["rules_hit"].explode().dropna().shape[0]) if total_runs else 0

    st.sidebar.divider()
    st.sidebar.subheader("Statistics")
    stats_data = {
        "Analyses": total_runs,
        "Slop": slop_runs,
        "Clean": not_slop,
        "Sources": uniq_sources,
        "Rule hits": rule_hit_count,
    }
    stats_df = pd.DataFrame(list(stats_data.items()), columns=["Metric", "Count"])

    def _row_style(row):
        if row["Metric"] == "Slop":
            return ["background-color: #ffe6e6; color: #770000"] * 2
        if row["Metric"] == "Clean":
            return ["background-color: #e6ffe6; color: #004d00"] * 2
        return [""] * 2

    styled = (
        stats_df.style
        .apply(_row_style, axis=1)
        .hide(axis="index")
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "left"), ("font-size", "0.85rem"), ("font-weight", "600")]},
            {"selector": "td", "props": [("font-size", "0.85rem"), ("padding", "0.2rem 0.4rem")]}
        ])
    )
    st.sidebar.dataframe(styled, width='stretch', hide_index=True, height=160)

    with st.sidebar.expander("SLOP STOP Config", expanded=False):
        rules_file = st.text_input("Rules file", value=st.session_state["rules_file"])
        frictions_file = st.text_input("Friction policies", value=st.session_state["frictions_file"])
        max_chars = st.number_input("Max chars", min_value=10000, max_value=1000000,
                                    value=int(st.session_state["max_chars"]), step=5000)
        use_selenium = st.checkbox("Enable headless Selenium", value=st.session_state["use_selenium"])
        block_private = st.checkbox("Block private IPs for URLs", value=st.session_state["block_private"])

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

        # Persist to session_state for downstream use
        st.session_state["rules_file"] = rules_file
        st.session_state["frictions_file"] = frictions_file
        st.session_state["max_chars"] = int(max_chars)
        st.session_state["use_selenium"] = bool(use_selenium)
        st.session_state["block_private"] = bool(block_private)
        st.session_state["providers"] = providers
        st.session_state["model_map"] = model_map




    # === MAIN AREA ===
    with st.sidebar.expander("Visuals", expanded=False):
        if total_runs:
            fig1 = px.line(df_logs, x="ts", y="combined_score", markers=True,
                           title="Combined score over time",
                           color=df_logs["decision_slop"].map({True: "slop", False: "clean"}),
                           color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig1, use_container_width=True)

            rule_counts = df_logs["rules_hit"].explode().dropna().value_counts().reset_index()
            rule_counts.columns = ["rule_id", "count"]
            rule_counts = rule_counts.head(15)
            if not rule_counts.empty:
                fig2 = px.bar(rule_counts, x="rule_id", y="count", title="Top rule hits",
                              color="count", color_continuous_scale="Plasma")
                fig2.update_layout(xaxis_title="", yaxis_title="count")
                st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.histogram(df_logs, x="combined_score", nbins=20,
                                title="Score distribution",
                                color=df_logs["decision_slop"].map({True: "slop", False: "clean"}),
                                color_discrete_sequence=px.colors.sequential.Magma)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Run at least one analysis to populate charts")

    with st.sidebar.expander("Top slop observations", expanded=False):
        top_rules = _top_counts(df_logs["rules_hit"] if total_runs else None, topn=10)
        st.write("No data yet" if top_rules.empty else st.dataframe(top_rules, width='stretch', hide_index=True))

    with st.sidebar.expander("Top friction points", expanded=False):
        top_fric = _top_counts(df_logs["frictions"] if total_runs else None, topn=10)
        st.write("No data yet" if top_fric.empty else st.dataframe(top_fric, width='stretch', hide_index=True))

    # === Sidebar: Log viewer ===
    with st.sidebar.expander("Run log", expanded=False):
        if total_runs:
            show = df_logs[["run_id", "ts", "source", "combined_score",
                            "decision_slop", "confidence", "rules_hit", "frictions"]]
            st.dataframe(show, width='stretch', hide_index=True)
        else:
            st.write("No runs yet")
    # === MAIN AREA: Run analysis ===
    with st.expander("Run analysis", expanded=True):
        if st.session_state.content:
            st.subheader("Preview")
            st.text_area("Normalized content sample",
                         st.session_state.content[:2000],
                         height=160, key="preview_area")
            run_col, info_col = st.columns([1, 3], vertical_alignment="center")
            with run_col:
                if st.button("Run slop detection", width='stretch'):
                 
                    report = analyze(
                        st.session_state.content,
                        st.session_state.meta,
                        st.session_state["rules_file"],
                        st.session_state["frictions_file"],
                        st.session_state.get("providers", []),
                        st.session_state.get("model_map", {}),
                        cfg,
                    )

                    st.success("Decision")
                    st.metric("Slop", str(report["decision_slop"]),
                              delta=f"{report['combined_score']:.3f}")
                    st.code(f"Confidence: {report['confidence']:.2f}")

                    st.info("Reasons")
                    annotated = _annotate_reasons(report.get("overall_reasons", []),
                                                  report.get("rules_findings", []))
                    for r in annotated:
                        st.write(f"- {r}")
                    st.caption("Notes: provider reasons marked with 'local rule did not hit' "
                               "did not contribute to the score. 'rule not in local ruleset' "
                               "means your provider referenced a rule your local engine does not evaluate.")

                    st.info("Rule findings")
                    st.json(report["rules_findings"])
                    st.info("LLM results (informational only)")
                    st.json(report["llm_results"])
                    st.subheader("Friction plan")
                    st.json(report["friction"])

                    j = json.dumps(report, indent=2)
                    st.download_button("Download JSON report", data=j,
                                       file_name="slopwatch_report.json",
                                       mime="application/json")
                    md = to_markdown(report)
                    st.download_button("Download Markdown report", data=md,
                                       file_name="slopwatch_report.md",
                                       mime="text/markdown")
            with info_col:
                st.markdown(
                    """
                    **Tip**  
                    You can tweak providers and models from the sidebar.  
                    The preview shows the first 2000 characters after normalization.
                    """
                )
        else:
            st.info("No content loaded yet. Use the sidebar to provide a URL, file, or pasted text, then come back here to run the analysis.")

    components.html(SPOT_GUIDE_HTML, height=8000, scrolling=True)


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import json
import os
from typing import List
from dotenv import load_dotenv

from core.config import RuntimeConfig
from core.normalizers.text_clean import normalize_text
from core.extractors.url_fetcher import fetch_url
from core.extractors.file_ingest import extract_text_from_file
from core.rules_engine import load_rules, run_rules, rules_score
from core.providers.openai_provider import OpenAIProvider
from core.providers.anthropic_provider import AnthropicProvider
from core.providers.gemini_provider import GeminiProvider
from core.scoring import aggregate_score, final_decision
from core.report import build_report
from core.friction import generate_friction

def parse_args():
    p = argparse.ArgumentParser(description="SLOPwatch CLI")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="URL to analyze")
    src.add_argument("--file", help="File path to analyze")
    src.add_argument("--stdin", action="store_true", help="Read text from stdin")
    p.add_argument("--use-selenium", action="store_true", help="Render dynamic pages with headless Chrome")
    p.add_argument("--providers", nargs="+", default=[], help="Providers: openai anthropic gemini")
    p.add_argument("--model-map", type=str, default="{}", help='JSON map like {"openai":"gpt-4o-mini"}')
    p.add_argument("--rules", default="rules/rules.example.json")
    p.add_argument("--frictions", default="rules/friction_policies.example.json")
    p.add_argument("--out", help="Path to write JSON report")
    return p.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    cfg = RuntimeConfig()
    providers_req = args.providers or cfg.providers_available
    model_map = {**cfg.default_models, **json.loads(args.model_map)}
    if args.url:
        text, meta = fetch_url(args.url, timeout_sec=cfg.timeout_sec, block_private_ips=cfg.block_private_ips, use_selenium=args.use_selenium or cfg.enable_selenium)
    elif args.file:
        text, meta = extract_text_from_file(args.file)
    else:
        text = input()
        meta = {"source": "stdin"}
    text = normalize_text(text, cfg.max_chars)
    rule_cfg = load_rules(args.rules)
    findings = run_rules(text, rule_cfg["rules"])
    rscore = rules_score(findings)
    llm_results = []
    sys_prompt = _read_system_prompt()
    for prov in providers_req:
        if prov == "openai" and cfg.openai_key:
            p = OpenAIProvider(cfg.openai_key)
            out = p.audit(text, findings, sys_prompt, model_map["openai"], timeout=cfg.timeout_sec)
            out["_provider"] = "openai"
            llm_results.append(out)
        elif prov == "anthropic" and cfg.anthropic_key:
            p = AnthropicProvider(cfg.anthropic_key)
            out = p.audit(text, findings, sys_prompt, model_map["anthropic"], timeout=cfg.timeout_sec)
            out["_provider"] = "anthropic"
            llm_results.append(out)
        elif prov == "gemini" and cfg.google_key:
            p = GeminiProvider(cfg.google_key, model_map["gemini"])
            out = p.audit(text, findings, sys_prompt, model_map["gemini"], timeout=cfg.timeout_sec)
            out["_provider"] = "gemini"
            llm_results.append(out)
    agg = aggregate_score(rscore, llm_results, rule_cfg["score"])
    frictions = generate_friction(text, findings, llm_results, agg["combined_score"], rule_cfg["score"]["slop_threshold"], args.frictions, cfg)
    report = build_report(text, meta, findings, rscore, llm_results, agg, rule_cfg["score"]["slop_threshold"], providers_req, model_map, rule_cfg["version"], frictions)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

def _read_system_prompt() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "prompts", "auditor_system_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    main()

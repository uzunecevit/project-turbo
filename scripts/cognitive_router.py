"""
PROJECT-TURBO | Cognitive Router v0.8

Prompt karmaşıklığına göre optimal KV config seçer.
Mevcut bitwidth sweep sonuçlarından türetilmiş routing table.

Usage:
    LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python scripts/cognitive_router.py "prompt here"
    LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python scripts/cognitive_router.py --interactive
"""

import os
import sys
import re
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ═══════════════════════════════════════════════════════════════════════════════
# Routing Table — test sonuçlarından türetilmiş
# ═══════════════════════════════════════════════════════════════════════════════

# Qwen3-8B bitwidth sweep sonuçları (v0.8):
#
# Config          | Math | Key  | A/B  | KV Savings
# ----------------|------|------|------|----------
# q8_0/q8_0       | ✅   | ✅   | base | 0%
# q8_0/turbo4     | ✅   | ⚠️   | 89%  | 25%
# q8_0/turbo3     | ❌   | ✅   | 35%  | 29%
# turbo4/turbo4   | ✅   | ✅   | 40%  | 51%
# turbo3/turbo3   | ❌   | ✅   | 19%  | 59%
#
# Karar matrisi:
# - Reasoning gereken prompt → turbo4 en az (q8_0/turbo4 veya turbo4/turbo4)
# - Sadece memory recall → turbo3 yeterli (q8_0/turbo3 veya turbo3/turbo3)
# - Her şey → turbo4/turbo4 (güvenli sığınak)

ROUTING_TABLE = {
    # Task tipi → (type_k, type_v, açıklama)
    "reasoning_heavy": (42, 42, "turbo4/turbo4 — Math, logic, multi-step reasoning"),
    "reasoning_light": (8, 42, "q8_0/turbo4 — Hafif reasoning, özetleme"),
    "memory_recall": (8, 41, "q8_0/turbo3 — Key recall, basit hatırlama"),
    "generation": (42, 42, "turbo4/turbo4 — Metin üretimi, yaratıcı yazım"),
    "default": (42, 42, "turbo4/turbo4 — Güvenli varsayılan"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Complexity Scoring
# ═══════════════════════════════════════════════════════════════════════════════

# Reasoning göstergeleri (ağır)
HEAVY_REASONING_PATTERNS = [
    r"\b(?:step by step|adım adım)\b",
    r"\b(?:solve|çöz|calculate|hesapla)\b",
    r"\b(?:why|neden|how|nasıl)\b",
    r"\b(?:prove|kanıtla|justify|gerekçe)\b",
    r"\b(?:compare|karşılaştır|analyze|analiz)\b",
    r"\b(?:if.*then|eğer.*ise)\b",
    r"\b\d+\s*[%+\-*/]\s*\d+\b",  # matematiksel operatörler
    r"\b(?:increase|decrease|artır|azalt).*%\b",
    r"\b(?:multiply|divide|çarp|böl)\b",
    r"\b(?:theorem|teorem|lemma|proof|ispat)\b",
]

# Reasoning göstergeleri (hafif)
LIGHT_REASONING_PATTERNS = [
    r"\b(?:summar|özet)\b",
    r"\b(?:explain|açıkla|describe|tanımla)\b",
    r"\b(?:list|liste|enumerate|say)\b",
    r"\b(?:what is|nedir|kimdir)\b",
]

# Memory recall göstergeleri
MEMORY_RECALL_PATTERNS = [
    r"\b(?:remember|hatırla|recall|çağrıştır)\b",
    r"\b(?:the key is|anahtar|key.*=)\b",
    r"\b\d{4,6}\b",  # 4-6 haneli sayılar (key recall)
    r"\b(?:what was|ne idi|earlier|önceki)\b",
    r"\b(?:my name is|adım|benim)\b",
]

# Generation göstergeleri
GENERATION_PATTERNS = [
    r"\b(?:write|yaz|compose|oluştur)\b",
    r"\b(?:story|hikâye|poem|şişir|essay|deneme)\b",
    r"\b(?:create|yarat|generate|üret)\b",
    r"\b(?:tell me about|anlat|hakkında bilgi)\b",
]


def score_complexity(prompt: str) -> dict:
    """Prompt'un karmaşıklığını skorla."""
    prompt_lower = prompt.lower()

    scores = {
        "heavy_reasoning": 0,
        "light_reasoning": 0,
        "memory_recall": 0,
        "generation": 0,
    }

    # Heavy reasoning
    for pat in HEAVY_REASONING_PATTERNS:
        if re.search(pat, prompt_lower):
            scores["heavy_reasoning"] += 2

    # Light reasoning
    for pat in LIGHT_REASONING_PATTERNS:
        if re.search(pat, prompt_lower):
            scores["light_reasoning"] += 1

    # Memory recall
    for pat in MEMORY_RECALL_PATTERNS:
        if re.search(pat, prompt_lower):
            scores["memory_recall"] += 1

    # Generation
    for pat in GENERATION_PATTERNS:
        if re.search(pat, prompt_lower):
            scores["generation"] += 1

    # Prompt uzunluğu bonusu
    if len(prompt) > 500:
        scores["heavy_reasoning"] += 1
    if len(prompt) > 1000:
        scores["heavy_reasoning"] += 1

    return scores


def select_config(scores: dict) -> tuple:
    """Skorlara göre optimal config seç."""
    dominant = max(scores, key=scores.get)

    if dominant == "heavy_reasoning" and scores["heavy_reasoning"] >= 2:
        return ROUTING_TABLE["reasoning_heavy"]
    elif dominant == "light_reasoning" and scores["light_reasoning"] >= 1:
        return ROUTING_TABLE["reasoning_light"]
    elif dominant == "memory_recall" and scores["memory_recall"] >= 1:
        return ROUTING_TABLE["memory_recall"]
    elif dominant == "generation" and scores["generation"] >= 1:
        return ROUTING_TABLE["generation"]
    else:
        return ROUTING_TABLE["default"]


def route(prompt: str) -> dict:
    """Prompt'u analiz et ve optimal config döndür."""
    scores = score_complexity(prompt)
    type_k, type_v, desc = select_config(scores)

    return {
        "prompt_preview": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        "scores": scores,
        "dominant": max(scores, key=scores.get),
        "config": (type_k, type_v),
        "description": desc,
        "kv_savings": {
            (8, 8): "0%",
            (8, 42): "~25%",
            (8, 41): "~29%",
            (42, 42): "~51%",
            (41, 41): "~59%",
        }.get((type_k, type_v), "unknown"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Cognitive Router")
    parser.add_argument("prompt", nargs="?", help="Prompt to analyze")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive:
        print("Cognitive Router — Interactive Mode")
        print("Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("Prompt > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue

            result = route(prompt)
            print(f"  Task:      {result['dominant']}")
            print(f"  Config:    {result['config'][0]}/{result['config'][1]}")
            print(f"  Desc:      {result['description']}")
            print(f"  KV Save:   {result['kv_savings']}")
            print(f"  Scores:    {result['scores']}")
            print()

    elif args.prompt:
        result = route(args.prompt)
        print(f"Prompt:    {result['prompt_preview']}")
        print(f"Task:      {result['dominant']}")
        print(f"Config:    {result['config'][0]}/{result['config'][1]}")
        print(f"Desc:      {result['description']}")
        print(f"KV Save:   {result['kv_savings']}")

    else:
        # Demo: test birkaç prompt
        test_prompts = [
            "Solve this step by step: A number is increased by 20%, then decreased by 20%, then increased by 50%. Final result is 108. What was the original number?",
            "The key is 84729. What is the key?",
            "Summarize the concept of recursion in programming.",
            "Write a short poem about technology.",
            "Explain quantum computing in simple terms.",
        ]

        print("Cognitive Router — Demo Mode\n")
        for prompt in test_prompts:
            result = route(prompt)
            print(f"Prompt: {result['prompt_preview']}")
            print(
                f"  → {result['dominant']}: {result['config'][0]}/{result['config'][1]} ({result['description']})"
            )
            print(f"  Scores: {result['scores']}")
            print()


if __name__ == "__main__":
    main()

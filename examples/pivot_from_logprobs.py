from __future__ import annotations

from model_belief.belief import compute_belief
from model_belief.config.loader import load_choice_map
from model_belief.pivot import PivotContext, get_pivot_finder, resolve_pivot


def main() -> None:
    # This example assumes you already have output text and token-level logprobs
    # from a previous LLM call (saved or cached). We skip generation entirely.
    choice_yaml = "configs/choice_maps/diaper_first_token.yaml"
    choice_map = load_choice_map(choice_yaml)

    existing_text = "Answer: Pampers."

    # Minimal mock of token-level logprobs content (aligned to output tokens).
    # Each item is a dict with a realized token and a list of top_logprobs.
    existing_logprobs_content = [
        {"token": "Answer", "logprob": -0.05, "top_logprobs": []},
        {"token": ":", "logprob": -0.01, "top_logprobs": []},
        {
            "token": " P",
            "logprob": -0.20,
            "top_logprobs": [
                {"token": " P", "logprob": -0.20},
                {"token": " H", "logprob": -0.45},
                {"token": " L", "logprob": -1.10},
                {"token": " None", "logprob": -1.40},
            ],
        },
        {"token": "ampers", "logprob": -0.02, "top_logprobs": []},
        {"token": ".", "logprob": -0.01, "top_logprobs": []},
    ]

    # 1) Pivot detection (uses first_token_available strategy)
    finder = get_pivot_finder(choice_map)
    ctx = PivotContext(text=existing_text)
    pivot = resolve_pivot(finder, existing_logprobs_content, choice_map, ctx)

    print("\n[Step 1] Pivot detection")
    print({
        "pivot_index": pivot.index,
        "pivot_token": pivot.token,
        "pivot_choice": pivot.choice_id,
        "method": pivot.method,
    })

    # 2) Belief computation at the pivot
    belief = compute_belief(
        logprobs_content=existing_logprobs_content,
        pivot_index=pivot.index,
        choice_map=choice_map,
    )

    print("\n[Step 2] Belief extraction at pivot")
    for item in belief.items:
        print(f"  {item.alt_id}: {item.prob:.4f}")


if __name__ == "__main__":
    main()

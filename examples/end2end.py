# examples/minimal_e2e.py
from __future__ import annotations

from model_belief.config.loader import load_choice_map
from model_belief.llm import generate_with_logprobs
from model_belief.pivot import get_pivot_finder, PivotContext, resolve_pivot
from model_belief.belief import compute_belief

def main() -> None:
    models_yaml = "configs/models.yaml"
    choice_yaml = "configs/choice_maps/diaper.yaml"

    print("\n=== Model-Belief End-to-End Example ===")
    print("Models config:", models_yaml)
    print("Choice map config:", choice_yaml)

    system_prompt = ("You are making a routine purchase decision in a store. "
                     "You are choosing among products based on overall impressions such as quality, comfort, and price."
                     "Your preferences are not perfectly sharp: when options are similar, small differences may not lead to a clear or consistent choice."
                     "There is no single right answer. Indicate which option you would choose, if any. "
                     "Answer naturally, as you would in a quick shopping situation. Do not choose more than one option.")

    user_prompt = ("You see the following baby diaper brands in the store: Pampers and Huggies. "
                   "Pampers diapers are generally described as soft and include a wetness indicator (though experiences vary), and are often perceived as slightly more trusted and premium overall. "
                   "Huggies diapers are often noted for having a snug fit and for helping to prevent leaks, though the fit and effectiveness can depend on the baby. "
                   "When choosing diapers, you weigh functionality, brand trust, and saving money. Comfort and reliability often matter more to you than small price differences, but as price gaps grow, saving money becomes increasingly important. "
                   "There is no clear cutoff for when a price difference feels “small” versus “large,” and it can depend on your overall impression in the moment. "
                   "The unit prices are 35.6 cents per Pampers diaper and 30.0 cents per Huggies diaper. "
                   "Indicate which brand you would choose to buy, or choose “neither” if you prefer other diaper brands. Answer naturally.")

    input_payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print("\n[Step 1] LLM generation")
    print("Prompt sent to model:\n", input_payload)

    # 1) Generate (text + token-level top_logprobs from OpenAI Responses API)
    gen = generate_with_logprobs(models_yaml_path=models_yaml, input=input_payload)

    print("\nLLM raw text output:\n", gen.text)
    print("Number of output tokens:", len(gen.logprobs_content))

    if not gen.logprobs_content:
        raise RuntimeError(
            "No token logprobs found in the response. Ensure models.yaml sets include: [message.output_text.logprobs] and top_logprobs > 0."
        )

    # 2) Pivot (choose finder based on choice_map.config.pivot.active)
    choice_map = load_choice_map(choice_yaml)

    print("\n[Step 2] Choice map & pivot strategy")
    print("Pivot strategy:", choice_map.config.pivot.active)
    print("Token universe:", sorted(choice_map.tokens.token_universe))

    finder = get_pivot_finder(choice_map)

    ctx = PivotContext(
        text=gen.text,  # used by llm_judge; rule-based ignores it
        metadata={"models_yaml_path": models_yaml},
    )
    pivot = resolve_pivot(finder, gen.logprobs_content, choice_map, ctx)

    print("\n[Step 3] Pivot detection")
    print({
        "pivot_index": pivot.index,
        "pivot_token": pivot.token,
        "pivot_choice": pivot.choice_id,
        "method": pivot.method,
    })

    # 3) Belief at pivot (softmax over choice-universe logits; aggregate by alternative)
    belief = compute_belief(
        logprobs_content=gen.logprobs_content,
        pivot_index=pivot.index,
        choice_map=choice_map,
    )

    print("\n[Step 4] Belief extraction at pivot")
    for item in belief.items:
        print(f"  {item.alt_id}: {item.prob:.4f}")


if __name__ == "__main__":
    main()
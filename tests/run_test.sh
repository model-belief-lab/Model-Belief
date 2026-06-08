: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in your shell before running}"
export RUN_OPENAI_INTEGRATION=1

## run llm responses tests
#pytest -m integration -s tests/test_llm_openai_client.py
#pytest -m integration -s tests/test_llm_openai_response.py
#pytest -m integration -s tests/test_llm_generate.py

## run pivot llm tests
pytest -q tests/test_pivot_llm.py -m integration

## run belief tests
#pytest -s tests/test_belief.py
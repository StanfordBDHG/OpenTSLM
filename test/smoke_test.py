import importlib

mods = [
    "opentslm",
    "opentslm.model.llm.OpenTSLM",
    "opentslm.model.llm.OpenTSLMFlamingo",
    "opentslm.prompt.full_prompt",
]

[importlib.import_module(m) for m in mods]
print("Smoke test passed: all modules imported successfully.")

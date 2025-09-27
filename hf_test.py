from src import OpenTSLM, TextPrompt, TextTimeSeriesPrompt, FullPrompt

# Load model
model = OpenTSLM.load_pretrained("OpenTSLM/gemma-3-270m-pt-har-flamingo")

# Create prompt with raw time series data (normalization handled automatically)
prompt = FullPrompt(
    pre_prompt=TextPrompt("You are an expert in HAR analysis."),
    text_time_series_prompt_list=[
        TextTimeSeriesPrompt("X-axis accelerometer", [2.34, 2.34, 7.657, 3.21, -1.2])
    ],
    post_prompt=TextPrompt("What activity is this? Reasn step by step providing a full rationale before replying.")
)

# Generate response
output = model.eval_prompt(prompt, normalize=True)
print(output)

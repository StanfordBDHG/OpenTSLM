from src import OpenTSLM, TextPrompt, TextTimeSeriesPrompt, FullPrompt

# Load model
model = OpenTSLM.load_pretrained("OpenTSLM/llama-3.2-1b-har-sp")

# Create prompt with raw time series data (normalization handled automatically)
prompt = FullPrompt(
    pre_prompt=TextPrompt("You are an expert in HAR analysis."),
    text_time_series_prompt_list=[
        TextTimeSeriesPrompt("X-axis accelerometer", [2.34, 2.34, 7.657, 3.21, -1.2])
    ],
    post_prompt=TextPrompt("What activity is this?")
)

# Generate response
output = model.eval_prompt(prompt, normalize=True)
print(output)

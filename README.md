# EmbedHealth


## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-org/embedhealth.git
   cd embedhealth
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## LLM Setup

EmbedHealth uses the Llama 3.2 1B model, which is stored in a Hugging Face repository which is restricted. Follow these steps to gain access and download:

1. **Request Access**  
   Submit a request to the repository administrator to gain read access.

2. **Authenticate with Hugging Face**  
   Log in to your Hugging Face account and configure the CLI:

   ```bash
   huggingface-cli login
   ```

3. **Create an API Token**  
   - Go to your Hugging Face settings: https://huggingface.co/settings/tokens
   - Generate a new token with `read` scope.
   - Copy the token for CLI login.

4. **Download the Model**  
   ```bash
   python scripts/download_model.py --model llama-3.2-1b
   ```

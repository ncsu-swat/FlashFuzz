
python3 -u llm_gemini.py --out-dir testharness/tf_cpu  --model gemini-3-pro-preview --max-tokens 60000 --retries 1 --temperature 1 --api-key  --timeout 300   > gemini_pro_output.log  2>&1


import os
import sys
import traceback

# Setup dummy api key file
key_path = "dummy_key.txt"
with open(key_path, "w") as f:
    f.write("sk-test-key")

try:
    print("Testing Imports...")
    from calvin_utils.gpt_sys_review.gpt_utils.openai_base import OpenAIBase
    from calvin_utils.gpt_sys_review.gpt_utils.openai_chat_base import OpenAIChatBase
    from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator
    from calvin_utils.gpt_sys_review.gpt_utils.title_screening import TitleScreener
    print("Imports Successful.")

    print("Testing OpenAIBase Instantiation...")
    # Standard
    base = OpenAIBase(api_key_path=key_path, is_azure=False)
    print("OpenAIBase (Standard) Initialized.")
    # Azure
    base_azure = OpenAIBase(api_key_path=key_path, is_azure=True, api_base="https://example.com", api_version="2023-01-01")
    print("OpenAIBase (Azure) Initialized.")

    print("Testing OpenAIChatBase Instantiation...")
    chat = OpenAIChatBase(api_key_path=key_path, question_type="extraction", is_azure=False)
    print("OpenAIChatBase Initialized.")

    print("Testing OpenAIJsonEvaluator Instantiation...")
    # Mock json file
    with open("dummy.json", "w") as f:
        f.write('{"file1": {"key": "value"}}')
    
    evaluator = OpenAIJsonEvaluator(
        api_key_path=key_path,
        json_file_path="dummy.json",
        keys_to_consider=["key"],
        question={"q1": "test"},
        question_type="extraction"
    )
    print("OpenAIJsonEvaluator Initialized.")

    print("Testing TitleScreener Instantiation...")
    # Mock csv
    with open("dummy.csv", "w") as f:
        f.write("Title,Abstract\nTest Title,Test Abstract")
    
    try:
        screener = TitleScreener(
            api_key_path=key_path,
            csv_path="dummy.csv",
            question="Is this relevant?"
        )
        print("TitleScreener Initialized.")
    except TypeError as e:
        print(f"TitleScreener Instantiation Failed (Expected if bad super call): {e}")

except Exception:
    traceback.print_exc()
finally:
    # Cleanup
    if os.path.exists(key_path):
        os.remove(key_path)
    if os.path.exists("dummy.json"):
        os.remove("dummy.json")
    if os.path.exists("dummy.csv"):
        os.remove("dummy.csv")
    if os.path.exists("dummy_cleaned.csv"):
        os.remove("dummy_cleaned.csv")

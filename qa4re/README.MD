## How to run?
1. make sure you have the datasets in the correct paths.
2. Run the generate-prompts.py to get the prompts that will be used by the LLM models.
3. Add the login token to enable the access to HuggingFace models
    - login("TOKEN_HERE") 
4. Make sure you have the required libs in requirements.txt (probably you will need more libs)
5. Choose the dataset to test and the LLM model (statically typed in the code) to run and then use: `python run_qa4re.py` 
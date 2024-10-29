from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Please, refer to https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF to download LLM.

MAX_TOKENS=2000

# Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = LlamaCpp(
    #model_path="models/llama-2-7b-chat.Q4_0.gguf",
    model_path="models/llama-2-7b-chat.Q6_K.gguf",
    n_gpu_layers=40,
    max_tokens=MAX_TOKENS,
    n_batch=512,  # Batch size for model processing
    verbose=False,  # Enable detailed logging for debugging
)

# Define the prompt template with a placeholder for the question
template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt, llm=llm)

print("Chatbot initialized, ready to chat...")
while True:
    question = input("> ")
    answer = llm_chain.run(question)
    print(answer, '\n')
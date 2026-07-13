import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_NAME, PROMPT_FILE

class InferenceEngine:
    def __init__(self):
        # LangChain's OpenAI wrapper pointed at your local Exo node
        self.llm = ChatOpenAI(
            openai_api_base="http://localhost:52415/v1",
            api_key="exo-local-token", # Required by SDK, ignored by Exo
            name=MODEL_NAME,
            temperature=0.1,
            max_completion_tokens=512
        )
        
        try:
            with open(PROMPT_FILE, "r") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            system_prompt = "You are a logistics assistant answering supply chain questions."

        # Define the LangChain prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"{system_prompt}\n\nContext:\n{{context}}"),
            ("user", "{query}")
        ])
        
        # Build the LangChain Expression Language (LCEL) chain
        self.chain = self.prompt_template | self.llm

    def generate(self, user_query: str, retrieved_context: str) -> tuple[str, float, int]:
        """Executes the LangChain LCEL chain."""
        start_time = time.time()
        
        try:
            # Invoke the chain with our dictionary variables
            response = self.chain.invoke({
                "context": retrieved_context,
                "query": user_query
            })
            
            response_text = response.content
            
            # Extract tokens from LangChain's response metadata
            token_usage = response.response_metadata.get("token_usage", {})
            total_tokens = token_usage.get("total_tokens", 0)
            
        except Exception as e:
            print(f"\n[!] LangChain to Exo connection failed: {e}")
            response_text = "Error: Could not communicate with local Exo inference node."
            total_tokens = 0
            
        latency = time.time() - start_time
        return response_text, latency, total_tokens
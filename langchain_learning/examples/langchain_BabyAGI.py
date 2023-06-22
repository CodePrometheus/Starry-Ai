# https://python.langchain.com/en/latest/use_cases/autonomous_agents/autogpt.html
import sys

sys.path.append(r"../")
# from langchain.llms import OpenAI, Anthropic
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,
    verbose=True
)

from typing import Optional

from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental import BabyAGI

# Connect to the Vector Store
# Depending on what vectorstore you use, this step may look different.

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# Run the BabyAGI
# Now it’s time to create the BabyAGI controller and watch it try to accomplish your objective.

OBJECTIVE = "利用  OpenAI 提供的 ChatGPT api 进行相似匹配"
# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)
baby_agi({"objective": OBJECTIVE})

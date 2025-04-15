from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_BASE= os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

# --------- Prompt per query generation ---------
query_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        devi cercare documenti con una query all'interno di un grande database di documenti legali, l'utente fa una domanda, estrai dalla domanda le migliori keywords per la ricerca. rispondi nel formato query: keywords"""),
    ("human", "DOMANDA: {question}")
])


# --------- Prompt per riassunto (sintesi) ---------
sintesi_prompt = ChatPromptTemplate.from_messages([
    ("system", """Sei un esperto di diritto italiano. Fai una sintesi molto breve estratta dai documenti:
    1. Tematica di riferimento.
    1. Articoli di legge con riferimenti completi, 
    2. Procedure ufficiali documentate"""),
    ("human", "DOCUMENTI DA ANALIZZARE:\n{document}")
])

# --------- Prompt template per RAG ---------
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Sei un esperto di diritto italiano. Rispondi:
    - Usa solo i documenti forniti
    - cita la legislazione collegata con riferimenti espliciti
    - RISPONDI IN ITALIANO
    - Se non trovi riscontri, specifica "Non posso rispondere"""),
    ("human", "DOCUMENTI DI RIFERIMENTO:\n{document}\n\nDOMANDA: {question}")
])

'''
    ("system", "<|im_start|>system\n"
               "Sei un esperto di diritto italiano. Rispondi:\n"
               "- Usa solo i documenti forniti\n"
               "- Cita le leggi quando presenti\n"
               "- RISPONDI IN ITALIANO\n"
               "- Se non trovi riscontri, specifica \"Non posso rispondere\"\n"
               "<|im_end|>"),
    ("human", "<|im_start|>user\n"
              "DOCUMENTI DI RIFERIMENTO:\n{document}\n\nDOMANDA: {question}\n"
              "<|im_end|>")
'''

# Initialize your LLM instance
llm = ChatOpenAI(
    openai_api_base=OPENAI_API_BASE,  
    openai_api_key=OPENAI_API_KEY,                   
    model_name="qwen2.5-7b-instruct-q4_0",    # non necessario, solo descrittivo
    temperature=0
)



query_chain = query_prompt | llm | StrOutputParser()
sintesi_chain = sintesi_prompt | llm | StrOutputParser()
rag_chain = rag_prompt | llm | StrOutputParser()

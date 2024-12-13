import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import tiktoken

# API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Token cost ---
PRICE_PER_1K_TOKENS = 0.0025  # GPT-4o:  1,000 tokens cost


# --- Embedding Modeli ve VectorStore ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# --- LLM Model ---
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)


template = """
You are a multilingual film assistant. Always respond in the same language as the user input.

If the user says greetings like "hello", "hi", "selam", "merhaba", respond with a friendly message:
"Hello! How can I assist you with movies today?"

If the user asks a question related to movies but there is no relevant context available in the knowledge base, respond:
"I'm sorry, I couldn't find any information about that movie."

If the user asks anything unrelated to movies, respond:
"I am a film assistant. Please ask me something related to movies."

Relevant Context:
{context}

Conversation History:
{history}

User: {user_input}
Assistant:"""


film_prompt = PromptTemplate(template=template, input_variables=["history", "user_input", "context"])


# --- LangChain Memory ---
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# --- Token Function ---
tokenizer = tiktoken.encoding_for_model("gpt-4")  
def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


# --- Search and Answer Check ---
# threshold 0.4  = It takes from among the similarities of 40% and above.
def retrieve_documents(query,threshold=0.4):

    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=3)

    # Filter by threshold value
    filtered_docs = [(doc, score) for doc, score in docs_with_scores if score >= threshold]

    return filtered_docs

def generate_response(query,threshold=0.4):
    retrieved_docs = retrieve_documents(query,threshold=threshold)
    #print("docs :",retrieved_docs)

    if not retrieved_docs:  # If no documents are found

        #context=""
        #return "Sorry, no movie information was found for what you were looking for. :/",context
        context = ""  # Boş bağlam
        # Prompt'u doldur LLM'e gönder
        filled_prompt = film_prompt.format(
            history=memory.load_memory_variables({})["history"],
            user_input=query,
            context=context
        )
        # LLM ile yanıt oluştur
        response = llm.predict(filled_prompt)
        return response, context

    
    context = "\n".join([doc.page_content for doc, _ in retrieved_docs])
    
    # Prompt fill
    filled_prompt = film_prompt.format(
        history=memory.load_memory_variables({})["history"],
        user_input=query,
        context=context
    )
    
    # Create an answer using LLM
    response = llm.predict(filled_prompt)
    
    return response,context

def total_tokens_in_prompt(history, user_input, response, template,context=""):
    """Number of tokens including prompt """
    # Prompt Template fill
    full_prompt = template.format(history=history, user_input=user_input, context=context)
    
    # token detail calculate
    prompt_tokens = count_tokens(full_prompt)
    response_tokens = count_tokens(response)
    
    total_tokens = prompt_tokens + response_tokens
    return total_tokens, prompt_tokens, response_tokens

# --- Streamlit frontend ---
st.title("🎬 Film Asistant Bot")
st.write("You can ask questions about movies  😊")

# chat reset
if "conversation" not in st.session_state:
    st.session_state.conversation = []

reset = st.button("New Chat")
if reset:
    st.session_state.conversation = []
    memory.clear()

user_input = st.text_input("Write your question here...", key="input")

if user_input:
    # answer generation
    response, context = generate_response(user_input)
    
    # chat history
    history = memory.load_memory_variables({})["history"]
    
    # Token calculate
    total_tokens, prompt_tokens, response_tokens = total_tokens_in_prompt(
        history=history, user_input=user_input, response=response, template=template, context=context
    )
    cost = (total_tokens / 1000) * PRICE_PER_1K_TOKENS

    # Chat history update
    memory.save_context({"input": user_input}, {"output": response})
    
    # Hide Chat
    st.session_state.conversation.append({
        "user": user_input,
        "bot": response,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "cost": cost
    })

    # Chat history show
    for chat in st.session_state.conversation:
        st.write(f"**Siz:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")
        st.write(f"🔢 Prompt Token: {chat['prompt_tokens']}, Answer Token: {chat['response_tokens']}")
        st.write(f"🔢 Sum Token: {chat['total_tokens']} | 💲 Estimated Cost: ${chat['cost']:.6f}")

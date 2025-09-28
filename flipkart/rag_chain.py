from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config

class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.3)
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k":3})

        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
                  Stick to context. Be concise, clear, and professional.

                  Showcase formatting rules:
                  - Always return multiple items as a **numbered list (1., 2., 3., etc.)**.
                  - Each item must be on its **own line**.
                  - After each item, insert a **newline** (`\\n`) so they don't run together.
                  - Format: Product Name – 1–2 **bold key features**.
                  - End with a short concluding sentence (e.g., "These are the most recommended options based on reviews.").

                  Example:
                  QUESTION: Suggest me top 5 headphones
                  ANSWER:
                  1. **HyperX Cloud II** – **7.1 surround sound**, **memory foam ear cushions**
                  2. **SteelSeries Arctis 7** – **Long-lasting battery**, **wireless connectivity**
                  3. **Sennheiser GSP 670** – **Advanced noise cancellation**, **ergonomic design**
                  4. **Turtle Beach Recon 200** – **Durable construction**, **crystal-clear audio**
                  5. **Razer Kraken X** – **Customizable lighting**, **7.1 surround sound**

                  These headphones are the most recommended based on reviews.

                  CONTEXT:
                  {context}

                  QUESTION: {input}"""),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{input}")  
])




        # qa_prompt = ChatPromptTemplate.from_messages([
        #     ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
        #                   Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""),
        #     MessagesPlaceholder(variable_name="chat_history"), 
        #     ("human", "{input}")  
        # ])

        # Retrieve Chat History
        history_aware_retriever = create_history_aware_retriever(
            self.model , retriever , context_prompt
        )

        # Question Answer Chain
        question_answer_chain = create_stuff_documents_chain(
            self.model , qa_prompt
        )

        
        rag_chain = create_retrieval_chain(
            history_aware_retriever,question_answer_chain
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )


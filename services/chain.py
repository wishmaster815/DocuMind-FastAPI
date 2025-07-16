from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever




def build_conversational_chain(llm, retriever):
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. Do NOT answer the question."),
            MessagesPlaceholder('chat_history'),
            ("user", "{input}")
        ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever,contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following retrieved context to answer "
         "the question. If you don't know the answer, say 'I don't know'. Keep the answer concise.\n\n{context}"),
         MessagesPlaceholder("chat_history"),
         ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever,qa_chain)
import pdf_04_retrieve
from util import prompts
from util.instances import BEG_MODEL, TOP_K, LLM
from langchain_core.prompts import ChatPromptTemplate


def pdf_retrieve_chain(query):

    result_docs = pdf_03_retrieve.search(query, BEG_MODEL, LLM, TOP_K)
    prompt = ChatPromptTemplate.from_template(prompts.PDF_ANSWER_TEMPLATE)
    chain = prompt | LLM
    result = chain.invoke({"retrieve_docs": result_docs, "question": query})
    return result

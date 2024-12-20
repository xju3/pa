import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI  
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI

load_dotenv()

_llm_instance = None

def get_llm_instance():
    global _llm_instance
    if _llm_instance is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_api = os.getenv("LM_STUDIO")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        _llm_instance = ChatOpenAI(api_key=openai_api_key, api_base=openai_base_api )
    return _llm_instance

def create_email_chain(llm):
    email_prompt = PromptTemplate(
        input_variables=["context"],
        template="You are drafting a professional email based on the following context:\n\n{context}\n\nProvide the complete email below."
    )
    return LLMChain(llm=llm, prompt=email_prompt)
def create_study_plan_chain(llm):
    study_plan_prompt = PromptTemplate(
        input_variables=["topic", "duration"],
        template="Create a detailed study plan for learning about {topic} over the next {duration}."
    )
    return LLMChain(llm=llm, prompt=study_plan_prompt)


def create_knowledge_qna_chain(llm):
    qna_prompt = PromptTemplate(
        input_variables=["question", "domain"],
        template="Provide a detailed answer to the following question within the context of {domain}:\n\n{question}"
    )
    return LLMChain(llm=llm, prompt=qna_prompt)


def create_action_items_chain(llm):
    action_items_prompt = PromptTemplate(
        input_variables=["notes"],
        template="Extract and list the main action items from the following meeting notes:\n\n{notes}"
    )
    return LLMChain(llm=llm, prompt=action_items_prompt)

def initialize_agent_executor():
    llm = get_llm_instance()
    tools = [
        Tool(
            name="DraftEmail",
            func=lambda context: create_email_chain(llm).run(context=context),
            description="Draft a professional email based on a given context. This tool is specifically for email drafting."
        ),
        Tool(
            name="GenerateStudyPlan",
            func=lambda topic, duration: create_study_plan_chain(llm).run(topic=topic, duration=duration),
            description="Generate a study plan for a topic over a specified duration."
        ),
        Tool(
            name="KnowledgeQnA",
            func=lambda question, domain: create_knowledge_qna_chain(llm).run(question=question, domain=domain),
            description="Answer a question based on a specified knowledge domain."
        ),
        Tool(
            name="ExtractActionItems",
            func=lambda notes: create_action_items_chain(llm).run(notes=notes),
            description="Extract action items from meeting notes."
        )
    ]
   
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)
   
    return agent
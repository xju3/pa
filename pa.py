import streamlit as st
from utils import (
    create_email_chain,
    create_study_plan_chain,
    create_knowledge_qna_chain,
    create_action_items_chain,
    initialize_agent_executor,
    get_llm_instance
)

llm = get_llm_instance()

email_chain = create_email_chain(llm)
study_plan_chain = create_study_plan_chain(llm)
knowledge_qna_chain = create_knowledge_qna_chain(llm)
action_items_chain = create_action_items_chain(llm)
agent_executor = initialize_agent_executor()


st.title("个人助理")

task_type = st.sidebar.selectbox("选择一项任务", [
    "起草邮件", "Q&A",
    "生成学习计划", "获取行动内容",
])

if task_type == "起草邮件":
    st.header("基于Context起草一封邮件")
    context_input = st.text_area("请输出邮件内容:", value="明天去办理身份证，向公司请假一天")
    if st.button("起草邮件"):
        result = email_chain.run(context=context_input)
        st.text_area("Generated Email", result, height=300)

elif task_type == "Q&A":
    st.header("知识问答")
    domain_input = st.text_input("请输入知识领域 (e.g.,财经类,技术类,健康类):", value='财经类')
    question_input = st.text_area("输入你的问题:", value='什么是ETF')
    if st.button("Get Answer"):
        result = knowledge_qna_chain.run(question=question_input, domain=domain_input)
        st.text_area("Answer", result, height=300)

elif task_type == "生成学习计划":
    st.header("生成个人学习计划")
    topic_input = st.text_input("学习内容", value='Java编程语言')
    duration_input = st.text_input("输出学习周期 (如, 2周, 1个月):", value="一个月")
    if st.button("Generate Study Plan"):
        result = study_plan_chain.run(topic=topic_input, duration=duration_input)
        st.text_area("Study Plan", result, height=300)

elif task_type == "获取行动内容":
    st.header("多会议记录或邮件中获取行动内容")
    notes_input = st.text_area("请输入邮件或会议内容:", value="下周,我们要完成本次的所有开发内容")
    if st.button("执行"):
        result = action_items_chain.run(notes=notes_input)·
        st.text_area("行动内容", result, height=300)
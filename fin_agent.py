import re
from typing import Sequence, Union

import pandas as pd
from langchain.agents import AgentExecutor, AgentOutputParser
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool, Tool

import config
from SQL_retrieve_chain import sql_retrieve_chain
from util.instances import LLM
from pdf_retrieve_chain import pdf_retrieve_chain
from util import prompts


def create_react_my_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: BasePromptTemplate
) -> Runnable:
    # noqa: E501
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    # 读取公司名称
    df = pd.read_csv(config.company_save_path)
    company_list = df['company']
    company_content = ''
    for company in company_list:
        company_content = company_content + "\n" + company

    # print(company_content)

    prompt = prompt.partial(
        tools=render_text_description(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
        company=company_content
    )
    llm_with_stop = llm.bind(stop=["\n观察"])
    temp_agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | MyReActSingleInputOutputParser()
    )
    return temp_agent


class MyReActSingleInputOutputParser(AgentOutputParser):

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:

        FINAL_ANSWER_ACTION = "Final Answer:"
        FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
            "Parsing LLM output produced both a final answer and a parse-able action:"
        )
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        else:
            return AgentFinish(
                {"output": text}, text
            )

    @property
    def _type(self) -> str:
        return "react-single-input"


auto_tools = [
    Tool(
        name="招股说明书",
        func=pdf_retrieve_chain,
        description="招股说明书检索",
        ),
    Tool(
        name="查询数据库",
        func=sql_retrieve_chain,
        description="查询数据库检索结果",
        ),
]
tmp_prompt = ChatPromptTemplate.from_template(prompts.AGENT_CLASSIFY_PROMPT_TEMPLATE)
agent = create_react_my_agent(LLM, auto_tools, prompt=tmp_prompt)

agent_executor = AgentExecutor(
        agent=agent, tools=auto_tools, verbose=True
    )
result = agent_executor.invoke({"question": "报告期内，华瑞电器股份有限公司人工成本占主营业务成本的比例分别为多少？"})
# result = agent_executor.invoke({"question": "请帮我计算，在20210105，中信行业分类划分的一级行业为综合金融行业中，涨跌幅最大股票的股票代码是？涨跌幅是多少？百分数保留两位小数。股票涨跌幅定义为：（收盘价 - 前一日收盘价 / 前一日收盘价）* 100%。"})
print(result["output"])

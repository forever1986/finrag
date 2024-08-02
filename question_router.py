import csv

from util import prompts
import config
import pandas as pd
from jsonlines import jsonlines
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 读取question

questions = []
with jsonlines.open(config.question_path, "r") as json_file:
    for obj in json_file.iter(type=dict, skip_invalid=True):
        questions.append(obj)

# 初始化模型
llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key="05f9fe2491eef6c3a7c2659957880e86.V7LJzSgrEK7raanZ",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 读取公司名称
df = pd.read_csv(config.company_save_path)
company_list = df['company']
company_content = ''
for company in company_list:
    company_content = company_content + "\n" + company

# 结果文件
answer_file = open(config.question_classify_path, 'w', newline='', encoding='utf-8-sig')
csvwriter = csv.writer(answer_file)
csvwriter.writerow(['问题id', '问题', '分类'])

# 获取分类
for cyc in range(100):
    question = questions[cyc]['question']
    temp_question = question.replace(' ', '')
    prompt = ChatPromptTemplate.from_template(prompts.CLASSIFY_PROMPT_TEMPLATE)
    # print(prompt.invoke({"company": company_content, "question": temp_question}))
    chain = prompt | llm
    response = chain.invoke({"company": company_content, "question": temp_question})
    print(response.content)
    csvwriter.writerow([str(questions[cyc]['id']),
                        str(questions[cyc]['question']),
                        response.content])



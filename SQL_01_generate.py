import csv
import re
import copy
import config
import pandas as pd

from util.instances import TOKENIZER, LLM
from util import prompts
from langchain_core.prompts import ChatPromptTemplate


def generate_sql(question, llm, example_question_list, example_sql_list, tmp_example_token_list, example_num=5):
    pattern1 = r'\d{8}'  # 过滤掉一些数字的正则表达式
    sql_pattern_start = '```sql'
    sql_pattern_end = '```'
    temp_question = question
    # 提取数字
    date_list = re.findall(pattern1, temp_question)
    temp_question2_for_search = temp_question
    # 将数字都替换为空格
    for t_date in date_list:
        temp_question2_for_search.replace(t_date, ' ')
    temp_tokens = TOKENIZER(temp_question2_for_search)
    temp_tokens = temp_tokens['input_ids']
    # 计算与已有问题的相似度--使用Jaccard进行相似度计算
    similarity_list = list()
    for cyc2 in range(len(tmp_example_token_list)):
        similarity_list.append(len(set(temp_tokens) & set(tmp_example_token_list[cyc2]))
                               / (len(set(temp_tokens)) + len(set(tmp_example_token_list[cyc2]))))

    # 求与第X个问题相似的问题
    t = copy.deepcopy(similarity_list)
    # 求m个最大的数值及其索引
    max_index = []
    for _ in range(example_num):
        number = max(t)
        index = t.index(number)
        t[index] = 0
        max_index.append(index)

    # 防止提示语过长
    temp_length_test = ""
    short_index_list = list()  # 匹配到的问题下标
    for index in max_index:
        temp_length_test = temp_length_test + example_question_list[index]
        temp_length_test = temp_length_test + example_sql_list[index]
        if len(temp_length_test) > 2000:
            break
        short_index_list.append(index)

    # print("找到相似的模板：", short_index_list)
    # 组装prompt
    prompt = ChatPromptTemplate.from_template(prompts.GENERATE_SQL_TEMPLATE)
    examples = ''
    for index in short_index_list:
        examples = examples + "问题：" + example_question_list[index] + '\n'
        examples = examples + "SQL：" + example_sql_list[index] + '\n'

    chain = prompt | llm
    response = chain.invoke({"examples": examples, "table_info": prompts.TABLE_INFO, "question": temp_question})
    # print("问题：", temp_question)
    # print("SQL：", response.content)
    sql = response.content
    start_index = sql.find(sql_pattern_start) + len(sql_pattern_start)
    end_index = -1
    if start_index >= 0:
        end_index = sql[start_index:].find(sql_pattern_end) + start_index
    if start_index < end_index:
        sql = sql[start_index:end_index]
        return prompt.invoke({"examples": examples, "table_info": prompts.TABLE_INFO, "question": temp_question}), sql
    else:
        print("generate sql error:", temp_question)
        return "error", "error"


if __name__ == '__main__':

    # 第一步：读取问题和SQL模板，使用tokenizer进行token化
    sql_examples_file = pd.read_csv(config.sql_examples_path, delimiter=",", header=0)
    g_example_question_list = list()
    g_example_sql_list = list()
    g_example_token_list = list()
    for cyc in range(len(sql_examples_file)):
        g_example_question_list.append(sql_examples_file[cyc:cyc + 1]['问题'][cyc])
        g_example_sql_list.append(sql_examples_file[cyc:cyc + 1]['SQL'][cyc])
        tokens = TOKENIZER(sql_examples_file[cyc:cyc + 1]['问题'][cyc])
        tokens = tokens['input_ids']
        g_example_token_list.append(tokens)

    # 第二步：测试问题及结果文件
    question_csv_file = pd.read_csv(config.question_classify_path, delimiter=",", header=0)

    question_sql_file = open(config.question_sql_path, 'w', newline='', encoding='utf-8-sig')
    csvwriter = csv.writer(question_sql_file)
    csvwriter.writerow(['问题id', '问题', 'SQL', 'prompt'])

    # 第三步：循环问题，使用Jaccard进行相似度计算问题与模板中的问题相似度最高的几条记录
    for cyc in range(len(question_csv_file)):
        if question_csv_file['分类'][cyc] == '查询数据库':
            result_prompt, result = generate_sql(question_csv_file['问题'][cyc], LLM, g_example_question_list,
                                                 g_example_sql_list, g_example_token_list)
            csvwriter.writerow([str(question_csv_file[cyc:(cyc + 1)]['问题id'][cyc]),
                                str(question_csv_file[cyc:(cyc + 1)]['问题'][cyc]),
                                result, result_prompt])
        else:
            print("pass question:", question_csv_file['问题'][cyc])
            pass

import csv
import re
import copy
import config
import pandas as pd

from util.instances import LLM, TOKENIZER
from util import prompts
from langchain_core.prompts import ChatPromptTemplate


def generate_answer(question, fa, llm, example_question_list, example_info_list, example_fa_list,
                    tmp_example_token_list, example_num=5):
    pattern1 = r'\d{8}'  # 过滤掉一些数字的正则表达式
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
        temp_length_test = temp_length_test + example_fa_list[index]
        if len(temp_length_test) > 2000:
            break
        short_index_list.append(index)

    # print("找到相似的模板：", short_index_list)
    # 组装prompt
    prompt = ChatPromptTemplate.from_template(prompts.ANSWER_TEMPLATE)
    examples = ''
    for index in short_index_list:
        examples = examples + "问题：" + example_question_list[index] + '\n'
        examples = examples + "资料：" + example_info_list[index] + '\n'
        examples = examples + "答案：" + example_fa_list[index] + '\n'

    chain = prompt | llm
    response = chain.invoke({"examples": examples, "FA": fa, "question": temp_question})
    # print("答案：", response.content)
    return response.content


if __name__ == '__main__':

    # 第一步：读取问题和FA模板，使用tokenizer进行token化
    sql_examples_file = pd.read_csv(config.sql_examples_path, delimiter=",", header=0)
    g_example_question_list = list()
    g_example_info_list = list()
    g_example_fa_list = list()
    g_example_token_list = list()
    for cyc in range(len(sql_examples_file)):
        g_example_question_list.append(sql_examples_file[cyc:cyc + 1]['问题'][cyc])
        g_example_info_list.append(sql_examples_file[cyc:cyc + 1]['资料'][cyc])
        g_example_fa_list.append(sql_examples_file[cyc:cyc + 1]['FA'][cyc])
        tokens = TOKENIZER(sql_examples_file[cyc:cyc + 1]['问题'][cyc])
        tokens = tokens['input_ids']
        g_example_token_list.append(tokens)

    # 第二步：拿到答案
    result_csv_file = pd.read_csv(config.question_sql_check_path, delimiter=",", header=0)

    answer_file = open(config.answer_path, 'w', newline='', encoding='utf-8-sig')
    csvwriter = csv.writer(answer_file)
    csvwriter.writerow(['问题id', '问题', '资料', 'FA'])

    # 第三步：循环问题，使用Jaccard进行相似度计算问题与模板中的问题相似度最高的几条记录
    for cyc in range(len(result_csv_file)):
        if result_csv_file['flag'][cyc] == 1:
            result = generate_answer(result_csv_file['问题'][cyc], result_csv_file['执行结果'][cyc], LLM,
                                     g_example_question_list, g_example_info_list, g_example_fa_list,
                                     g_example_token_list)
            csvwriter.writerow([str(result_csv_file[cyc:(cyc + 1)]['问题id'][cyc]),
                                str(result_csv_file[cyc:(cyc + 1)]['问题'][cyc]),
                                str(result_csv_file[cyc:(cyc + 1)]['执行结果'][cyc]),
                                result])

import sqlite3

import pandas as pd

import config
from SQL_01_generate import generate_sql
from SQL_02_query import query_db
from SQL_03_answer_from_SQL import generate_answer
from util.instances import TOKENIZER, LLM

g_example_question_list = list()
g_example_sql_list = list()
g_example_fa_list = list()
g_example_info_list = list()
g_example_token_list = list()


def sql_retrieve_chain(query):
    if len(g_example_question_list) <= 0:
        sql_examples_file = pd.read_csv(config.sql_examples_path, delimiter=",", header=0)
        for cyc in range(len(sql_examples_file)):
            g_example_question_list.append(sql_examples_file[cyc:cyc + 1]['问题'][cyc])
            g_example_sql_list.append(sql_examples_file[cyc:cyc + 1]['SQL'][cyc])
            g_example_info_list.append(sql_examples_file[cyc:cyc + 1]['资料'][cyc])
            g_example_fa_list.append(sql_examples_file[cyc:cyc + 1]['FA'][cyc])
            tokens = TOKENIZER(sql_examples_file[cyc:cyc + 1]['问题'][cyc])
            tokens = tokens['input_ids']
            g_example_token_list.append(tokens)
    result_prompt, sql = generate_sql(query, LLM, g_example_question_list, g_example_sql_list, g_example_token_list)
    conn = sqlite3.connect('/root/autodl-tmp/bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db')
    cs = conn.cursor()
    success_flag, exc_result = query_db(sql, cs)
    conn.close()
    answer = generate_answer(query, exc_result, LLM, g_example_question_list, g_example_info_list, g_example_fa_list,
                             g_example_token_list)
    return answer

import csv
import sqlite3

import pandas as pd

import config


def query_db(sql, tmp_cs):
    temp_sql = sql
    try:
        tmp_cs.execute(temp_sql)
        result_cols = tmp_cs.fetchall()
        exc_result = str(result_cols)
        success_flag = 1
    except:
        exc_result = "error"
        success_flag = 0
        print("")
    return success_flag, exc_result


if __name__ == '__main__':
    # 打开生成的sql文件
    question_sql_file = pd.read_csv(config.question_sql_path, delimiter=",", header=0)

    # 打开sql执行结果文件
    file = open(config.question_sql_check_path, 'w', newline='', encoding='utf-8-sig')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['问题id', '问题', 'SQL', 'flag', '执行结果'])

    # 数据库连接
    conn = sqlite3.connect('/root/autodl-tmp/bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db')
    cs = conn.cursor()

    # 执行SQL 并返回结果写入sql执行结果文件
    for cyc in range(len(question_sql_file)):
        if question_sql_file['SQL'][cyc] != 'error':
            success_flag, exc_result = query_db(question_sql_file['SQL'][cyc], cs)
            csvwriter.writerow([str(question_sql_file[cyc:(cyc + 1)]['问题id'][cyc]),
                                str(question_sql_file[cyc:(cyc + 1)]['问题'][cyc]),
                                question_sql_file['SQL'][cyc],
                                success_flag,
                                exc_result])
        else:
            pass

    file.close()
    conn.close()




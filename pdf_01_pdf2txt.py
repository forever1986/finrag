import re
import pdfplumber


# 通过表格的top和bottom来读取页面的文章，通过3种情况
# 1） 第一种情况：top和bottom为空，则代表纯文本
# 2） 第二种情况，top为空，bottom不为空，则代表处理最后一个表格下面的文本
# 3） 第三种情况，top和bottom不为空，则代表处理表格上面的文本
def check_lines(page, top, bottom):
    try:
        # 获取文本框
        lines = page.extract_words()
    except Exception as e:
        print(f'页码: {page.page_number}, 抽取文本异常，异常信息: {e}')
        return ''
    # empty util
    check_re = '(?:。|；|单位：元|单位：万元|币种：人民币)$'
    page_top_re = '(招股意向书(?:全文)?(?:（修订版）|（修订稿）|（更正后）)?)'

    text = ''
    last_top = 0
    last_check = 0
    if top == '' and bottom == '':
        if len(lines) == 0:
            print(f'{page.page_number}页无数据, 请检查！')
            return ''
    for l in range(len(lines)):
        each_line = lines[l]
        # 第一种情况：top和bottom为空，则代表纯文本
        if top == '' and bottom == '':
            if abs(last_top - each_line['top']) <= 2:
                text = text + each_line['text']
            elif last_check > 0 and (page.height * 0.9 - each_line['top']) > 0 and not re.search(check_re, text):
                if '\n' not in text and re.search(page_top_re, text):
                    text = text + '\n' + each_line['text']
                else:
                    text = text + each_line['text']
            else:
                if text == '':
                    text = each_line['text']
                else:
                    text = text + '\n' + each_line['text']
        # 第二种情况，top为空，bottom不为空，则代表处理最后一个表格下面的文本
        elif top == '':
            if each_line['top'] > bottom:
                if abs(last_top - each_line['top']) <= 2:
                    text = text + each_line['text']
                elif last_check > 0 and (page.height * 0.85 - each_line['top']) > 0 and not re.search(check_re, text):
                    if '\n' not in text and re.search(page_top_re, text):
                        text = text + '\n' + each_line['text']
                    else:
                        text = text + each_line['text']
                else:
                    if text == '':
                        text = each_line['text']
                    else:
                        text = text + '\n' + each_line['text']
        # 第三种情况，top和bottom不为空，则代表处理表格上面的文本
        else:
            if top > each_line['top'] > bottom:
                if abs(last_top - each_line['top']) <= 2:
                    text = text + each_line['text']
                elif last_check > 0 and (page.height * 0.85 - each_line['top']) > 0 and not re.search(check_re, text):
                    if '\n' not in text and re.search(page_top_re, text):
                        text = text + '\n' + each_line['text']
                    else:
                        text = text + each_line['text']
                else:
                    if text == '':
                        text = each_line['text']
                    else:
                        text = text + '\n' + each_line['text']
        last_top = each_line['top']
        last_check = each_line['x1'] - page.width * 0.83

    return text


# 删除没有数据的列
def drop_empty_cols(data):
    # 删除所有列为空数据的列
    transposed_data = list(map(list, zip(*data)))
    filtered_data = [col for col in transposed_data if not all(cell == '' for cell in col)]
    result = list(map(list, zip(*filtered_data)))
    return result


# 通过判断页面是否有表格
# 1） 如果没有表格，则按照读取文本处理
# 2） 如果有表格，则获取每个表格的top坐标和bottom坐标，按照表格顺序，先读取表格之上的文字，在使用markdown读取表格
# 3） 不断循环2），等到最后一个表格，只需要读取表格之下的文字即可
def extract_text_and_tables(page):
    all_text = ""
    bottom = 0
    try:
        tables = page.find_tables()
    except:
        tables = []
    if len(tables) >= 1:
        count = len(tables)
        for table in tables:
            # 判断表格底部坐标是否小于0
            if table.bbox[3] < bottom:
                pass
            else:
                count -= 1
                # 获取表格顶部坐标
                top = table.bbox[1]
                text = check_lines(page, top, bottom)
                text_list = text.split('\n')
                for _t in range(len(text_list)):
                    all_text += text_list[_t] + '\n'

                bottom = table.bbox[3]
                new_table = table.extract()
                r_count = 0
                for r in range(len(new_table)):
                    row = new_table[r]
                    if row[0] is None:
                        r_count += 1
                        for c in range(len(row)):
                            if row[c] is not None and row[c] not in ['', ' ']:
                                if new_table[r - r_count][c] is None:
                                    new_table[r - r_count][c] = row[c]
                                else:
                                    new_table[r - r_count][c] += row[c]
                                new_table[r][c] = None
                    else:
                        r_count = 0

                end_table = []
                for row in new_table:
                    if row[0] is not None:
                        cell_list = []
                        cell_check = False
                        for cell in row:
                            if cell is not None:
                                cell = cell.replace('\n', '')
                            else:
                                cell = ''
                            if cell != '':
                                cell_check = True
                            cell_list.append(cell)
                        if cell_check:
                            end_table.append(cell_list)
                end_table = drop_empty_cols(end_table)

                markdown_table = ''  # 存储当前表格的Markdown表示
                for i, row in enumerate(end_table):
                    # 移除空列，这里假设空列完全为空，根据实际情况调整
                    row = [cell for cell in row if cell is not None and cell != '']
                    # 转换每个单元格内容为字符串，并用竖线分隔
                    processed_row = [str(cell).strip() if cell is not None else "" for cell in row]
                    markdown_row = '| ' + ' | '.join(processed_row) + ' |\n'
                    markdown_table += markdown_row
                    # 对于表头下的第一行，添加分隔线
                    if i == 0:
                        separators = [':---' if cell.isdigit() else '---' for cell in row]
                        markdown_table += '| ' + ' | '.join(separators) + ' |\n'
                all_text += markdown_table + '\n'

                if count == 0:
                    text = check_lines(page, '', bottom)
                    text_list = text.split('\n')
                    for _t in range(len(text_list)):
                        all_text += text_list[_t] + '\n'

    else:
        text = check_lines(page, '', '')
        text_list = text.split('\n')
        for _t in range(len(text_list)):
            all_text += text_list[_t] + '\n'

    return all_text


def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for i, page in enumerate(pdf.pages):
            all_text += extract_text_and_tables(page)

    return all_text


if __name__ == '__main__':
    # 使用示例
    test_pdf_path = "data/pdf/03c625c108ac0137f413dfd4136adb55c74b3805.pdf"
    extracted_text = extract_text(test_pdf_path)

    pdf_save_path = "data/pdf_txt_file/安徽黄山胶囊股份有限公司.txt"
    with open(pdf_save_path, 'w', encoding='utf-8') as file:
        file.write(extracted_text)

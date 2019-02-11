import pandas as pd
from sklearn.utils import shuffle
import re
import jieba

def print_sep():
    print('------------------------------------------')

def filter_out_classes(file, level=1):
    """
    输入原始数据,为所有类别进行编号，并返回相应的字典
    :param file:原始数据文件名
    :param level:类别级数
    :return:    key:类别名   res_dict[key]:类别编号
    """

    code = 'gb18030'
    end = None

    data = pd.read_csv(file,
                       sep='\t',
                       encoding=code,
                       nrows=end
                       )

    # 只获取类别列
    type_data = data.loc[:, 'TYPE']

    res_dict = {}
    index = 0

    for i in range(len(type_data)):
        classes = type_data[i].split("--")
        cls_name = ''

        # 取出对应级别的类别名
        for j in range(level):
            cls_name += classes[j]
            # j+1 == currentLevel
            if j + 1 < len(classes):
                cls_name += '--'

        # 放入字典
        if not cls_name in res_dict.keys():
            res_dict[cls_name] = index
            index += 1

    return res_dict


def export_classes(cls_dic, output_file):
    """
    将类别字典中的内容导出到文件中
    :param cls_dic: 类别字典
    :param output_file: 导出文件路径
    """
    of = open(output_file, "w", encoding="utf-8")
    for className in cls_dic.keys():
        of.write(className + " " + str(cls_dic[className]) + '\n')
    of.close()


def get_classes(classes_file):
    """
    获得类别文件中的类别信息
    :param classes_file: 类别文件路径
    :return: 类别字典
    """
    dic = {}

    with open(classes_file, 'r', encoding="utf-8") as clsFile:
        for line in clsFile:
            t = line.split(' ')
            dic[t[0]] = t[1].replace('\n', '')  # 去除行尾的换行符
    return dic


def transfer_to_ft_format(file_path, output_path, class_path):
    """
    将数据转换为fastText的指定格式
    :param file_path: 数据文件的路径
    :param output_path: 导出文件的目录路径
    :param class_path: 类别文件的路径
    """
    code = 'gb18030'
    end = None
    dic = get_classes(class_path)
    data = pd.read_csv(file_path,
                       sep='\t',
                       encoding=code,
                       nrows=end
                       )

    df = shuffle(data) # 打乱顺序

    print_sep()
    print('正在转换类别编码....')
    count = 0
    keys = dic.keys()
    klen = len(keys)
    for key in keys:
        df.replace(regex=re.compile('^' + key + '[\s\S]*'), value='_label_' + str(dic[key]), inplace=True)
        count += 1
        print('已转化:%.2f%%' % ((count/klen) * 100))

    print_sep()
    print('正在分词....')
    for index, row in df.iterrows():
        name = row['ITEM_NAME']
        row['ITEM_NAME'] = " ".join(jieba.cut(name, cut_all=True))
    print('分词完成')
    print_sep()

    # 重新对打乱后的dataFrame 进行编号
    df.index = range(len(df))
    traindf = df.loc[:400000, :]
    testdf = df.loc[400001:500000, :]

    traindf.to_csv(output_path + '/test.txt', sep='\t', index=False, encoding="utf-8", header=0)
    testdf.to_csv(output_path + '/train.txt', sep='\t', index=False, encoding="utf-8", header=0)


# export_classes(filter_out_classes("./data/train.tsv", 3), './data/classes.txt')
# print(get_classes('./data/classes.txt'))

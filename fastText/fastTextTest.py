# _*_coding:utf-8 _*_
import formatUtil as fu
import fasttext

# fu.export_classes(fu.filter_out_classes("./data/train.tsv", 3), './data/classes.txt')
# fu.transfer_to_ft_format('./data/train.tsv', './data', './data/classes.txt')

#训练模型
print('--------------------')
print('训练模型中....')
classifier = fasttext.supervised("./data/train.txt", "./model/fasttext.model", label_prefix="_label_")
print('训练完毕')
print('--------------------')
#load训练好的模型
#classifier = fasttext.load_model('./model/fasttext.model.bin', label_prefix='_label_')

result = classifier.test("./data/test.txt")
print(result.precision)
print(result.recall)

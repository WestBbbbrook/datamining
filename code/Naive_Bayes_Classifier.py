#coding: utf-8
import os
import time
import random
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
#import pylab as pl
import matplotlib.pyplot as plt


def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set

def TextProcessing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder) #拼接路径
        files = os.listdir(new_folder_path)
        # 类内循环
        j = 1
        for file in files:
            if j > 1000: # 每类text样本数最多1000
                break
            with open(os.path.join(new_folder_path, file), 'r') as fp:
               raw = fp.read()
               word_list = list(raw.split('\n'))
               data_list.append(word_list)
               class_list.append(folder.decode('utf-8'))
               j += 1
            # print raw
            ## --------------------------------------------------------------------------------
            ## jieba分词
             # genertor转化为list，每个词unicode格式
            # jieba.disable_parallel() # 关闭并行分词模式
            # print word_list
            ## --------------------------------------------------------------------------------


    ## 划分训练集和测试集
    # train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)
    data_class_list = zip(data_list, class_list)
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size)+1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if all_words_dict.has_key(word):
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list)[0])

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    #n = 1
    for t in range(deleteN, len(all_words_list), 1):
        #if n > 1000: # feature_words的维度1000
         #   break
        # print all_words_list[t]
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
        #    n += 1
        #print "选取特征词"
        #print feature_words
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list,flag='sklearn'):
    ## -----------------------------------------------------------------------------------
    if flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        # print classifier.predict(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.predict(test_feature)[0],
        # print ''
        test_accuracy = classifier.score(test_feature_list, test_class_list)
    else:
        test_accuracy = []
    return test_accuracy


def select_word(folder_path,folder_all_word,test_size):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)  # 拼接路径
        files = os.listdir(new_folder_path)
        # 类内循环
        j = 1
        for file in files:
            if j > 1000:  # 每类text样本数最多1000
                break
            with open(os.path.join(new_folder_path, file), 'r') as fp:
                raw = fp.read()
                word_list = list(raw.split('\n'))
                data_list.append(word_list)
                class_list.append(folder.decode('utf-8'))
                j += 1
    ## 划分训练集和测试集
    # train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)
    data_class_list = zip(data_list, class_list)
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    with open(folder_all_word, 'r') as f:
        raw = f.read()
        all_words_list = list(raw.split())

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

if __name__ == '__main__':

    print "start"

    ## 文本预处理
    folder_path = r'E:\新建文件夹\文本\新建文件夹\python\DataMinning\newsplit'.decode('utf-8')
    #all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    folder_all_word = r'E:\新建文件夹\文本\新建文件夹\python\DataMinning\tfidf\dict.txt'.decode('utf-8')
    feature_words, train_data_list, test_data_list, train_class_list, test_class_list = select_word(folder_path,folder_all_word,test_size=0.2)
    # 生成stopwords_set
    #stopwords_file = r'C:\Users\Y-GH\PycharmProjects\document-classification\Database\stopwords_cn.txt'
    #stopwords_set = MakeWordsSet(stopwords_file)

    ## 文本特征提取和分类
    flag = 'sklearn'
    deleteNs = range(0, 1000, 20)
    test_accuracy_list = []
    for deleteN in deleteNs:
       # feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
        test_accuracy_list.append(test_accuracy)
    print test_accuracy_list

    # 结果评价
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.savefig('result.png')

    print "finished"
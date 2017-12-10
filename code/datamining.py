# -*- coding: utf-8 -*-
import jieba
import jieba.posseg as pseg
import os,re,collections
import sys
import numpy as np
from numpy import nan as Na
import pandas as pd
from pandas import Series,DataFrame
sys.setrecursionlimit(999999999)#增加递归次数
stopwords = {}.fromkeys([ line.rstrip() for line in open('stopwords.txt','r',encoding='utf-8')])
# print(stopwords)
#stopwords = {}.fromkeys(['时代', '新机遇','机遇','意识','人'])
#初始每类的路径列表
newsPathList=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\news\\caijing",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\news\\guoji",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\news\\IT",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\news\\junshi",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\news\\nengyuan",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\news\\qiche",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\news\\tiyu",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\news\\wenhua",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\news\\yule",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\news\\jiankang"]
#分词后的每类的路径列表
splitPathList=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\split\\caijing",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\split\\guoji",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\split\\IT",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\split\\junshi",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\split\\nengyuan",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\split\\qiche",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\split\\tiyu",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\split\\wenhua",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\split\\yule",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\split\\jiankang"]
#原始的10类分别的dict路径列表
oriDictPathList=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\caijing.txt",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\guoji.txt",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\IT.txt",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\junshi.txt",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\nengyuan.txt",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\qiche.txt",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\tiyu.txt",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\wenhua.txt",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\yule.txt",
           "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\jiankang.txt"]
#tfidf
allwordpath="E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\allword.txt"
wordfrepath=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\wordfrequency\\caijing.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\wordfrequency\\guoji.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\wordfrequency\\IT.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\wordfrequency\\junshi.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\wordfrequency\\nengyuan.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\wordfrequency\\qiche.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\wordfrequency\\tiyu.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\wordfrequency\\wenhua.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\wordfrequency\\yule.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\wordfrequency\\jiankang.txt"]
tfidfpath=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\tfidf\\caijing.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\tfidf\\guoji.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\tfidf\\IT.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\tfidf\\junshi.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\tfidf\\nengyuan.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\tfidf\\qiche.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\tfidf\\tiyu.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\tfidf\\wenhua.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\tfidf\\yule.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\tfidf\\jiankang.txt"]
newdictpath=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newdict\\caijing.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newdict\\guoji.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newdict\\IT.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newdict\\junshi.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newdict\\nengyuan.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newdict\\qiche.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newdict\\tiyu.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newdict\\wenhua.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newdict\\yule.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newdict\\jiankang.txt"]
newtfidfpath=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newtfidf\\caijing.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newtfidf\\guoji.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newtfidf\\IT.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newtfidf\\junshi.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newtfidf\\nengyuan.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newtfidf\\qiche.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newtfidf\\tiyu.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newtfidf\\wenhua.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newtfidf\\yule.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\newtfidf\\jiankang.txt"]
dictpath="E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\dict.txt"
inputdatapath=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\inputdata\\caijing.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\inputdata\\guoji.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\inputdata\\IT.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\inputdata\\junshi.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\inputdata\\nengyuan.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\inputdata\\qiche.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\inputdata\\tiyu.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\inputdata\\wenhua.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\inputdata\\yule.txt",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\inputdata\\jiankang.txt"]
splitwordfrePathList=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\splitwordfre\\caijing",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\splitwordfre\\guoji",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\splitwordfre\\IT",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\splitwordfre\\junshi",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\splitwordfre\\nengyuan",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\splitwordfre\\qiche",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\splitwordfre\\tiyu",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\splitwordfre\\wenhua",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\splitwordfre\\yule",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\splitwordfre\\jiankang"]
newSplitPathList=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\newsplit\\caijing",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\newsplit\\guoji",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\newsplit\\IT",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\newsplit\\junshi",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\newsplit\\nengyuan",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\newsplit\\qiche",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\newsplit\\tiyu",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\newsplit\\wenhua",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\newsplit\\yule",
             "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\newsplit\\jiankang"]

#遍历txt文件，分词、取名次、去停用词
#filepath为文件夹路径list,i是第几类
#会一层一层遍历到最内层的txt读取新闻并进行分词
def gci(filepath,i):
#遍历filepath下所有文件，包括子目录
   files = os.listdir(filepath)
   for fi in files:
       path = os.path.join(filepath,fi)
       if os.path.isdir(path):
           gci(path)
       else:
           sliptword(path,i)
#分词、取名次、去停用词
#path是文本路径，i是类
#oriDictPathList[i]是第i类的字典
def  sliptword(path,i):
    print(path)
    # news为我原来放新闻的文件夹名字，用正则表达式把它替换成split，换个文件夹保存
    strinfo = re.compile('news')
    writepath=strinfo.sub('split',path) #分词结果写入的路径split
    with open(path, "r",encoding='utf-8') as f:
        text = f.read()
    #print(text)
    str = ""
    str2=""
    result = pseg.cut(text)  ##词性标注，标注句子分词后每个词的词性
    for w in result:#遍历每个词语
        #print(w.word, "/", w.flag, ", ", end=" ")
        if w.flag.startswith('n'):#如果词性是n开头的说明是名次，留下
            #print(w.word, "/", w.flag)
            if w.word not in stopwords:#如果不在停用次表里，留下保存在字符串里
                # with open(writepath, "a") as f:
                #     #f.write(w.word+"/"+w.flag+"\n")
                #     f.write(w.word + "\n")
                str = str + w.word+"\n"
                str2 =str2+w.word+" "
    with open(writepath,"a")as f:
        f.write(str)
    with open(oriDictPathList[i], "a")as f:
        f.write(str2)
#生成字典，每个类别生成一个
# def createdict():
#    index=1
#    #遍历filepath下所有文件，包括子目录
#    filepath="E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\split\\caijing"
#    files = os.listdir(filepath)
#    text = ""
#    for fi in files:
#        path = os.path.join(filepath,fi)
#        if os.path.isdir(path):
#            gci(path)
#        else:
#            with open(path,'r')as fr:
#                text = text + fr.read().replace('\n'," ")
#            print(index)
#            if index%100==0:
#                 with open("E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\caijing.txt", 'a', encoding='utf-8')as fo:
#                     fo.write(text)
#                     text=""
#            index = index + 1
# def createdict(filepath):
#      # 遍历filepath下所有文件，包括子目录
#     files = os.listdir(filepath)
#     num = 0
#     for fi in files:
#         path = os.path.join(filepath, fi)
#         #最底层num.txt路径列表
#         files2 = os.listdir(path)
#         dict = collections.Counter([])
#         for fi2 in files2:
#             # with open(fi2, "r") as f:
#             #     text = f.read()
#             path2 = os.path.join(path, fi2)
#             dict = dict + collections.Counter([line.rstrip() for line in open(path2)])
#             print(len(dict),"\n")
#             print(dict)
#         writepath = ["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\dict\\体育.txt",
#                      "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\dict\\军事.txt",
#                      "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\dict\\汽车.txt",
#                      "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\dict\\财经.txt",
#                      ]
#         with open(writepath[num],"a") as f:
#             for k,v in dict.items():
#                  f.write(str(k) + ":"+str(v)+"\n")
#         num = num+1

#计算每类的tfidf
# oriDictPathList=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\caijing.txt",
#            "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\guoji.txt",
#            "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\IT.txt",
#            "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\junshi.txt",
#            "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\nengyuan.txt",
#            "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\qiche.txt",
#            "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\tiyu.txt",
#            "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\wenhua.txt",
#            "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\yule.txt",
#            "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\oridict\\jiankang.txt"]
#计算tfdif
#oriDictPathList为原始的字典list，存放每类的所有词语，不去重，维度很高
def tfidf(oriDictPathList):
    import sklearn
    from sklearn.feature_extraction.text import CountVectorizer
    # 语料
    corpus=[]#存放每类的字典
    for i in range(len(oriDictPathList)):
        print(i,"\n")
        with open(oriDictPathList[i],'r',encoding='utf-8')as f:
            corpus.append(f.read())
            print(len(corpus[i]),"\n")
    # corpus = [
    #     'This is the first document.',
    #     'This is the second second document.',
    #     'And the third one.',
    #     'Is this the first document?',
    # ]
    # 将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer()
    # 计算个词语出现的次数
    X = vectorizer.fit_transform(corpus)
    # 获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()
    #print("word:",word)
    #把10类中出现的所有词语存放到allword.txt里
    with open("E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\tfidf\\allword.txt",'w',encoding='utf-8') as f:
        print(len(word))
        s='\n'.join(word)
        print(len(s))
        f.write('\n'.join(word))
    # 查看词频结果
    # print("X.toarray():",X.toarray())
    #np.set_printoptions(threshold='nan')
    np.set_printoptions(threshold=np.inf)
    # print(X.toarray()[0])
    print(1)
    #wordfrepath为存放每类词频文件的list
    #每个文件写如本类的词频，此时的词频维度是多类的总维度
    for i in range(len(wordfrepath)):
        print("i:",i)
        s = str(X.toarray()[i])
        s = s.lstrip('[')
        s = s.rstrip(']')
        with open(wordfrepath[i], 'w', encoding='utf-8')as f:
            f.write(s)
    from sklearn.feature_extraction.text import TfidfTransformer
    # 类调用
    transformer = TfidfTransformer()
    print("transformer:",transformer)
    # 将词频矩阵X统计成TF-IDF值
    tfidf = transformer.fit_transform(X)
    # 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
    # print("tfidf.toarray()",tfidf.toarray())
    np.set_printoptions(threshold=np.inf)#加上这句可以全输出或者全写入，不然中间是省略号
    print(2)
    #tfidfpath为存放每类的tfidf数值的list
    for i in range(len(tfidfpath)):
        print(i)
        s = str(tfidf.toarray()[i])
        s = s.lstrip('[')
        s = s.rstrip(']')
        with open(tfidfpath[i], 'w', encoding='utf-8')as f:
            f.write(s)
#快排，因为要挑选出每类tfidf数值最大的若干个词语，同时还要把对应词语也筛选出来，所以手写快排用它的索引
def parttion(v1,v2, left, right):
    key1 = v1[left]
    key2 = v2[left]
    low = left
    high = right
    while low < high:
        while (low < high) and (v1[high] <= key1):
            high -= 1
        v1[low] = v1[high]
        v2[low] = v2[high]
        while (low < high) and (v1[low] >= key1):
            low += 1
        v1[high] = v1[low]
        v2[high] = v2[low]
        v1[low] = key1
        v2[low] = key2
    return low
def quicksort(v1,v2, left, right):
    if left < right:
        p = parttion(v1,v2, left, right)
        print(p)
        quicksort(v1,v2, left, p-1)
        quicksort(v1,v2, p+1, right)
    return v1,v2
#降低维度
#把每类的tfidf和对应的词语重排序，写入新文件newtfidfpath，newdictpath
def reducedimension(tfidfpath,allwordpath,newtfidfpath,newdictpath):
    for i in range(len(tfidfpath)):
        with open(tfidfpath[i],'r',encoding='utf-8')as f:
            text=f.read()
            tfidftemp = text.split()
           # print("i1", i)
        with open(allwordpath,'r',encoding='utf-8')as f:
            text=f.read()
            allwordlisttemp =text.split()
           # print("i2", i)
        #print(len(tfidftemp))
        tfidflist = []
        allwordlist = []
        for j in range(len(tfidftemp)):
            k = float(tfidftemp[j])
            if k > 9.99999999e-05:
                tfidflist.append(k)
                allwordlist.append(allwordlisttemp[j])
        #print(tfidflist)
        #print(allwordlist)
        newtfidflist,newallwordlist=quicksort(tfidflist,allwordlist,0,len(tfidflist)-1)
        with open(newtfidfpath[i],'w',encoding='utf-8')as f:
            f.write(" ".join(str(newtfidflist)))
            #print("i3 tfidf",i)
        with open(newdictpath[i],'w',encoding='utf-8')as f:
            f.write(" ".join(newallwordlist))
            #print("i3 word",i)
#创建新字典
#在每类里找前n个tfidf最大的词语放进总字典，这里用scipy里dataframe和series合并
def createdict(newtfidfpath,newdictpath,dictpath):
    #dictdataframe = pd.DataFrame()
    l=[]
    #print(dictdataframe)
    for i in range(len(newtfidfpath)):
        with open(newtfidfpath[i],'r',encoding='utf-8')as f:
            tfidflist=[float(e) for e in f.read().split()[0:1000]]
        #print(tfidflist)
        #print(len(tfidflist))
        with open(newdictpath[i],'r',encoding='utf-8')as f:
            wordlist=f.read().split()[0:1000]
        #print(wordlist)
        #print(len(wordlist))
        s=Series(tfidflist,wordlist)
        l.append(s)
    dictdataframe = pd.DataFrame(l)
    #存起来
    pd.set_option('max_colwidth', 20000000)
    #print(dictdataframe)
    #print(" ".join(dictdataframe.columns.tolist()))
    with open(dictpath,'w',encoding='utf-8')as f:
        f.write(" ".join(dictdataframe.columns.tolist()))
# csvpath=["E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\caijing.csv",
#          "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\guoji.csv",
#          "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\IT.csv",
#          "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\junshi.csv",
#          "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\nengyuan.csv",
#          "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\qiche.csv",
#          "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\tiyu.csv",
#          "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\wenhua.csv",
#          "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\yule.csv",
#          "E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\jiankang.csv",
#          ]
# def createinputdata(dictpath,splitPathList,csvpath):
#     # 生成每篇series扩展成总的dict的index
#     # 生成一个大的dataframe
#     # 两个dataframe相乘
#     # k1=Series([1,1],index=['c','d'])
#     # k2=Series([1.5,2.6,7.6,8.9],index=['a','b','c','d'])
#     # k3=k1*k2
#     # print(k3)
#     articlelist=[]
#     with open(dictpath,'r',encoding='utf-8')as f:
#         dictlist=f.read().split()
#     dictSeries=Series(np.ones(len(dictlist)).tolist(),dictlist)
#     print(dictSeries)
#     for i in range(len(splitPathList)):
#         files = os.listdir(splitPathList[i])
#         index=0
#         listtemp=[]
#         for fi in files:
#             path = os.path.join(splitPathList[i], fi)
#             with open(path,'r')as f:
#                 article=list(set(f.read().strip().split('\n')))
#                 articleSeries = Series(np.ones(len(article)).tolist(), article).unique
#             #print(article)
#             articleSeries=dictSeries*articleSeries
#             #print(3)
#            # print(articleSeries.dropna())
#             articlelist.append(articleSeries)
#             listtemp.append((articleSeries))
#             index=index+1
#             if index%500==0 :
#                 articledataframe = DataFrame(articlelist)
#                 articlelist.clear()
#                 # print(articledataframe)
#                 if index<=50000:
#                     with open("E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\articletrain.csv", 'a')as f:
#                         articledataframe.to_csv(f, header=False)
#                 else:
#                     with open("E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\articletest.csv", 'a')as f:
#                         articledataframe.to_csv(f, header=False)
#             print("i index",i," ",index)
#         # tempdataframe = DataFrame(articlelist)
#         # tempdataframe.to_csv(csvpath[i])
#     # articledataframe = DataFrame(articlelist)
#     # #print(articledataframe)
#     # articledataframe.to_csv("E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\articledataframe.csv")
#根据新字典重新分词，新的分词结果存放在newsplit里，把不在字典里的词语扔掉
def createnewsplit(dictpath,splitPath):
    strinfo = re.compile('split')  # news
    with open(dictpath,'r',encoding='utf-8')as f:
        worddict=f.read().split()
        #ds = Series(np.ones(len(worddict)).tolist(), worddict)
        #print(worddict)
    for i in range(len(splitPath)):
        files=os.listdir(splitPath[i])
        index1=1
        for fi in files:
            path = os.path.join(splitPath, fi)
            if index1 >= 0:
                try:
                    list1 = [line.rstrip('\n') for line in open(path, 'r',encoding='utf-8')]
                except:
                    list1 = [line.rstrip('\n') for line in open(path, 'r')]
                list2=[e for e in list1 if e in worddict]
                list3=list(set(list1))
                writepath = strinfo.sub('newsplit', path)  # 分词结果写入的路径split
                with open(writepath, 'w', encoding='utf-8')as f:
                    f.write(" ".join(list3))
            #print("index1",index1)
            index1 = index1 + 1
#生成svm的输入数据
def svminputdata(dictpath,SplitPathList):
    with open(dictpath,'r',encoding='utf-8')as f:
    	dict=f.read().split()
    rindex=list(range(len(dict)))
    #rindex=["t"+str(e) for e in range(len(dict))]
    sd=Series(np.ones(len(dict)).tolist(),index=dict)
    strinfo = re.compile('split')  # news
    trainindex=1
    testindex=1
    for i in range(len(SplitPathList)):
        index1 = 1
        num=i+1
        files = os.listdir(SplitPathList[i])
        for fi in files:
            #print("s1")
            path=os.path.join(SplitPathList[i],fi)
            try:
                with open(path, 'r', encoding='utf-8')as f:
                    list1 = f.read().split()
            except:
                with open(path, 'r')as f:
                    list1 = f.read().split()
            list2 = [e for e in list1 if e in dict]
            # print(list2)
            s = Series(list2)
            s = s.value_counts()
            # print(list(s.index))
            # print(list(s.values))
            s2 = Series(s.values, index=s.index)
            # print("s2",s2)
            s3 = s2 * sd
            s3 = Series(s3.values, index=rindex)
            # s3=s3.fillna(0)
            # s3=s3.dropna()
            # print("s3",s3)
            s4 = s3[s3.notnull()]
            # print("s4",s4)
            s4index = s4.index
            s4values = s4.values
            # pint(s4index)
                # print(s4values)
            if index1<=500:
                if trainindex>=0:
                    str1 = ""
                    for j in range(len(s4)):
                        str1 = str1 + str(trainindex) + " " + str(s4index[j]) + " " + str(int(s4values[j])) + "\n"
                    with open("E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\svm\\train1.data", 'a',
                              encoding='utf-8')as f:
                        f.write(str1)
                    with open("E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\svm\\train1.label", 'a',
                              encoding='utf-8')as f:
                        f.write(str(num) + "\n")
                    writepath = strinfo.sub('splitwordfre', path)
                    # with open(writepath, 'w', encoding='utf-8')as f:
                    #     f.write(" ".join(list2))
                trainindex += 1
            if index1>500 and index1<=1000:
                if testindex>=0:
                    str1 = ""
                    for j in range(len(s4)):
                        str1 = str1 + str(int(testindex)) + " " + str(s4index[j]) + " " + str(int(s4values[j])) + "\n"
                    with open("E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\svm\\test1.data", 'a',
                              encoding='utf-8')as f:
                        f.write(str1)
                    with open("E:\\新建文件夹\\文本\\新建文件夹\\python\\DataMinning\\inputdata\\svm\\test1.label", 'a',
                              encoding='utf-8')as f:
                        f.write(str(num) + "\n")
                    writepath = strinfo.sub('splitwordfre', path)
                    # with open(writepath, 'w', encoding='utf-8')as f:
                    #     f.write(" ".join(list2))
                testindex += 1
            if index1==1000:
                break
            index1+=1
            # if trainindex==100:
            #     break
            print("type trainindex testindex articleindex",num," ",trainindex," ",testindex," ",index1)
        #print("index1",index1)
        # if index1==3:
        #     break
# 生成libsvm的输入数据，libsvm很慢
def libsvminputdata(dictpath, newdictpath,newtfidfpath,newSplitPathList):
    with open(dictpath,'r',encoding='utf-8')as f:
        dict=f.read().split()
    sd=Series(np.ones(len(dict)).tolist(),index=dict)
    print(len(dict))
    sl=[]
    rindex=[float(e) for e in range(len(dict))]
    for i in range(len(newdictpath)):
        with open(newdictpath[i], 'r', encoding='utf-8')as f:
            alldict = f.read().split()
        with open(newtfidfpath[i], 'r', encoding='utf-8')as f:
            alltfidf = [float(e) for e in f.read().split()]
        print(len(alldict))
        print(len(alltfidf))
        sad = Series(alltfidf, index=alldict)
        sl.append(sad)
    for i in range(len(newSplitPathList)):
        files = os.listdir(newSplitPathList[i])
        num=i+1
        for fi in files:
            #print("s1")
            path=os.path.join(newSplitPathList[i],fi)
            try:
                with open(path, 'r', encoding='utf-8')as f:
                     list1 = f.read().split()
            except:
                with open(path,'r')as f:
                    list1=f.read().split()
            #print(list1)
            s=Series(np.ones(len(list1)).tolist(),index=list1)
            print("s1",len(s))
            s2=s*sl[i]
            print("s2",len(s2))
            s3=s2*sd
            #s3=Series(s3.values,index=rindex)
            print("s3",len(s3))
            break
            #s3=s3.fillna(0)
            #s3=s3.dropna()
            #print("s3",s3)
            s4=s3[s3.notnull()]
            #print("s4",s4)
            s4index=s4.index
            s4values=s4.values
            #print(s4index)
            #print(s4values)
            str1=""
            for j in range(len(s4)):
                str1 = str1+str(trainindex) + " " +str(s4index[j])+" "+str(int(s4values[j]))+"\n"
            with open("",'a',encoding='utf-8')as f:
                f.write(str1)
            with open("",'a',encoding='utf-8')as f:
                f.write(str(num)+"\n")
        break
#test
# for i in range(10):
#     if i>1:
#         gci(newsPathList[i],i)
#tfidf(oriDictPathList)
#reducedimension(tfidfpath,allwordpath,newtfidfpath,newdictpath)
#createdict(newtfidfpath,newdictpath,dictpath)
#createinputdata(dictpath,splitPathList,csvpath)
#createnewsplit(dictpath,splitPathList)
#svminputdata(dictpath,splitPathList)
#libsvminputdata(dictpath, newdictpath,newtfidfpath,newSplitPathList)


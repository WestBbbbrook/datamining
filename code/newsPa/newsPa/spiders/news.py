# -*- coding: utf-8 -*-
import scrapy
from scrapy.http import Request
from scrapy.selector import Selector
import os
import codecs
import time
import datetime

mkpath = "C:\\Users\\chenteng\\Desktop\\NEWS-PA\\newsPa\\caijing\\"
#txtid = 0
#base_dir = mkpath
class NewsSpider(scrapy.Spider):
    name = 'news'
    allowed_domains = ['chinanews.com']
    start_urls = ['http://www.chinanews.com/']
    

    def parse(self, response):
    	date_list =  getBetweenDay("20091201")
    	for day in date_list:
    		year = day[0:4]
    		dayday = day[4:8]
    		#global base_dir
    		#base_dir = mkpath+day+"\\"
    		#mkdir(base_dir)
    		#global txtid
    		#txtid = 0
    		#time.sleep(0.1)
    		#修改cj字段为ty、gj、IT等等，自己看中国新闻网每类网址的名称
    		total = "http://www.chinanews.com/scroll-news/cj/{0}/{1}/news.shtml".format(year,dayday)
    		yield Request(total,meta = {"day":day},callback = self.info_1)
    	

    def info_1(self,response):
    	selector = Selector(response)
    	day = response.meta["day"]
    	base_dir = mkpath+day+"\\"
    	mkdir(base_dir)
    	txtid = 0
    	print "===============base_dir=============="
    	list = selector.xpath("//div[@class='dd_bt']/a/@href").extract()
    	for url in list:
    		txtid += 1
    		filename = base_dir  + str(txtid) +'.txt'
    		yield Request(url,meta = {"filename":filename},callback = self.info_2)

    def info_2(self,response):
    	selector = Selector(response)
    	filename = response.meta["filename"]
    	print "===============filename=============="
    	list = selector.xpath("//div[@class='left_zw']/p/text()").extract()
    	print list
    	#global txtid
    	#txtid +=1
    	#filename = base_dir  + str(txtid) +'.txt'
    	#print filename
    	f = codecs.open(filename,'a','utf-8')
    	for i in range(len(list)-1):
    		print '=========' + str(i) + "=========="
    		f.write(list[i])
    	f.close()


def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\") 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
 
        print path+' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print path+' 目录已存在'
        return False

def getBetweenDay(begin_date):  
    date_list = []  
    begin_date = datetime.datetime.strptime(begin_date, "%Y%m%d")  
    end_date = datetime.datetime.strptime(time.strftime('%Y%m%d',time.localtime(time.time())), "%Y%m%d")  
    while begin_date <= end_date:  
        date_str = begin_date.strftime("%Y%m%d")  
        date_list.append(date_str)  
        begin_date += datetime.timedelta(days=1)  
    return date_list
    	


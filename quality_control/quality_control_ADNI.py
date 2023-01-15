import re
import os
# -*- coding: utf-8 -*-
import xlsxwriter as xw

path='/data/datasets/ADNI/ADNI_T1_Fixed_20210719'

workbook = xw.Workbook('quality_control_ADNI.xlsx')  # 创建工作簿
worksheet1 = workbook.add_worksheet("ADNI")  # 创建子表
worksheet2 = workbook.add_worksheet("ADNI_err")
worksheet1.activate()  # 激活表
worksheet2.activate()
title = ['img', 'IQR', 'rank']  # 设置表头
worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
data=[]
err=[]

def xw_toExcel(data):  # xlsxwriter库储存数据到excel
       i = 2  # 从第二行开始写入数据
       for j in range(len(data)):
              insertData = [data[j]["img"], data[j]["IQR"], data[j]["rank"]]
              row = 'A' + str(i)
              worksheet1.write_row(row, insertData)
              i += 1

def xw_toExcel_err(err):  # xlsxwriter库储存数据到excel
       i = 1  # 从第二行开始写入数据
       for j in range(len(err)):
              insertData = [err[j]]
              row = 'A' + str(i)
              worksheet2.write_row(row, insertData)
              i += 1


for id in sorted(os.listdir(os.path.join(path))):
    flag = 1
    if '.' in id:
        continue
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_bl.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_bl.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1bl.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1bl.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m06.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m06.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m06.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m06.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m12.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m12.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m12.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m12.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m18.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m18.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m18.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m18.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m24.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m24.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m24.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m24.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m30.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m30.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m30.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m30.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m36.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m36.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m36.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m36.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m42.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m42.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m42.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m42.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m48.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m48.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m48.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m48.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m54.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m54.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m54.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m54.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m60.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m60.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m60.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m60.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m72.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m72.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m72.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m72.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m120.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m120.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m120.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m120.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if os.path.exists(os.path.join(path, id, 'report', 'catlog_m144.txt')):
        flag = 0
        f = open(os.path.join(path, id, 'report', 'catlog_m144.txt'), encoding='utf-8')
        res = re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)', f.read())
        try:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m144.nii'), "IQR": res[0][0], "rank": res[0][1]})
        except:
            data.append({"img": os.path.join(path, id, 'mri/mwp1m144.nii'), "IQR": 'NA', "rank": 'NA'})
        f.close()
    if flag:
        print(id)
        raise ValueError

# "-------------数据用例-------------"
# testData = [
#        {"id": 1, "name": "立智", "price": 100},
#        {"id": 2, "name": "维纳", "price": 200},
#        {"id": 3, "name": "如家", "price": 300},
# ]
xw_toExcel(data)
xw_toExcel_err(err)
workbook.close()
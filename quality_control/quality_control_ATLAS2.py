import re
import os
# -*- coding: utf-8 -*-
import xlsxwriter as xw
DATA_PATH='/data/datasets/Preprocessing/'
path=os.path.join(DATA_PATH, 'ATLAS_2/ATLAS_2_pred')

workbook = xw.Workbook('quality_control_ATLAS2.xlsx')  # 创建工作簿
worksheet1 = workbook.add_worksheet("ATLAS_2")  # 创建子表
worksheet2 = workbook.add_worksheet("ATLAS_2_err")
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

for tt in sorted(os.listdir(path)):
       for sub in sorted(os.listdir(os.path.join(path,tt))):
              for ses in sorted(os.listdir(os.path.join(path,tt,sub))):
                     if os.path.exists(os.path.join(path,tt,sub,ses,'err')):
                            err.append(os.path.join(path,tt,sub,ses))
                            continue
                     f=open(os.path.join(path,tt,sub,ses,'report','catlog_T1w.txt'),encoding='utf-8')
                     res=re.findall('Image Quality Rating \(IQR\):  (\w*.\w*%) \((\w\+?-?)\)',f.read())
                     # print(res)
                     # print(sub)
                     data.append({"img":os.path.join(path,tt,sub,ses,'mri','mwp1T1w.nii'),"IQR":res[0][0],"rank":res[0][1]})
                     f.close()

# "-------------数据用例-------------"
# testData = [
#        {"id": 1, "name": "立智", "price": 100},
#        {"id": 2, "name": "维纳", "price": 200},
#        {"id": 3, "name": "如家", "price": 300},
# ]
xw_toExcel(data)
xw_toExcel_err(err)
workbook.close()

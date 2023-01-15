import csv

#AIBL
def retrieve_AIBL_quality(img_path):
    with open('/data/home/feiyl/UROP/quality_control/quality_control_AIBL.csv','r') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        if row['img']==img_path:
            return row['IQR'],row['rank']
      raise ValueError("没有找到指定image")

def retrieve_ATLAS2_quality(img_path):
    with open('/data/home/feiyl/UROP/quality_control/quality_control_ATLAS2.csv','r') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        if row['img']==img_path:
            return row['IQR'],row['rank']
      print(img_path)
      raise ValueError("没有找到指定image")

def retrieve_ABIDE_quality(img_path):
    with open('/data/home/feiyl/UROP/quality_control/quality_control_ABIDE.csv','r') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        if row['img']==img_path:
            return row['IQR'],row['rank']
      print(img_path)
      raise ValueError("没有找到指定image")

def retrieve_MINDS_quality(img_path):
    with open('/data/home/feiyl/UROP/quality_control/quality_control_MINDS.csv','r') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        if row['img']==img_path:
            return row['IQR'],row['rank']
      print(img_path)
      raise ValueError("没有找到指定image")

def retrieve_ADNI_quality(img_path):
    with open('/data/home/feiyl/UROP/quality_control/quality_control_ADNI.csv','r') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        if row['img']==img_path:
            return row['IQR'],row['rank']
      print(img_path)
      raise ValueError("没有找到指定image")

def retrieve_COBRE_quality(img_path):
    with open('/data/home/feiyl/UROP/quality_control/quality_control_COBRE.csv','r') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        if row['img']==img_path:
            return row['IQR'],row['rank']
      print(img_path)
      raise ValueError("没有找到指定image")

# quality=retrieve_AIBL_quality('/data/datasets/AIBL/AIBL_pre/100/mri/mwp1bl.nii')
# print(quality[0])
# print(quality[1])
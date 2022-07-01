import os
import shutil
import pandas as pd

def move1():
    basepath='./train/images'
    copy_path = './trainset/images6/'
    for img_name in os.listdir(basepath):
        shutil.copy(basepath+'/'+img_name, copy_path+img_name)
            

def move2():
    basepath='./trainset1'
    copy_path = './trainset/images6/'
    for img_name in os.listdir(basepath):
        shutil.copy(basepath+'/'+img_name, copy_path+img_name[:-4]+'(1).jpg')

def move3():
    basepath='./trainset2'
    copy_path = './trainset/images3/'
    for img_name in os.listdir(basepath):
        shutil.copy(basepath+'/'+img_name, copy_path+img_name[:-4]+'(2).jpg')

def move4():
    basepath='./trainset3'
    copy_path = './trainset/images/'
    for img_name in os.listdir(basepath):
        shutil.copy(basepath+'/'+img_name, copy_path+img_name[:-4]+'(2).jpg')

def move5():
    basepath='./trainset4'
    copy_path = './trainset/images6/'
    for img_name in os.listdir(basepath):
        shutil.copy(basepath+'/'+img_name, copy_path+img_name[:-4]+'(2).jpg')

def move6():
    basepath='./trainset5'
    copy_path = './trainset/images6/'
    for img_name in os.listdir(basepath):
        shutil.copy(basepath+'/'+img_name, copy_path+img_name[:-4]+'(3).jpg')

def re_label():
    df = pd.read_csv('./train/grade_labels.csv')
    new_grade=[[label,label,label,label] for label in df['grade']]
    new_imname=[[name, name[:-4]+'(1).jpg',name[:-4]+'(2).jpg',name[:-4]+'(3).jpg'] for name in df['imname']]

    grade, imname = [], []

    for labels, imnames in zip(new_grade,new_imname):
        grade+=labels
        imname+=imnames

    new_df = pd.DataFrame({'imname': imname, 'grade': grade})
    new_df.to_csv('./trainset/new_grade_labels6.csv')


def main():
    move1()
    move2()
    # move3()
    # move4()
    move5()
    move6()
    re_label()

if __name__ == '__main__':
    if os.path.exists('./trainset/images6'):  # 반복적인 실행을 위해 디렉토리를 삭제
        shutil.rmtree('./trainset/images6')   
    os.mkdir('./trainset/images6')
    
    main()
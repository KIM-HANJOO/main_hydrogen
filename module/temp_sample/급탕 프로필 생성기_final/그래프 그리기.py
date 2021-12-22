import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sum_list = []
max_list = []
for i in range(9):
    df = pd.read_excel('{}번째 세대 급탕프로필(평일).xlsx'.format(i+1))
    df = df.drop('Unnamed: 0',axis=1)
    sum = df.sum().sum()/261
    max = df.max().max()
    sum_list.append(sum)
    max_list.append(max)
    print('{}번째 일평균 : {}'.format(i+1,sum))
    print('{}번째 최대 : {}'.format(i+1,max))
    plt.subplot(4,3,i+1)
    plt.plot(df)

path_dir = r"C:\Users\user\Desktop\pycharm\급탕 프로필 생성기_final"
file_list = os.listdir(path_dir)
condition = '(이상치제거)(급탕)(평일)*.*'
file_week = glob.glob(condition)

for i in range(3):
    df = pd.read_excel('{}'.format(file_week[i]))
    l = len(df)
    df = df.transpose()
    sum = df.sum().sum()/l
    max = df.max().max()
    sum_list.append(sum)
    max_list.append(max)
    print('{} 일평균 : {}'.format(file_week[i], sum))
    print('{}  최대 : {}'.format(file_week[i],max))
    plt.subplot(4,3,10+i)
    plt.plot(df)

x = range(12)
ax = plt.bar(x,max_list)
plt.xticks(x,['1','2','3','4','5','6','7','8','9','actual1','actual2','actual3'])
for p in ax.patches:
    left, bottom, width, height = p.get_bbox().bounds
    plt.annotate("%.1f" % (height), (left+width/2, height*1.01), ha='center')
plt.show()
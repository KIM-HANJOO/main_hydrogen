#condition_list = ['공공행정 국방 및 사회보장 행정', '교육 서비스업', \
#           '금융 및 보험업', '도매 및 소매업', '부동산업 및 임대업', \
#           '사업시설관리 및 사업지원 서비스업', '숙박 및 음식점업', \
#           '예술 스포츠 및 여가관련 서비스업', '전문 과학 및 기술 서비스업', \
#           '출판 영상 방송통신 및 정보서비스업']

################### 전처리 대상 ##########################

cwdir = 'D:\\project\\봄가을 전처리\\보낼파일\\봄가을 자동 완성'

# 정규화 할 업태 이름을 넣어주세요!
# 전처리가 되어 있어야만 동작합니다!
condition_list = ['공공행정 국방 및 사회보장 행정', '교육 서비스업', \
           '금융 및 보험업', '도매 및 소매업', '부동산업 및 임대업', \
           '사업시설관리 및 사업지원 서비스업', '숙박 및 음식점업', \
           '예술 스포츠 및 여가관련 서비스업', '전문 과학 및 기술 서비스업', \
           '출판 영상 방송통신 및 정보서비스업']

#########################################################

import sys
sys.path.append(cwdir)
import model_library as lib
import os

for folder in os.listdir(cwdir + '\\전체 업태') :
    if folder in condition_list :
        folder_dir = cwdir + '\\전체 업태\\' + folder
        if os.path.isdir(folder_dir) :
            src = folder_dir + '\\7_봄가을'
            dst = folder_dir + '\\정규화'
            f = ['주말', '주중']
            lib.newfolder(dst)
            lib.newfolderlist(dst, f)

            for subdir in os.listdir(src) :
                tempdir = src + '\\' + subdir
                print(tempdir)
                for excel in os.listdir(tempdir) :
                    os.chdir(tempdir)

                    temp = lib.read_excel(excel)
                    for i in range(temp.shape[0]) :
                        sum = temp.sum().sum()

                    sum = sum / temp.shape[0]

                    for i in range(temp.shape[0]) :
                        for col in temp.columns :
                            temp.loc[i, col] = temp.loc[i, col] / sum

                    os.chdir(dst + '\\' + subdir)
                    temp.to_excel('{}_정규화.xlsx'.format(excel[ : -5]))
                    print('{}_정규화.xlsx saved'.format(excel[ : -5]), end = '\r')
                    
##########################< 아래 부분은 모델3 전용>############################


cluster_all = ''
folder_list = []

for folder in cluster_all :
	if 'group_' in folder :
		folder_list.append('normalized_g_{}'.format(folder[-1 :]))

lib.newfolderlist(cluster_all, folder_list)
for folder in folder_list :
	lib.newfolderlist(cluster_all + '\\' + folder, ['주중', '주말'])
	
for folder in cluster_all :
	if 'group_' in folder :
		group_num = folder[-1 :]
		tempdir = cluster_all + '\\' + folder_dir
		
		for de in os.listdir(tempdir) :
			ttempdir = tempdir + '\\' + de
			dst = cluster_all + '\\normalized_g_{}\\'.format(group_num) + de
			
			for excel in os.listdir(ttempdir) :
				os.chdir(ttempdir)
				temp = lib.read_excel(excel)
				for i in range(temp.shape[0]) :
					sum = temp.sum().sum()

				sum = sum / temp.shape[0]

				for i in range(temp.shape[0]) :
					for col in temp.columns :
						temp.loc[i, col] = temp.loc[i, col] / sum

				os.chdir(dst)
				temp.to_excel('{}_정규화.xlsx'.format(excel[ : -5]))
				print('{}_정규화.xlsx saved'.format(excel[ : -5]), end = '\r')
				

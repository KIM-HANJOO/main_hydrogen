import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import os.path
import math
import random
import beta_rv_lib as bl


# 원하는 세대 수와 세대 특징 입력된 파일 읽기
def profile_generator(save_dir, profile_num, model1_file, model2_file, model3_file_day, model3_file_end, model4_file_day, model4_file_end) :
	
	for i in range(profile_num):
		'''
		모델 1
		'''
		
		# beta distribution
		num = bl.beta_rv(model1_file)
	
	    # ~ # 1일 평균량 구하기
	    # ~ week_1year = (2.6294 * bed + 0.8801)
	    # ~ weekend_1year = (2.8418 * bed +0.5123)
	
	    # ~ plus_year_week = random.uniform(-0.20,0.26)
	    # ~ plus_year_weekend = random.uniform(-0.30,0.26)
	
	    # ~ ave_week_1day = (week_1year*(1+plus_year_week))
	    # ~ ave_weekend_1day = (weekend_1year*(1+plus_year_weekend))
	
		'''
		모델2
		'''
		
		# beta distribution
		num2 = bl.beta_rv(model2_file)
		
	    # ~ # 침실수에 따른 표준편차 구하기
	    # ~ if bed == 2:
	        # ~ st_week = random.uniform(0.809,2.017)
	        # ~ st_weekend = random.uniform(0.879,2.579)
	    # ~ elif bed == 3:
	        # ~ st_week = random.uniform(0.758,3.172)
	        # ~ st_weekend = random.uniform(0.851,3.633)
	    # ~ elif bed == 4:
	        # ~ st_week = random.uniform(1.100,5.598)
	        # ~ st_weekend = random.uniform(1.108,6.224)
	    # ~ else:
	        # ~ print('요청한 침실수{}는 사용할 수 없습니다. 침실수 2,3,4만 사용가능합니다.'.format(bed*4))
	        # ~ break
	
	
	
	
		'''
		모델3
		'''
		
	    # 다변량 이용하여 고정 프로필 1개 만들기(평일)
	    sample = model3_file_day
	    sample = sample.transpose()
	    var_1 = sample.cov()
	    mean_1 = sample.mean()
	
	    fixed_profile_week = pd.DataFrame()
	
	    X = np.random.multivariate_normal(mean_1, var_1)
	    while 1:
	        for x in X:
	            if x < 0:
	                X = np.random.multivariate_normal(mean_1, var_1)
	                a = 0
	                break
	            a = 1
	        if a is 0:
	            continue
	        break
	    fixed_profile_week['fixed'] = X
	
	    # 다변량 이용하여 고정 프로필 1개 만들기(주말)
	    sample = model3_file_end
	    sample = sample.transpose()
	    sample = sample.transpose()
	    var_1 = sample.cov()
	    mean_1 = sample.mean()
	
	
	    fixed_profile_weekend = pd.DataFrame()
	
	    X = np.random.multivariate_normal(mean_1, var_1)
	    while 1:
	        for x in X:
	            if x < 0:
	                X = np.random.multivariate_normal(mean_1, var_1)
	                a = 0
	                break
	            a = 1
	        if a is 0:
	            continue
	        break
	    fixed_profile_weekend['fixed'] = X
	
	
	
		'''
		모델4
		'''
		
		
		
	    # 일마다 변하는 프로필 만들기(평일)
	    test = model4_file_day
	    test = test.transpose()
	    var_1 = test.cov()
	    mean_1 = test.mean()
	    changed_profile_week = pd.DataFrame()
	
	    for t in range(261):
	        # 평균과 표준편차를 이용하여 1일 사용량 도출
	        while 1:
	            week_1day = np.random.normal(ave_week_1day, st_week)
	            if week_1day > 0:
	                break
	        #weekend_1day = np.random.normal(ave_weekend_1day, st_weekend)
	        # 1일 프로필에 변화를 주는 프로필 생산
	        Y = []
	        X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore')
	        while 1:
	            for k in range(24):
	                x = fixed_profile_week.iloc[k,0] + X[k]
	                if x < 0:
	                    X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore')
	                    a = 0
	                    break
	                a = 1
	            if a is 0:
	                continue
	            break
	        for f in range(24):
	            y = (fixed_profile_week.iloc[f,0] + X[f])
	            Y.append(y)
	        Y = Y/sum(Y)
	        Y = Y * week_1day
	        changed_profile_week['{}일'.format(t+1)] = Y
	
	    # 일마다 변하는 프로필 만들기(주말)
	    test = model4_file_end
	    test = test.transpose()
	    var_1 = test.cov()
	    mean_1 = test.mean()
	    changed_profile_weekend = pd.DataFrame()
	
	    for t in range(104):
	        # 평균과 표준편차를 이용하여 1일 사용량 도출
	        #week_1day = np.random.normal(ave_week_1day, st_week)
	        while 1:
	            weekend_1day = np.random.normal(ave_weekend_1day, st_weekend)
	            if weekend_1day > 0:
	                break
	        # 1일 프로필에 변화를 주는 프로필 생산
	        Y = []
	        X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore')
	        while 1:
	            for k in range(24):
	                x = fixed_profile_weekend.iloc[k,0] + X[k]
	                if x < 0:
	                    X = np.random.multivariate_normal(mean_1, var_1, check_valid='ignore')
	                    a = 0
	                    break
	                a = 1
	            if a is 0:
	                continue
	            break
	        for f in range(24):
	            y = (fixed_profile_weekend.iloc[f,0] + X[f])
	            Y.append(y)
	        Y = Y/sum(Y)
	        Y = Y * weekend_1day
	        changed_profile_weekend['{}일'.format(t+1)] = Y
	        
	    print('{}번째전력프로필 완성'.format(i+1))
	    
	    os.chdir(save_dir)
	    
	    # 완성된 파일 저장
	    changed_profile_week.to_excel('{}번째 세대 전력프로필(평일).xlsx'.format(i+1))
	    changed_profile_weekend.to_excel('{}번째 세대 전력프로필(주말).xlsx'.format(i+1))
	
	

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import os.path
import math
import random
import scipy.stats as ss


# 원하는 세대 수와 세대 특징 입력된 파일 읽기
want = pd.read_excel('원하는 세대 특징(이파일만변경).xlsx')
for p in range(len(want)):
    occ = want.iloc[p,0]

    # 1일 평균량 구하기
    week_1year = (3.6699 * occ - 3.6433)
    weekend_1year = (3.5497 * occ - 2.9459)

    plus_year_week = random.uniform(-0.35, 0.39)
    plus_year_weekend = random.uniform(-0.34, 0.36)

    ave_week_1day = (week_1year*(1+plus_year_week))
    ave_weekend_1day = (weekend_1year*(1+plus_year_weekend))

    # 침실수에 따른 표준편차 구하기
    if occ == 2:
        st_week = random.uniform(2.457,3.991)
        st_weekend = random.uniform(3.176,3.654)
    elif occ == 3:
        st_week = random.uniform(3.762,6.387)
        st_weekend = random.uniform(4.656,8.655)
    elif occ == 4:
        st_week = random.uniform(4.024,9.616)
        st_weekend = random.uniform(4.636,11.288)
    else:
        print('요청한 거주자수{}는 사용할 수 없습니다. 거주자수 2,3,4만 사용가능합니다.'.format(occ))
        break

    # 다변량 이용하여 고정 프로필 1개 만들기(평일)
    sample = pd.read_excel('(model3)(급탕)(평일)다변량용.xlsx')
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
    sample = pd.read_excel('(model3)(급탕)(주말)다변량용.xlsx')
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

    # 일마다 변하는 프로필 만들기(평일)
    model4_beta_week = pd.read_excel('(model4)(급탕)(평일)beta_분포.xlsx')
    model4_beta_weekend = pd.read_excel('(model4)(급탕)(주말)beta_분포.xlsx')
    changed_profile_week = pd.DataFrame()
    changed_profile_weekend = pd.DataFrame()

    for t in range(261):
        while 1:
            week_1day = np.random.normal(ave_week_1day, st_week)
            if week_1day > 0:
                break
        Y = []
        for i in range(24):
            c_model4 = model4_beta_week.iloc[0, i]
            k_model4 = model4_beta_week.iloc[1, i]
            loc_model4 = model4_beta_week.iloc[2, i]
            alpa_model4 = model4_beta_week.iloc[3, i]
            temp = ss.beta.rvs(c_model4, k_model4, loc_model4, alpa_model4)
            while fixed_profile_week.iloc[i,0] + temp < 0:
                temp = ss.beta.rvs(c_model4, k_model4, loc_model4, alpa_model4)
            Y.append(temp + fixed_profile_week.iloc[i,0])
        Y = Y / np.sum(Y)
        Y = Y * week_1day
        changed_profile_week['{}일'.format(t+1)] = Y

    # 일마다 변하는 프로필 만들기(주말)
    for t in range(104):
        while 1:
            weekend_1day = np.random.normal(ave_weekend_1day, st_weekend)
            if weekend_1day > 0:
                break
        Y = []
        for i in range(24):
            c_model4 = model4_beta_weekend.iloc[0, i]
            k_model4 = model4_beta_weekend.iloc[1, i]
            loc_model4 = model4_beta_weekend.iloc[2, i]
            alpa_model4 = model4_beta_weekend.iloc[3, i]
            temp = ss.beta.rvs(c_model4, k_model4, loc_model4, alpa_model4)
            while fixed_profile_weekend.iloc[i, 0] + temp < 0:
                temp = ss.beta.rvs(c_model4, k_model4, loc_model4, alpa_model4)
            Y.append(temp + fixed_profile_weekend.iloc[i, 0])
        Y = Y / np.sum(Y)
        Y = Y * weekend_1day
        changed_profile_weekend['{}일'.format(t + 1)] = Y
    print('{}번째급탕프로필 완성'.format(p + 1))

    # 완성된 파일 저장
    changed_profile_week.to_excel('{}번째 세대 급탕프로필(평일).xlsx'.format(p+1))
    changed_profile_weekend.to_excel('{}번째 세대 급탕프로필(주말).xlsx'.format(p+1))
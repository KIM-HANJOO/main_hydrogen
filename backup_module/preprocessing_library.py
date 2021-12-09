import numpy as np
import pandas as pd
import os
import math
import shutil

#########################################################################################
#########################################################################################
##                                                                                     ##
##                               #### HOW TO IMPORT ####                               ##
##                                                                                     ##
## home desktop | import model_lib                                                     ##
##                                                                                     ##
## import sys
## sys.path.append('C:\\Users\\joo09\\Documents\\GitHub\\LIBRARY')
## import preprocessing_all as ppa
##                                                                                     ##
##                                                                                     ##
##                                                                                     ##
##                           #### HOW TO RE - IMPORT ####                              ##
##                                                                                     ##
## from imp import reload
## reload(ppa)
##                                                                                     ##
#########################################################################################
#########################################################################################


import sys
import model_library as lib


def make_md() :
    md = pd.DataFrame(columns = ['month', 'day', 'all_start', 'all_end'])
    month = range(1, 13)

    md.loc[:, 'month'] = month

    md.loc[0, 'day'] = 31
    md.loc[1, 'day'] = 28
    md.loc[2, 'day'] = 31
    md.loc[3, 'day'] = 30
    md.loc[4, 'day'] = 31
    md.loc[5, 'day'] = 30
    md.loc[6, 'day'] = 31
    md.loc[7, 'day'] = 31
    md.loc[8, 'day'] = 30
    md.loc[9, 'day'] = 31
    md.loc[10, 'day'] = 30
    md.loc[11, 'day'] = 31

    sum = 0
    for i in range(md.shape[0]) :
        md.loc[i, 'all_start'] = sum
        sum = sum + md.loc[i, 'day'] - 1

        md.loc[i, 'all_end'] = sum
        sum = sum + 1
        
    return md

def find_position(month_, day_) :
    
    md = pd.DataFrame(columns = ['month', 'day', 'all_start', 'all_end'])
    month = range(1, 13)

    md.loc[:, 'month'] = month

    md.loc[0, 'day'] = 31
    md.loc[1, 'day'] = 28
    md.loc[2, 'day'] = 31
    md.loc[3, 'day'] = 30
    md.loc[4, 'day'] = 31
    md.loc[5, 'day'] = 30
    md.loc[6, 'day'] = 31
    md.loc[7, 'day'] = 31
    md.loc[8, 'day'] = 30
    md.loc[9, 'day'] = 31
    md.loc[10, 'day'] = 30
    md.loc[11, 'day'] = 31

    sum = 0
    for i in range(md.shape[0]) :
        md.loc[i, 'all_start'] = sum
        sum = sum + md.loc[i, 'day'] - 1

        md.loc[i, 'all_end'] = sum
        sum = sum + 1

    curr_md = md[md['month'] == month_]
    curr_md.reset_index(drop = True, inplace = True)
    start = curr_md.loc[0, 'all_start']
    position = start + day_ - 1

    return position

def find_date(num) :
    df = pd.DataFrame(columns = ['num', 'month', 'day'])
    
    num_list = []
    for i in range(365) :
        num_list.append(i)
        
    df.loc[:, 'num'] = num_list
    
    md = make_md()
    
    for i in range(md.shape[0]) :
        if (num >= md.loc[i, 'all_start']) & (num <= md.loc[i, 'all_end']) :
            now_month = md.loc[i, 'month']
            now_day = num - md.loc[i, 'all_start'] + 1
            
    return [now_month, now_day]


def para() :
    first_sat_sun = [5, 6]
    weekends = []

    for x in first_sat_sun :
        x = x - 1
        while x < 365 :
            weekends.append(x)
            x = x + 7

    weekends = sorted(weekends)
    weekdays = []
    for i in range(0, 365) :
        if i not in weekends :
            weekdays.append(i)
    
    day_list = []
    day_num = []
    for i in range(261) :
        day_list.append('day')
        day_num.append(i)
        
    end_list = []
    end_num = []
    for i in range(104) :
        end_num.append(i)
        end_list.append('end')
    all_list = day_list + end_list

    ult_para = pd.DataFrame(columns = ['weekday/end', 'num', 'actual', 'month', 'day', 'drop'])    
    ult_para.loc[:, 'weekday/end'] = all_list
    ult_para.loc[0 : 0 + 261 - 1, 'num'] = day_num
    ult_para.loc[261 : 261 + 104 - 1, 'num'] = end_num
    ult_para.loc[0 : 0 + 261 - 1, 'actual'] = weekdays
    ult_para.loc[261 : 261 + 104 - 1, 'actual'] = weekends
    
    for i in range(ult_para.shape[0]) :
        temp = ult_para.loc[i, 'actual']
        month_day = find_date(temp)
        ult_para.loc[i, 'month'] = month_day[0]
        ult_para.loc[i, 'day'] = month_day[1]
        
    start_summer = find_position(6, 22)
    end_summer = find_position(9, 23)

    start_winter_2 = find_position(12, 22)
    end_winter_1 = find_position(3, 21)

    start_winter_1 = 0
    end_winter_2 = 364

    for i in range(ult_para.shape[0]) :
        act = ult_para.loc[i, 'actual']
        if (start_summer <= act <= end_summer) | \
            (start_winter_1 <= act <= end_winter_1) | (start_winter_2 <= act <= end_winter_2) :
            ult_para.loc[i, 'drop'] = 'drop'
        else :
            ult_para.loc[i, 'drop'] = 'save'
            
        
    return ult_para

def change_para(ult_para) :
    change_log = pd.DataFrame(columns = ['weekday/end', '1_split', '2_missing', '3_outliers', '4_linear', \
                                        '5_semi', '6_done', '7_sf'])
    
    change_log.loc[:, 'weekday/end'] = ult_para.loc[:, 'weekday/end'].tolist()
    change_log.loc[:, '1_split'] = ult_para.loc[:, 'num'].tolist()
    
    change_log.loc[:, '2_missing'] = change_log.loc[:, '1_split']
    change_log.loc[:, '3_outliers'] = change_log.loc[:, '1_split']

    change_log

    return change_log


#########################################################################################

class Preprocessing() :
    def __init__(self, main_dir) :
        self.main_dir = main_dir
        self.folder_list = ['공공행정 국방 및 사회보장 행정', '교육 서비스업', \
            '금융 및 보험업', '도매 및 소매업', '부동산업 및 임대업', \
            '사업시설관리 및 사업지원 서비스업', '숙박 및 음식점업', \
            '예술 스포츠 및 여가관련 서비스업', '전문 과학 및 기술 서비스업', \
            '출판 영상 방송통신 및 정보서비스업']

        all_param = pd.DataFrame(columns = ['condition', 'stmd'])

        for i in range(len(self.folder_list)) :
            all_param.loc[i, 'condition'] = self.folder_list[i]
            
        all_param.loc[0, 'stmd'] = 20
        all_param.loc[1, 'stmd'] = 10
        all_param.loc[2, 'stmd'] = 20
        all_param.loc[3, 'stmd'] = 10
        all_param.loc[4, 'stmd'] = 10
        all_param.loc[5, 'stmd'] = 10
        all_param.loc[6, 'stmd'] = 10
        all_param.loc[7, 'stmd'] = 10
        all_param.loc[8, 'stmd'] = 10
        all_param.loc[9, 'stmd'] = 10

        self.all_param = all_param
        all_param = None

        for folder in os.listdir(main_dir) :
            if os.path.isdir(main_dir + '\\' + folder) :
                temp_dir = main_dir + '\\' + folder
                break
        self.folder_dir = temp_dir

        temp = pd.DataFrame(columns = ['condition', 'condition_dir', 'splitdir', 'sortdir', 'outlierdir', 'dirbtween',\
                                                'donedir', 'finaldir', 'sfdir', 'paradir', 'paraalldir'])
        temp.loc[:, 'condition'] = self.folder_list
        for i in range(temp.shape[0]) :
            temp_dir = main_dir + '\\' + temp.loc[i, 'condition']
            temp.loc[i, 'condition_dir'] = temp_dir
            temp.loc[i, 'splitdir'] = temp_dir + '\\1_주말_주중_나누기'
            temp.loc[i, 'sortdir'] = temp_dir + '\\2_결측값_기준_이하_제거'
            temp.loc[i, 'outlierdir'] = temp_dir + '\\3_이상치제거'
            temp.loc[i, 'dirbtween'] = temp_dir + '\\4_선형보간'
            temp.loc[i, 'donedir'] = temp_dir + '\\5_전처리완료'
            temp.loc[i, 'finaldir'] = temp_dir + '\\6_최종'
            temp.loc[i, 'sfdir'] = temp_dir + '\\7_봄가을'
            temp.loc[i, 'paradir'] = temp_dir + '\\para'
            temp.loc[i, 'paraalldir'] = temp_dir + '\\para_all'
        self.dir_list = temp
        temp = None

        # print(self.folder_list)
        # print(self.all_param)
        # print(self.dir_list)
        # print(self.dir_list)
        condition_list = self.dir_list.loc[:, 'condition'].tolist()
        print('condition list = \n{}'.format(condition_list), '\n')

    def set(self, condition) :
        run = 0
        for i in range(self.all_param.shape[0]) :
            if condition in self.all_param.loc[i, 'condition'] :
                self.condition_now = condition
                self.condition_set = condition
                self.stmd_set = self.all_param.loc[i, 'stmd']
                self.whisker = 4.5
                self.interpolation_limit = 50
                self.zero_value_ratio = 30
                
                run = 1

        if run == 1 :                    
            max_lenght = 21
            
            temp_string_list = ['condition_set', 'stmd_set', \
                                'whisker', 'interpolation_limit', 'zero_value_ratio']
            for string in temp_string_list :
                print('{}'.format(string) + ' ' * (19  - len(string)) \
                    + '=' + ' ' * 3 + '{}'.format(eval('self.{}'.format(string))))

            print('\n')

        else : 
            print('condition name not readable')
        


    def do(self, condition) :
        self.set(condition)
        # print('condition = {}'.format(condition))
        folderlist = ['주중', '주말']
        
        run = 0
        for i in range(self.dir_list.shape[0]) :
            if condition in self.dir_list.loc[i, 'condition'] :
                category_ami = 0
                currentfolder = self.dir_list.loc[i, 'condition']

                if condition in self.all_param.loc[i, 'condition'] :
                    stmd = self.all_param.loc[i, 'stmd']
                else :
                    print('stmd not found')



                maindir = self.dir_list.loc[i, 'condition_dir']
                rawdir = maindir + '\\0_raw'
                progressfolder = ['1_주말_주중_나누기', '2_결측값_기준_이하_제거', '3_이상치제거', '4_선형보간', \
                  '5_전처리완료', '6_최종', '7_봄가을', 'para', 'para_all']
                
                splitdir = self.dir_list.loc[i, 'splitdir']
                sortdir = self.dir_list.loc[i, 'sortdir']
                outlierdir = self.dir_list.loc[i, 'outlierdir']
                dirbtween = self.dir_list.loc[i, 'dirbtween']
                donedir = self.dir_list.loc[i, 'donedir']
                finaldir = self.dir_list.loc[i, 'finaldir']
                sfdir = self.dir_list.loc[i, 'sfdir']
                paradir = self.dir_list.loc[i, 'paradir']
                paraalldir = self.dir_list.loc[i, 'paraalldir']
                plotdir = maindir

                run = 1

        if run == 0 :
            # print('conditon name not readable')
            # print(self.dir_list)
            pass

        elif run == 1 :
            md = make_md()

            # 기상청 기준, 2019년 여름은 6월 22일 - 9월 23일, 겨울은 12월 22일부터 2020년 3월 20일
            # 또한 기상청 기준, 2018년 겨울은 2019년 3월 21일까지

            start_summer = find_position(6, 22)
            end_summer = find_position(9, 23)

            start_winter_2 = find_position(12, 22)
            end_winter_1 = find_position(3, 21)

            start_winter_1 = 0
            end_winter_2 = 364

            ult_para = para()

            change_log = change_para(ult_para)
            change_log.loc[:, '2_missing'] = change_log.loc[:, '1_split']
            change_log.loc[:, '3_outliers'] = change_log.loc[:, '1_split']

            ### 자동 폴더 만들기
            lib.newfolderlist(maindir, progressfolder)
            for name in progressfolder :
                if name in os.listdir(maindir) :
                    tempdir = maindir + '\\' + name
                    lib.newfolderlist(tempdir, folderlist)
                    
            ### 세대별 데이터 주중, 주말 나누기
            # 주중, 주말로 데이터 나누기

            # 해당 년도(2019)의 첫 토요일과 일요일
            first_sat_sun = [5, 6]
            weekends = []

            for x in first_sat_sun :
                x = x - 1
                while x < 365 :
                    weekends.append(x)
                    x = x + 7
                    
            weekends = sorted(weekends)
            weekdays = []
            for i in range(0, 365) :
                if i not in weekends :
                    weekdays.append(i)

            # 파일 불러와서 주중, 주말 나누어 '주말주중' 폴더에 저장

            # '주말주중' 폴더의 하위폴더 디렉토리
            for folder in folderlist :
                if '주중' in folder :
                    weekdays_dir = maindir + '\\1_주말_주중_나누기\\' + folder
                elif '주말' in folder :
                    weekends_dir = maindir + '\\1_주말_주중_나누기\\' + folder

            for excel in os.listdir(rawdir) :
                os.chdir(rawdir)
                temp = lib.read_excel(excel)
                temp.reset_index(drop = True, inplace = True)
                
                temp_weekdays = temp.loc[weekdays, :]
                temp_weekends = temp.loc[weekends, :]
                
                temp_weekdays.reset_index(drop = True, inplace = True)
                temp_weekends.reset_index(drop = True, inplace = True)

                os.chdir(weekdays_dir)
                temp_weekdays.to_excel('{}_주중.xlsx'.format(excel[: -5]))
                
                os.chdir(weekends_dir)
                temp_weekends.to_excel('{}_주말.xlsx'.format(excel[: -5]))
                
                print('{}를 주말/주중으로 잘라 저장했습니다'.format(excel), end = '\r')

            print('\n', '작업 완료')

            print('\n')


            ### 결측값 특정기준 이하 세대만 분류
            nan_df_days = pd.DataFrame(columns = ['excel', 'nan_percentage'])
            nan_df_ends = pd.DataFrame(columns = ['excel', 'nan_percentage'])
            num_nan_days = 0
            num_nan_ends = 0

            for subdir in os.listdir(splitdir) :
                tempdir = splitdir + '\\' + subdir
                os.chdir(tempdir)
                for excel in os.listdir(tempdir) :
                    temp = lib.read_excel(excel)
                
                    if '주중' in subdir :
                        nan_df_days.loc[num_nan_days, 'excel'] = excel
                        nan_df_days.loc[num_nan_days, 'nan_percentage'] = lib.pernull(temp)
                        num_nan_days += 1
                        
                    elif '주말' in subdir :
                        nan_df_ends.loc[num_nan_ends, 'excel'] = excel
                        nan_df_ends.loc[num_nan_ends, 'nan_percentage'] = lib.pernull(temp)
                        num_nan_ends += 1
                    print('{} 결측값 퍼센트 저장'.format(excel), end = '\r')

            os.chdir(plotdir)
            nan_df_days.to_excel('주중_결측값퍼센트.xlsx')
            nan_df_ends.to_excel('주말_결측값퍼센트.xlsx')
            print('\n', '작업 완료')

            print('\n')

            meanper_days = lib.ave(nan_df_days.loc[:, 'nan_percentage'].tolist())
            meanper_ends = lib.ave(nan_df_ends.loc[:, 'nan_percentage'].tolist())

            total = len(nan_df_days.loc[:, 'nan_percentage'].tolist())
            mean_df = pd.DataFrame(columns = ['excel', 'average_nanperc'])
            num_mean = 0

            for index in range(total) :
                mean_per = (nan_df_days.loc[index, 'nan_percentage'] + nan_df_ends.loc[index, 'nan_percentage']) / 2
                mean_df.loc[num_mean, 'excel'] = nan_df_days.loc[index, 'excel'][ : -8]
                mean_df.loc[num_mean, 'average_nanperc'] = mean_per
                num_mean += 1

            savelist = []
            for i in range(mean_df.shape[0]) :
                if mean_df.loc[i, 'average_nanperc'] < stmd :
                    savelist.append(mean_df.loc[i, 'excel'] + '_')
                    
            print('결측값이 {}% 미만인 데이터는 총 {} / {}개, 전체 세대의 {}% 입니다.'.format(stmd, \
                    len(savelist), len(os.listdir(rawdir)), len(savelist) / len(os.listdir(rawdir)) * 100))

            for folder in os.listdir(splitdir) :
                tempdir = splitdir + '\\' + folder
                
                for excel in os.listdir(tempdir) :
                    check = 0
                    os.chdir(tempdir)
                    for name in savelist :
                        if name in excel : 
                            check = 1
                    
                    if check == 1 :
                        temp = lib.read_excel(excel)

                        tempdir_sort = sortdir + '\\' + folder
                        os.chdir(tempdir_sort)
                        temp.to_excel(excel)
                        print('{} is pasted'.format(excel), end = '\r')
                            
                    elif check == 0 :
                        print('{} is out'.format(excel), end = '\r')

            print('\n', 'all done')

            print('\n')

            ### tukey's fences 으로 이상치 바꾸기

            outlier_df = lib.outlier_df_maker(sortdir, outlierdir)
            
            ######

            std = 1

            num_15 = []
            num_30 = []
            num_45 = []

            for i in range(outlier_df.shape[0]) :
                if outlier_df.loc[i, '1.5'] > std :
                    num_15.append(outlier_df.loc[i, 'excel'])
                if outlier_df.loc[i, '3.0'] > std :
                    num_30.append(outlier_df.loc[i, 'excel'])
                if outlier_df.loc[i, '4.5'] > std :
                    num_45.append(outlier_df.loc[i, 'excel'])

            print(len(num_15))
            print(len(num_30))
            print(len(num_45))

            print('whisker 1.5 = {}% ({} / {})'.format(len(num_15) / outlier_df.shape[0] * 100, len(num_15), outlier_df.shape[0]))
            print('whisker 3.0 = {}% ({} / {})'.format(len(num_30) / outlier_df.shape[0] * 100, len(num_30), outlier_df.shape[0]))
            print('whisker 4.5 = {}% ({} / {})'.format(len(num_45) / outlier_df.shape[0] * 100, len(num_45), outlier_df.shape[0]))
                
            temp_df = pd.DataFrame(columns = ['1.5', '3.0', '4.5'])
            temp_df.loc[0, '1.5' : '4.5'] = len(num_15), len(num_30), len(num_45)
            temp_df.loc[1, '1.5' : '4.5'] = len(num_15) / outlier_df.shape[0] * 100, \
                                            len(num_30) / outlier_df.shape[0] * 100, len(num_45) / outlier_df.shape[0] * 100

            temp_df.index = ['meters (total{})'.format(outlier_df.shape[0]), 'percentage']
            temp_df

            #########

            oneper_list, tukey_list = lib.outlier_df_check(outlier_df, 4.5, 1)

            for folder in os.listdir(sortdir) :
                if '주말' in folder :
                    weekends_sort = sortdir + '\\' + folder
                elif '주중' in folder :
                    weekdays_sort = sortdir + '\\' + folder
                    
            for folder in os.listdir(outlierdir) :
                if '주말' in folder :
                    weekends_out = outlierdir + '\\' + folder
                elif '주중' in folder :
                    weekdays_out = outlierdir + '\\' + folder
                    
            lib.check1per(weekends_sort, weekends_out, 4.5)
            lib.check1per(weekdays_sort, weekdays_out, 4.5)


            ### 선형보간
            interpolate_standard = 50 # days

            dropnum = 0
            savenum = 0

            os.chdir(outlierdir)
            for folder in os.listdir(outlierdir) :
                tempdir = outlierdir + '\\' + folder
                if os.path.isdir(tempdir) :
                    for excel in os.listdir(tempdir) :
                        os.chdir(tempdir)
                        temp = lib.read_excel(excel)

                        temp, dropindex = lib.interp(temp)
                    
                        if len(dropindex) < interpolate_standard:
                            os.chdir(dirbtween + '\\' + folder)
                            temp.to_excel(excel)
                            savenum += 1
                            print('{} saved'.format(excel), end = '\r')

                        else :
                            dropnum += 1
                            print('{} droped, {}th'.format(excel, dropnum), end = '\r')
                            
                        ##################################################
                        
                        
                        weekday_rath = 0
                        if '주중' in excel :
                            weekday_rath = 1
                            
                        ch_t = change_para(ult_para)
                        
                        if weekday_rath == 1 :
                            ch_t = ch_t[ch_t['weekday/end'] == 'day']
                            add_dir = '\\주중'
                        else :
                            ch_t = ch_t[ch_t['weekday/end'] == 'end']
                            add_dir = '\\주말'
                        
                        ch_t.reset_index(drop = True, inplace = True)
                        
                        
                        for num_drop in dropindex :
                            for i in range(ch_t.shape[0]) :
                                if ch_t.loc[i, '1_split'] == num_drop :
                                    ch_t.loc[i, '4_linear'] = 'drop'
                                
                        startnum = 0
                        for i in range(ch_t.shape[0]) :
                            if ch_t.loc[i, '4_linear'] != 'drop' :
                                ch_t.loc[i, '4_linear'] = startnum
                                startnum += 1
                                
                        #             print(dropindex)
                        #             print(ch_t.loc[:, '4_linear'])
                        #             print(ch_t)
                        #             print(temp.index)

                        
                        if len(dropindex) < interpolate_standard :
                            add = ''
                        else :
                            add = '_drop'
                            
                        os.chdir(paradir + add_dir)
                        ch_t.to_excel('{}_log{}.xlsx'.format(excel[ : -5], add))
                        os.chdir(tempdir)
                        
                        ##################################################

            temp_df = pd.DataFrame(columns = ['all', 'saved', 'dropped'])
            temp_df.loc[0, 'saved'] = savenum
            temp_df.loc[0, 'dropped'] = dropnum
            temp_df.loc[0, 'all'] = savenum + dropnum

            temp_df.loc[1, 'all'] = 100
            temp_df.loc[1, 'saved'] = savenum / (savenum + dropnum) * 100
            temp_df.loc[1, 'dropped'] = dropnum / (savenum + dropnum) * 100

            temp_df.index = ['num', 'percentage']


            error_excel = []
            error_values = []
            for folder in os.listdir(dirbtween) :
                tempdir = dirbtween + '\\' + folder
                os.chdir(tempdir)
                for excel in os.listdir(tempdir) :
                    temp = lib.read_excel(excel)
                    if lib.pernull(temp) != 0 :
                        error_excel.append(excel)
                        error_values.append(lib.pernull(temp))

            if len(error_excel) != 0 :
                print('\n', 'some not interpolated appropriately', '\n')
                for i in range(len(error_excel)) :
                    print('{} have {} null values'.format(error_excel[i], error_values[i]))
            else : 
                print('\n', 'interpolation done')

            print(temp_df, '\n', '\n')

            ### 주말, 주중 모두 살아있는 세대 분류

            days = []
            ends = []

            os.chdir(dirbtween)
            for folder in os.listdir(dirbtween) :
                tempdir = dirbtween + '\\' + folder
                if '주중' in folder :
                    days = os.listdir(tempdir)
                else :
                    ends = os.listdir(tempdir)
                    

            sharedays = []        
            shareends = []

            for x in days :
                for y in ends :
                    if x[ : -7] == y[ : -7] :
                        sharedays.append(x)
                        shareends.append(y)
                    
            print('주중 {}, 주말 {} 세대 공통으로 포함, 최종 전처리 완료'.format(len(sharedays), len(shareends)))

            for folder in os.listdir(outlierdir) :
                if '주중' in folder :
                    daysfolder = folder
                elif '주말' in folder :
                    endsfolder = folder

            for x in sharedays :
                src = dirbtween + '\\' + daysfolder + '\\' + x
                dst = donedir + '\\' + daysfolder + '\\' + x
                shutil.copyfile(src, dst)
                
            for y in shareends :
                src = dirbtween + '\\' + endsfolder + '\\' + y
                dst = donedir + '\\' + endsfolder + '\\' + y
                shutil.copyfile(src, dst)

                
                
            ### nan값 살아있는 세대 확인

            error_excel = []
            error_values = []
            for folder in os.listdir(donedir) :
                tempdir = donedir + '\\' + folder
                os.chdir(tempdir)
                for excel in os.listdir(tempdir) :
                    temp = lib.read_excel(excel)
                    if lib.pernull(temp) != 0 :
                        error_excel.append(excel)
                        error_values.append(lib.pernull(temp))

            if len(error_excel) != 0 :
                print('some not interpolated appropriately', '\n')
                for i in range(len(error_excel)) :
                    print('{} have {} null values'.format(error_excel[i], error_values[i]))
            else : 
                print('interpolation done')
                
                
            ### 오류 세대 지우기 (결측값 30% 이상)

            remove_excel = []

            for folder in os.listdir(donedir) :
                tempdir = donedir + '\\' + folder
                for excel in os.listdir(tempdir) :
                    os.chdir(tempdir)
                    temp = lib.read_excel(excel)
                    temp.reset_index(drop = True, inplace = True)
                    
                    zero_num = 0
                    for i in range(temp.shape[0]) :
                        for cat in temp.columns :
                            if temp.loc[i, cat] == 0 :
                                zero_num += 1
                                
                    if zero_num < round(temp.shape[0] * temp.shape[1] * 0.3) :
                        os.chdir(finaldir + '\\' + folder)
                        temp.to_excel(excel)
                    else :
                        remove_excel.append(excel[ : -7])
                        print('{} have more than 20% of data as zero'.format(excel))

            for folder in os.listdir(finaldir) :
                tempdir = finaldir + '\\' + folder
                for excel in os.listdir(tempdir) :
                    for remove in remove_excel :
                        if remove in excel :
                            lib.remove(tempdir + '\\' + excel)
                            print('{} is removed'.format(excel))
                            
            print('\n', 'done')

            temp_name = []
            temp_num = []

            for folder in os.listdir(maindir) :
                for j in range(1, 7) :
                    if '{}_'.format(j) in folder[ : 2] :
                        tempdir = maindir + '\\' + folder
                        temp_name.append(folder)
                        num = 0
                        for subdir in os.listdir(tempdir) :
                            if os.path.isdir(tempdir + '\\' + subdir) :
                                num += len(os.listdir(tempdir + '\\' + subdir))
                        temp_num.append(num)
                        
            print(temp_name)
            print(temp_num)
            temp_df = pd.DataFrame(columns = ['주말 or 주중 세대수', '감소 비율(%)'])

            for i in range(len(temp_name)) :
                temp_df.loc[i, :] = temp_num[i]
                
            temp_df.loc[0, '감소 비율(%)'] = 0

            for i in range(temp_df.shape[0]) :
                if i > 0 :
                    temp_df.loc[i, '감소 비율(%)'] = round((temp_df.loc[i, '주말 or 주중 세대수'] - temp_df.loc[i - 1, '주말 or 주중 세대수'])\
                                                / temp_df.loc[i - 1, '주말 or 주중 세대수'] * 100, 2)



            temp_df.index = ['주말 / 주중 나누기', '결측값 기준 초과 세대 버림', '이상치 제거 / 선형보간', '선형보간 기준 초과 세대 버림', \
                            '주말, 주중 모두 포함 세대만', '데이터 추가 확인']
            temp_df.columns = ['주말 or 주중 세대수(개)', '이전 단계 대비 감소 비율(%)']

            os.chdir(maindir)
            temp_df.to_excel('info_{}_std_{}.xlsx'.format(currentfolder, stmd))
            print('done, saved as \ninfo_{}_std_{}.xlsx'.format(currentfolder, stmd))

            ### 여름, 겨울 찾아서 지우기

            for folder in os.listdir(finaldir) :
                tempdir = finaldir + '\\' + folder
                if os.path.isdir(tempdir) :
                    for excel in os.listdir(tempdir) :
                        
                        drop_index = []
                        
                        os.chdir(tempdir)
                        temp = lib.read_excel(excel)
                        os.chdir(paradir + '\\' + folder)
                        for log_name in os.listdir(paradir + '\\' + folder) :
                            if excel[: -5] in log_name :
                                log = lib.read_excel(log_name)
            #                     print(log_name)
            #             print(log)
                        
                        if folder == '주중' :
                            folder_name = 'day'
                        else :
                            folder_name = 'end'
                        
                        for i in range(temp.shape[0]) :
                            for j in range(log.shape[0]) :
                                if log.loc[j, '4_linear'] == i :
                                    cnum = log.loc[j, '1_split']
                                    for q in range(ult_para.shape[0]) :
                                        if ult_para.loc[q, 'weekday/end'] == folder_name :
                                            if ult_para.loc[q, 'num'] == cnum :
                                                if ult_para.loc[q, 'drop'] == 'drop' :
                                                    drop_index.append(i)
                        #             print(drop_index)
                        #             print('all {}, drop {}, percentage = {}%'.format(temp.shape[0], len(drop_index), len(drop_index) / temp.shape[0] * 100))

                        
                        drop_info = pd.DataFrame(columns = ['index', 'org_index', 'month', 'day', 'drop', 'drop_percentage(%)'])
                        drop_info.loc[:, 'org_index'] = log.loc[:, '1_split']
                        drop_info.loc[:, 'index'] = log.loc[:, '4_linear']
                        
                        for i in range(drop_info.shape[0]) :
                            for j in range(ult_para.shape[0]) :
                                if ult_para.loc[j, 'weekday/end'] == folder_name :
                                    if ult_para.loc[j, 'num'] == drop_info.loc[i, 'org_index'] :
                                        drop_info.loc[i, 'month'] = ult_para.loc[j, 'month']
                                        drop_info.loc[i, 'day'] = ult_para.loc[j, 'day']
                                        drop_info.loc[i, 'drop'] = ult_para.loc[j, 'drop']
                        
                        drop_info.loc[0, 'drop_percentage(%)'] = len(drop_index) / temp.shape[0] * 100
                        #             print(drop_info)  

                        for i in range(drop_info.shape[0]) :
                            if drop_info.loc[i, 'drop'] == 'drop' :
                                drop_info.loc[i, 'index'] = 'drop'
                                
                        startnum2 = 0
                        for i in range(drop_info.shape[0]) :
                            if drop_info.loc[i, 'index'] != 'drop' :
                                drop_info.loc[i, 'index'] = startnum2
                                startnum2 += 1
                        
                        os.chdir(paraalldir + '\\' + folder)
                        drop_info.to_excel('{}_para_all.xlsx'.format(excel[ : -5]))
                        
                                    
                        temp = temp.drop(drop_index)
                        temp.reset_index(drop = True, inplace = True)

                        os.chdir(sfdir + '\\' + folder)
                        temp.to_excel('{}_봄가을.xlsx'.format(excel[ : -5]))
                        print('{} done'.format(excel), end = '\r')
                        
                        
                        
            print('all done')

            # for excel in os.listdir :



            #  ### 인덱스 넘버를 날짜로 바꾸기
            #             md_index = []
            #             for i in range(ult_para.shape[0]) :
            #                 if ult_para.loc[i, 'drop'] == 'save' :
            #                     month = ult_para.loc[i, 'month']
            #                     day = ult_para.loc[i, 'day']

            #                     month = str(month)
            #                     day = str(day)

            #                     if len(month) == 1 :
            #                         month = '0' + month

            #                     if len(day) == 1 :
            #                         day = '0' + day

            #                     md = month + day
            #                     md_index.append(md)

            #             temp.index = md_index





                                                


        







        
    

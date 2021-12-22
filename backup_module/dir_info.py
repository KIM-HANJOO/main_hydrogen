import os
from pathlib import PathBrowser

class Info() :
	def __init__(self) :
		
		# directories
		self.module_dir = os.getcwd()
		self.main_dir = module_dir.parent
		self.prep_dir = main_dir + '\\1_preprocessing'
		self.model_dir = main_dir + '\\2_model'
		self.cluster_dir = main_dir + '\\0_temp_dir(clustering)'
		
		
		#  make condition table for facilities
		facility_list = ['업무시설', '판매 및 숙박시설', '문화시설', '교육시설']
		self.facility_list = facility_list
		
		# temp1 => conditions included in facility_list[0]
		temp1 = ['출판 영상 방송통신 및 정보서비스업', '금융 및 보험업', '부동산업 및 임대업', \
				'전문 과학 및 기술 서비스업', '사업시설관리 및 사업지원 서비스업']
				
		# temp2 => conditions included in facility_list[1]
		temp2 = ['도매 및 소매업', '숙박 및 음식점업']
		
		# temp3 => conditions included in facility_list[2]
		temp3 = ['예술 스포츠 및 여가관련 서비스업']
		
		# temp4 => conditions included in facility_list[3]
		temp4 = ['교육 서비스업']
		
		# make dataframe for directory table
		facility_df = pd.DataFrame(columns = facility_list)
		
		max_len = 0
		for num in range(len(facility_list)) :
			if max_len < len(locals()['temp{}'.format(num + 1)]) :
				max_len = len(locals()['temp{}'.format(num + 1)])
				
		for i, name in enumerate(facility_list) :
			for j in range(max_len) :
				length = len(locals()['temp{}'.format(i + 1)])
				if j < length :
					facility_df.loc[j, name] = locals()['temp{}'.format(i + 1)][j]
				else :
					facility_df.loc[j, name] = 'empty'


		for i in range(facility_df.shape[0]) :
			for cat in facility_df.columns :
				if str(facility_df.loc[i, cat]) != 'empty' :
					facility_df.loc[i, cat] = model_dir + '\\' + str(facility_df.loc[i, cat])
					

		# preprocessing - sub directories
		sub_directories = ['1_weekends/weekdays', '2_delete_below_missing_data_standard', \
					'3_outliers_deleted', '4_interpolated', '5_delete_below_interpolation_standard',\
					'6_preprocessed', '7_preprocessed(spring_fall)', 'parameters', 'normalized']
		self.condition_sub_dir = sub_directories
					
					

'''
note :

'''

###################################################################################
import os
import dir_info
di = dir_info.Info()
main_dir = di.main_dir
prep_dir = di.prep_dir 					# \1_preprocessing
model_dir = di.model_dir 				# \2_model
module_dir = di.module_dir 				# \module
facility_dir = di.facility_dir			# \facility
plot_dir = di.plot_dir					# \plot
cluster_dir = di.cluster_dir 			# \0_temp_dir(clustering)
facility_list = self.facility_list		#['업무시설', '판매 및 숙박시설', '문화시설', '교육시설']

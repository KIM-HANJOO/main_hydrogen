import os
import shutil

cwd = os.getcwd()
src = cwd + '\\전체 업태'
dst = cwd + '\\INPUT_HERE'


def newfolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print ('Error: Creating directory. ' +  directory)


def newfolderlist(directory, folderlist):
	for i, names in enumerate(folderlist):
		directory_temp = directory + '\\' + names
		try:
			if not os.path.exists(directory_temp):
				os.makedirs(directory_temp)
		except OSError:
			print ('Error: Creating directory. ' +  directory_temp)

def copyfile(src_dir, dst_dir, src_file) :
	src = src_dir + '\\' + src_file
	dst = dst_dir + '\\' + src_file
	shutil.copyfile(src, dst)


folderlist = os.listdir(src)
newfolderlist(dst, folderlist)

for folder in os.listdir(dst) :
		
	src_dir = src + '\\' + folder + '\\0_raw'
	dst_dir = dst + '\\' + folder
	
	
	for excel in os.listdir(src_dir) :
		copyfile(src_dir, dst_dir, excel)
		
	print("{}\t{} done".format(folder, excel), end = '\r')

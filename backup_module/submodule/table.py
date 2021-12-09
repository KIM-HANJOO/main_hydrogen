
#########################################################################################
#########################################################################################
##                                                                                     ##
##                               #### HOW TO IMPORT ####                               ##
##                                                                                     ##
## home desktop | import table                                                         ##
##                                                                                     ##
## import sys
## sys.path.append('C:\\Users\\joo09\\Documents\\GitHub\\LIBRARY')
## import table as tbl
##                                                                                     ##
##                                                                                     ##
##                                                                                     ##
##                           #### HOW TO RE - IMPORT ####                              ##
##                                                                                     ##
## from imp import reload
## reload(tbl)
##                                                                                     ##
#########################################################################################
#########################################################################################

import pandas as pd
import os

def read_excel(excel) :
    df = pd.read_excel(excel)
    if 'Unnamed: 0' in df.columns :
        df.drop('Unnamed: 0', axis = 1, inplace = True)

    if 'Unnamed: 0.1' in df.columns :
        df.drop('Unnamed: 0.1', axis = 1, inplace = True)

    return df

class Table :
    def __init__(self, df, title) :
        self.DataFrame_name = '< {} >'.format(title)
        
        if isinstance(df, pd.DataFrame) :
            temp_df = df.reset_index(drop = True)
        
        else :
            url = 'https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv'
            df = pd.read_csv(url, index_col=0)
            df.columns = df.columns.str.replace(".", "_") 
            temp_df = df.reset_index(drop = True)

        self.DataFrame = temp_df
        # parameters
        
        temp_max = 0
        for col in temp_df.columns :
            for i in range(temp_df.shape[0]) :
                if len(str(temp_df.loc[i, col])) > temp_max :
                    temp_max = len(str(temp_df.loc[i, col]))
        for col in temp_df.columns :
            if len(col) > temp_max :
                temp_max = len(col)
                
        self.columns_length = temp_max            
        
        self.index_length = len(str(df.shape[0]))
        self.index_space = self.columns_length - self.index_length
        self.space = 2
        self.columns = len(df.columns)
        self.total_length = self.index_length + self.index_space + self.columns_length * self.columns + self.space * (self.columns - 1)
        self.fence = '=' * self.total_length
        self.fence2 = ''
        if (self.total_length - len(self.DataFrame_name) - 2) % 2 == 0.5 :
            sep_length = int((self.total_length - len(self.DataFrame_name)) // 2)
            self.fence2 += '=' * sep_length
            self.fence2 += self.DataFrame_name
            self.fence2 += '=' * (sep_length + 1)
            
        else :
            sep_length = int((self.total_length - len(self.DataFrame_name)) / 2)
            self.fence2 += '=' * sep_length
            self.fence2 += self.DataFrame_name
            self.fence2 += '=' * sep_length
        
        self.table = ''
        self.print_list = ['columns_length', 'index_length', 'space', 'index_space', 'total_length']
        self.print_info()
        temp_df = None
        
    def add_print_info(self, add_info, add_position) :
        self.print_list.insert(add_position, add_info)
        
    def print_info(self) :
        max_str = 0
        for string in self.print_list :
            if len(string) > max_str :
                max_str = len(string)
                
        print('<info>')
        for string in self.print_list :
            print(' : {}'.format(string) + ' ' * (max_str - len(string) + 1) + '= {}'.format(eval('self.{}'.format(string))))
        print('\n')

    def set_index_length(self, var) :
        self.index_length = var
        self.total_length = self.index_length + self.index_space + self.columns_length * self.columns + self.space * (self.columns - 1)
        self.fence = '=' * self.total_length
        print('index_length set to {}'.format(var))
        self.print_info()
        
    def set_columns_length(self, var) :
        self.columns_length = var
        self.total_length = self.index_length + self.index_space + self.columns_length * self.columns + self.space * (self.columns - 1)
        self.fence = '=' * self.total_length
        print('columns_length set to {}'.format(var))
        self.print_info()
        
    def set_space(self, var) :
        self.space = var
        self.total_length = self.index_length + self.index_space + self.columns_length * self.columns + self.space * (self.columns - 1)
        self.fence = '=' * self.total_length
        print('space set to {}'.format(var))
        self.print_info()
        
    def make(self) :
        # overdrive
        self.table = ''
        temp = ''
        temp_df = self.DataFrame.copy()
        # add to temp -> add temp to table
        temp += self.fence2 + '\n'
        column_series = ''
        column_series += ' ' * (self.index_length + self.index_space)
        for num_col, col in enumerate(temp_df.columns) :
            if num_col != len(self.DataFrame.columns) :
                column_series += str(col) + ' ' * (self.columns_length - len(str(col)))
                column_series += ' ' * self.space
            else :
                column_series += str(col) + ' ' * (self.columns_length - len(str(col)))
        temp += column_series + '\n'
        temp += self.fence + '\n'
        # add temp, series
        for i in range(temp_df.shape[0]) :
            str_series = ''
            str_series += str(i) + ' ' * (self.index_length - len(str(i))) + ' ' * self.index_space
                
            for num_col, col in enumerate(self.DataFrame.columns) :
                if num_col != len(self.DataFrame.columns) :
                    str_series += str(temp_df.loc[i, col]) + ' ' * (self.columns_length - len(str(temp_df.loc[i, col])))
                    str_series += ' ' * self.space
                else :
                    str_series += str(temp_df.loc[i, col]) + ' ' * (self.columns_length - len(str(temp_df.loc[i, col])))
                    
            str_series += '\n'
            
            temp += str_series
        temp += self.fence
        
        temp_df = None
        self.table = temp
        return temp

    def save(self, directory, filename) :
        os.chdir(directory)
        file = open("{}.txt".format(filename), 'w')
        file.write(self.table)
        file.close()

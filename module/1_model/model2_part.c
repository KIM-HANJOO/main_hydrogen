#include <stdio.h>

float ave(float * list1_pointer, int len){
	float sum = 0.0f ;
	for (int i = 0 ; i < len ; i ++ ) {
		sum += list1_pointer[i] ;
		}
	return sum / len ;
	}

float model2_part(float * temp, int len)
{
	/*
	float list[5] = {1.0, 2.0, 3.0, 4.0, 5.0} ;
	int len = sizeof(list) / sizeof(float) ;
	float ave1 ;
	ave1 = average(list, len) ;
	printf("%f", ave1) ;
	*/
	
	int d = len / 24 ;
	
	float all_list[len] ;
	float all_day[d] ;
	float all_day_ave ;
	
	for (int day = 0 ; day < d; day ++ ){
		int start = 24 * day + 0;
		int end = 24 * day + 23 ;
		
		float temp_day[24] = temp[start : end] ;
		float temp_day_ave = ave(temp_day, 24) ;
		
		all_day[day] = temp_day_ave ;
		
		}
		
	all_day_ave = ave(temp_day, d) ;
	
	for (int i = 0 ; i < len; i++){
		all_list[i] = temp[i] / all_day_ave ;
		}
	
	return all_list ;	
}

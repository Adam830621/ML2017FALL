import numpy as np
import csv 
import sys
# read model
weight = np.load('my_data/model.npy')


TEST_IN = sys.argv[1]
RES = sys.argv[2]
#TEST_IN = 'my_data/test.csv'
#RES = 'result/res.csv'

hour = 9
t_hour = 9 - hour
m = 9*hour
#-------training data------------------
data = []
for i in range(18):
	data.append([])

#----------------------------------------------------
#read file
#----------------------------------------------------
 
n_row = 0
train_in = open('my_data/train.csv', 'r', encoding='big5') 
row = csv.reader(train_in , delimiter=",")
for r in row:
	if n_row != 0:
		for i in range(3,27):
			if r[i] != "NR":
				data[(n_row-1)%18].append( float( r[i] ) )
			else:
				data[(n_row-1)%18].append( float( 0 ) )	
	n_row =n_row+1
train_in.close()
#----------------------------------------------------
#data transform
#----------------------------------------------------

train_x = []
y_hat = []
 
for i in range(12):
	for j in range(471):
		train_x.append( [] )
		for t in range(18):
			for s in range(hour):
				train_x[471*i+j].append( data[t][480*i+j+s] )
		y_hat.append( data[9][480*i+j+hour] )


train_x = np.array(train_x)
y_hat = np.array(y_hat)


#train_x = np.delete(train_x,0,1)
#y_hat = np.reshape(y_hat,(5652,1))


amb = np.copy(train_x[:,0:hour])  #X
ch4 = np.copy(train_x[:,hour:2*hour]) #X
co = np.copy(train_x[:,2*hour:3*hour]) #X
nmhc = np.copy(train_x[:,3*hour:4*hour]) #X
no = np.copy(train_x[:,4*hour:5*hour]) #X
no2 = np.copy(train_x[:,5*hour:6*hour]) 
nox = np.copy(train_x[:,6*hour:7*hour]) #X
o3 = np.copy(train_x[:,7*hour:8*hour])
pm10 = np.copy(train_x[:,8*hour:9*hour])
pm25 = np.copy(train_x[:,9*hour:10*hour])
rh = np.copy(train_x[:,11*hour:12*hour]) #X
so2 = np.copy(train_x[:,12*hour:13*hour]) 
thc = np.copy(train_x[:,13*hour:14*hour]) #X
wd_hr = np.copy(train_x[:,14*hour:15*hour]) #X
wind_dir = np.copy(train_x[:,15*hour:16*hour]) #X
wind_speed = np.copy(train_x[:,16*hour:17*hour]) #X
ws_hr = np.copy(train_x[:,17*hour:18*hour]) #X



train_data = np.column_stack((pm10, pm25, no2, no, so2, o3, pm25**3, pm10**3,pm25*o3,np.ones([len(amb),1])))



nomal = np.zeros(m)
for index in range(m):
    nomal[index] = train_data[:,index].max()



#-------input testing data -----------------------------------------------
test_in = []


for i in range(240):
	test_in.append([])


n_row = 0
text_in = open(TEST_IN, 'r') 
row = csv.reader(text_in , delimiter=",")
for r in row:
    for i in range(2,11):
        if r[i] != "NR":
            test_in[n_row//18].append( float( r[i] ) )
        else:
            test_in[n_row//18].append( float( 0 ) )	
    n_row += 1
    
    
text_in.close()
test_in = np.array(test_in)
############################################################################### 

t_amb 			= np.copy(test_in[:,t_hour:hour+t_hour])  #X
t_ch4 			= np.copy(test_in[:,hour+2*t_hour:2*hour+2*t_hour]) #X
t_co 			= np.copy(test_in[:,2*hour+3*t_hour:3*hour+3*t_hour]) #X
t_nmhc 		= np.copy(test_in[:,3*hour+4*t_hour:4*hour+4*t_hour]) #X
t_no 			= np.copy(test_in[:,4*hour+5*t_hour:5*hour+5*t_hour]) #X
t_no2 			= np.copy(test_in[:,5*hour+6*t_hour:6*hour+6*t_hour]) 
t_nox 			= np.copy(test_in[:,6*hour+7*t_hour:7*hour+7*t_hour]) #X
t_o3 			= np.copy(test_in[:,7*hour+8*t_hour:8*hour+8*t_hour])
t_pm10 		= np.copy(test_in[:,8*hour+9*t_hour:9*hour+9*t_hour])
t_pm25 		= np.copy(test_in[:,9*hour+10*t_hour:10*hour+10*t_hour])
t_rh 			= np.copy(test_in[:,11*hour+12*t_hour:12*hour+12*t_hour]) #X
t_so2 			= np.copy(test_in[:,12*hour+13*t_hour:13*hour+13*t_hour]) 
t_thc 			= np.copy(test_in[:,13*hour+14*t_hour:14*hour+14*t_hour]) #X
t_wd_hr 		= np.copy(test_in[:,14*hour+15*t_hour:15*hour+15*t_hour]) #X
t_wind_dir 	= np.copy(test_in[:,15*hour+16*t_hour:16*hour+16*t_hour]) #X
t_wind_speed 	= np.copy(test_in[:,16*hour+17*t_hour:17*hour+17*t_hour]) #X
t_ws_hr 		= np.copy(test_in[:,17*hour+18*t_hour:18*hour+18*t_hour]) #X
                         
                         




test_data = np.column_stack((t_pm10, t_pm25, t_no2, t_no, t_so2, t_o3, t_pm25**3, t_pm10**3, t_pm25*t_o3, np.ones([len(t_amb),1])))
#test_data = np.column_stack((t_pm10,t_pm25,t_o3,t_wind_dir,t_wind_speed,t_wd_hr,t_ws_hr,t_co,t_pm10**3,t_pm25**3,t_pm25*t_o3,np.ones([len(t_amb),1])))
#test_data = np.column_stack((t_amb, t_ch4, t_co, t_nmhc, t_no, t_rh, t_thc, t_wd_hr, t_wind_dir, 
#                             t_wind_speed, t_ws_hr, t_pm10, t_pm25, t_no2, t_nox, t_so2, t_o3))
#test_data = np.array(t_pm25)

#-------------------------------------------------------------


for index in range(m):
    test_data[:, index] = test_data[:, index] / nomal[index]

#-------------------------------------------------------------------------------

#------------------testing------------------------------------------------------

value = 0
ID = 0
finalString = "id,value\n"
for i in range(240) :
    value = (round(np.dot(test_data[i], weight)))# + bias))
    finalString = finalString + "id_" + str(i) + "," + str(value) + "\n"
f = open(RES, "w")
f.write(finalString)
f.close()
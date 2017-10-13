import csv 
import numpy as np
from numpy.linalg import inv
epoch  = 5000
lamda  = 0.1
learning_rate = 0.001
hour = 9
t_hour = 9 - hour
m = 1*hour #numbers of features * hour

for k in range(4):

    data = []
    for i in range(18):
    	data.append([])
    
    #----------------------------------------------------
    #read file
    #----------------------------------------------------
     
    n_row = 0
    train_in = open('train.csv', 'r', encoding='big5') 
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
    

    
    #train_data = np.column_stack((pm10, pm25, no2, no, so2, o3, pm25**3, pm10**3,pm25*o3,np.ones([len(amb),1])))
    #train_data = np.column_stack((pm10,pm25,o3,wind_dir,wind_speed,wd_hr,ws_hr,co,pm10**3,pm25**3,pm25*o3,np.ones([len(amb),1])))
    #train_data = np.column_stack((amb, ch4, co, nmhc, no, rh, thc, wd_hr, wind_dir, wind_speed, ws_hr, pm10, pm25, no2, nox, so2, o3, np.ones([len(amb),1])))
    train_data = np.column_stack((pm25, np.ones([len(amb),1])))
    '''
    for i in range(471):
        np.delete(train_data,2826,0)
        np.delete(y_hat,2826,0)
    
    rem = []
    
    for i in range(len(train_data[:,0])):
        if -1 in pm25[i,:] or y_hat[i] == -1 :
            rem.append(i)
    
    np.delete(train_data,rem,0)
    np.delete(y_hat,rem,0)
    '''
    
    
    #---- normalize -----
    nomal = np.zeros(m)
    for index in range(m):
        nomal[index] = train_data[:,index].max()
        train_data[:, index] = train_data[:, index] / nomal[index]


    #--------------------


    
    def cost_func(weight, bias, x, y_hat, m):
        hy = np.dot(x, weight) #+ bias
        cost =  (( y_hat - hy ) ** 2).sum() + (lamda ** (k+1)) * (weight ** 2).sum()
        #print ('hy is\n', hy) 
        #print hy - y_hat
        #print ((hy - y_hat) ** 2)  
        #print 'sum is', cost.sum()
        return cost
    
    #-----Initail-------------------
    weight = inv(train_data.T.dot(train_data)).dot(train_data.T).dot(y_hat)
    bias   = 1.0 #np.array(np.ones((1,)), dtype = 'float32') 
    cost = cost_func(weight, bias, train_data, y_hat, m)
    prev_gra_w = np.zeros(len(train_data[0,:]))
    prev_gra_b = 0
    #------GD--------------------
    loss = "epoch,cost\n"
    RMSE = "epoch,error\n"
    for epo in range(epoch):
        hy = np.dot(train_data, weight) #+ bias
        diff = y_hat - hy  
        
        gra_w = (-2.0) * np.dot(train_data.T, diff) + 2 * (lamda ** (k+1)) * weight
        gra_b = (-2.0) * diff.sum()
        prev_gra_w += gra_w**2
        prev_gra_b += gra_b**2
        ada_w = np.sqrt(prev_gra_w)
        ada_b = np.sqrt(prev_gra_b)
        bias = bias - learning_rate * gra_b / ada_b
        weight = weight - learning_rate *gra_w / ada_w
        '''
        bias = bias - learning_rate * (-2.0) * diff.sum()
        
        #print(bias)
        
        for index in range(m):    
            weight[index] = (1 - 2 * (lamda ** (k+1)) * learning_rate) * weight[index] \
                            - learning_rate * (-2.0) * ((diff * train_data[:, index]).sum())
        #weight[index] - 2*(lamda**(k+1)) * learning_rate - learning_rate * gra / ada                     
        '''
        
        cost = cost_func(weight, bias, train_data, y_hat, m)
        #print('k = ', k,'epoch', epo, 'cost is ', cost)
        y = np.zeros(5652)
        error = 0
        #for i in range(5652):
        y = (np.round(np.dot(train_data, weight))) #+ bias))
        error =  ((y_hat - y) ** 2).sum()
        print ('k = ', k,'epoch', epo, 'cost is ', cost, 'error rate is',  ((error/5652.0) ** (0.5))) 
        
        #---------------write cost--------------------
        if k == 0:
            loss = loss + "epo_" + str(epo) + "," + str(cost) + "\n"
            RMSE = RMSE + "epo_" + str(epo) + "," + str((error/5652.0)**(0.5)) +"\n"
            f = open('cost0.csv', "w")
            f2 = open('error0.csv',"w")
            f.write(loss)
            f2.write(RMSE)
            f.close()
            f2.close()
        if k == 1:
            loss = loss + "epo_" + str(epo) + "," + str(cost) + "\n"
            RMSE = RMSE + "epo_" + str(epo) + "," + str((error/5652.0)**(0.5)) +"\n"
            f = open('cost1.csv', "w")
            f2 = open('error1.csv',"w")
            f.write(loss)
            f2.write(RMSE)
            f.close()
            f2.close()
        if k == 2:
            loss = loss + "epo_" + str(epo) + "," + str(cost) + "\n"
            RMSE = RMSE + "epo_" + str(epo) + "," + str((error/5652.0)**(0.5)) +"\n"
            f = open('cost2.csv', "w")
            f2 = open('error2.csv',"w")
            f.write(loss)
            f2.write(RMSE)
            f.close()
            f2.close()
        if k == 3:
            loss = loss + "epo_" + str(epo) + "," + str(cost) + "\n"
            RMSE = RMSE + "epo_" + str(epo) + "," + str((error/5652.0)**(0.5)) +"\n"
            f = open('cost3.csv', "w")
            f2 = open('error3.csv',"w")
            f.write(loss)
            f2.write(RMSE)
            f.close()
            f2.close()
    '''    
    y = np.zeros(5652)
    error = 0
    for i in range(5652):
        y[i] = (round(np.dot(train_data[i], weight) + bias))
        error = error + ((y_hat[i] - y[i]) ** 2) 
    
    print ('error rate is',  ((error/5652.0) ** (0.5))) 
    '''
    
    
    
    
    
    
    
    #-------input testing data -----------------------------------------------
    test_in = []
    
    
    for i in range(240):
    	test_in.append([])
    
    
    n_row = 0
    text_in = open('test.csv', 'r') 
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
                             
                             

    
    
    
    #test_data = np.column_stack((t_pm10, t_pm25, t_no2, t_no, t_so2, t_o3, t_pm25**3, t_pm10**3, t_pm25*t_o3, np.ones([len(t_amb),1])))
    #test_data = np.column_stack((t_pm10,t_pm25,t_o3,t_wind_dir,t_wind_speed,t_wd_hr,t_ws_hr,t_co,t_pm10**3,t_pm25**3,t_pm25*t_o3,np.ones([len(t_amb),1])))
    #test_data = np.column_stack((t_amb, t_ch4, t_co, t_nmhc, t_no, t_rh, t_thc, t_wd_hr, t_wind_dir, t_wind_speed, t_ws_hr, t_pm10, t_pm25, t_no2, t_nox, t_so2, t_o3, np.ones([len(t_amb),1])))
    test_data = np.column_stack((t_pm25, np.ones([len(t_amb),1])))
    
    #-------------------------------------------------------------
    
    
    for index in range(m):
        test_data[:, index] = test_data[:, index] / nomal[index]
    
    #-------------------------------------------------------------------------------
    
    #------------------testing------------------------------------------------------
    if k == 0:
        value = 0
        ID = 0
        finalString = "id,value\n"
        for i in range(240) :
            value = (round(np.dot(test_data[i], weight)))# + bias))
            finalString = finalString + "id_" + str(i) + "," + str(value) + "\n"
        f = open('k0.csv', "w")
        f.write(finalString)
        f.close()
        
    if k == 1:
        value = 0
        ID = 0
        finalString = "id,value\n"
        for i in range(240) :
            value = (round(np.dot(test_data[i], weight)))# + bias))
            finalString = finalString + "id_" + str(i) + "," + str(value) + "\n"
        f = open('k1.csv', "w")
        f.write(finalString)
        f.close()
        
    if k == 2:
        value = 0
        ID = 0
        finalString = "id,value\n"
        for i in range(240) :
            value = (round(np.dot(test_data[i], weight)))# + bias))
            finalString = finalString + "id_" + str(i) + "," + str(value) + "\n"
        f = open('k2.csv', "w")
        f.write(finalString)
        f.close()
    
    if k == 3:
        value = 0
        ID = 0
        finalString = "id,value\n"
        for i in range(240) :
            value = (round(np.dot(test_data[i], weight)))# + bias))
            finalString = finalString + "id_" + str(i) + "," + str(value) + "\n"
        f = open('k3.csv', "w")
        f.write(finalString)
        f.close()
    
    
    
    #-------------------------------------------------------------------------------





















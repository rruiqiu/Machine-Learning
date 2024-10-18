import numpy as np
import matplotlib.pyplot as plt
import math




X_train = np.linspace(0.,1.,201) # training set
X_test = np.linspace(0.,1.,101) # testing set

# X_train_shuffle = np.linspace(0.,1.,201)
# t_train_shuffle = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(201)

np.random.seed(8681)
t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(201)
t_test = np.sin(4*np.pi*X_test) + 0.3 * np.random.randn(101)
t_actual = np.sin(4*np.pi*X_train) 

shuffle_data = np.column_stack((X_train, t_train))
np.random.shuffle(shuffle_data)





#implement k nearest neighbours， k range from 1 to 60 inclusive
k_list = list(range(1,61))
# k = 45
K_shuffle_list = list(range(1,61))

    
def main():
  #generate 60 KNN
  KNN_model()
  #5 fold validation    
  avg_error = []
  avg_train_error = []
  for j in range(len(K_shuffle_list)):
    #splite shuffle for 5 parts, step 50 and train each of them, computer the error of each and find the average of 5, repeat the process with every N value
    # print(len(shuffle_data))
    total_error = 0
    total_train_error = 0
    for i in range(0,len(shuffle_data),40):
      prediction_points = []
      # print(shuffle_data[i:i+50])
      test_set = shuffle_data[i:i+40]
      train_set = np.concatenate((shuffle_data[:i], shuffle_data[i+40:]), axis=0)
      # print(len(test_set))
      # print(len(train_set))
      #now perform the K model selection and also calcualte the training Error
      
      for k in range(len(train_set)):
        prediction_points.append(compute_distance_Shuffle(train_set,K_shuffle_list[j],k))

      #compute the testing points and calculate the error
      KNN = np.column_stack((train_set[:,0],prediction_points))
      total_error = total_error + compute_point_RMSE(KNN,test_set,K_shuffle_list[j])
      total_train_error = total_train_error + compute_train_err(KNN,train_set)
    avg_error.append([K_shuffle_list[j],total_error/5])
    avg_train_error.append([K_shuffle_list[j],total_train_error/5])
  avg_error = np.array(avg_error)
  avg_train_error = np.array(avg_train_error)
  
  plt.figure()
  # plt.scatter(train_set[:,0],prediction_points,color = 'blue',label="Training Points")
  plt.plot(avg_error[:,0],avg_error[:,1],'o-',color = 'blue',label="Validation Error")
  # plt.scatter(avg_error[:,0],avg_error[:,1],color = 'blue')
  plt.plot(avg_train_error[:,0],avg_train_error[:,1],'o-',color = 'orange',label="Training Error")
  # plt.scatter(avg_train_error[:,0],avg_train_error[:,1],color = 'orange')
  # plt.scatter(test_set[:,0],test_set[:,1],color = 'orange',label="Testing Points")
  # plt.plot(X_train,t_actual,color = "red",label="True curve")
  plt.xlabel('k')
  plt.ylabel('Error')
  plt.legend()  
  plt.show()



def compute_train_err(KNN,train_set):
  error = calculate_error(KNN[:,1],train_set[:,1])
  return error
  
def compute_point_RMSE(KNN,test_set,k):
  #input x value, and find y_predict, compare with y_Test
  prediction_points = []
  for i in range(len(test_set)):
    #for each i in test_set, compute the prediction points
    prediction_points.append([test_set[i][0],compute_distance_KNN(KNN,k,i,test_set)])
  prediction_points = np.array(prediction_points)
  error = calculate_error(test_set[:,1],prediction_points[:,1])
  # print(error)
  return error

#calculate MSE
def calculate_error(test_Set,prediction_set):
  prediction_set = prediction_set.reshape(-1,1)
  test_Set = test_Set.reshape(-1,1)
  error = (prediction_set - test_Set)**2
  return np.mean(error)

def calculate_error_RMSE(test_Set,prediction_set):
  prediction_set = prediction_set.reshape(-1,1)
  test_Set = test_Set.reshape(-1,1)
  error = (prediction_set - test_Set)**2
  return np.sqrt(np.mean(error))

def Euclidean_distance(x,x1):
  # return math.sqrt(pow(x - x1,2)+pow(y -y1,2))
  return math.sqrt(pow(x - x1,2))

def compute_distance_KNN(Knn,k,point_index,test_set):
  total_distance = []
  for i in range(len(Knn)):
    #Euclidian distance
    distance = Euclidean_distance(Knn[i][0],test_set[point_index][0])
    total_distance.append([distance,Knn[i][1]])
  #sort the list in ascending order
  total_distance.sort(key=lambda x: x[0])
  total_sum = 0
  for i in range(k):
    total_sum = total_sum + total_distance[i][1]
  return total_sum/k


def compute_distance_Shuffle(train_set,k,point_index):
  total_distance = []
  for i in range(len(train_set)):
    #Euclidian distance
    # distance = math.sqrt(pow(train_set[i][0] - train_set[point_index][0],2)+pow(train_set[i][1] -train_set[point_index][1],2))
    distance = Euclidean_distance(train_set[i][0],train_set[point_index][0])
    total_distance.append([distance,train_set[i][1]])
  #sort the list in ascending order
  total_distance.sort(key=lambda x: x[0])
  total_sum = 0
  for i in range(k):
    total_sum = total_sum + total_distance[i][1]
  return total_sum/k


def KNN_model():
  for j in range(len(k_list)):
    prediction_points = []
    for i in range(len(X_train)):
    # print(X_train[i])
      prediction_points.append(compute_distance(i,k_list[j]))
    if(j % 10 == 0):
      fig,axs = init_plt_5x2()
      add_to_plt(axs,j % 10,prediction_points,j)
    else:
      add_to_plt(axs,j % 10,prediction_points,j)
    if(k_list[j] ==12):
      plt_prediction(prediction_points)
      #calcualte error on test points
      KNN = np.column_stack((X_train,prediction_points))
      test_set = np.column_stack((X_test, t_test))
      error = compute_point_RMSE(KNN,test_set,k_list[j])
      print("testing MSE error is: ",error, "RMSE is:",np.sqrt(error))
# def compute_Test_err(prediction_points):
#   for i in range(len(X_test)):
      
      
def compute_distance(point_index,k):
  total_distance = []
  for i in range(len(X_train)):
    #Euclidian distance
    distance = Euclidean_distance(X_train[i],X_train[point_index])
    
    total_distance.append([distance,t_train[i]])
  #sort the list in ascending order
  total_distance.sort(key=lambda x: x[0])
  total_sum = 0
  for i in range(k):
    total_sum = total_sum + total_distance[i][1]
  return total_sum/k

def plt_prediction(prediction_points):
#function to plot train and test points
  plt.figure()
  plt.scatter(X_train,prediction_points,color = 'blue',label="Prediction")
  plt.scatter(X_train,t_train,color = 'orange',label="Training Points")
  plt.scatter(X_test,t_test,color = "yellow",label="Testing Points")
  plt.plot(X_train,t_actual,color = "red",label="True curve")
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()  

def init_plt_5x2():
  fig, axs = plt.subplots(nrows=5, ncols=2,sharex=False)
  fig.tight_layout(pad=0.5)
  fig.legend()
  return fig,axs

def add_to_plt(axs,M,prediction_points,title):
  row = M // 2
  col = M % 2

  axs[row,col].scatter(X_train,t_train,color = 'orange',label="Training Points")
  axs[row,col].scatter(X_train,prediction_points,color = 'blue',label="prediction Points")
  axs[row,col].plot(X_train,t_actual,color = "red",label="True curve")
  axs[row,col].set_xlabel("x")
  axs[row,col].set_ylabel("y")
  axs[row,col].title.set_text("k = " + str(title+1))
  axs[row,col].legend()

if __name__ == "__main__":
  main()
  


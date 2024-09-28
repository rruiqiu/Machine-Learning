import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = np.linspace(0.,1.,10) # training set
X_valid = np.linspace(0.,1.,100) # validation set

#student number is 400318681

np.random.seed(8681)
t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10)
t_test = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
t_actual = np.sin(4*np.pi*X_valid) 



def matrix_init():
  t_train_vector = np.array([t_train]).T
  X_train_vector = np.array([X_train]).T
  t_valid_vector = np.array([t_valid]).T
  X_valid_vector = np.array([X_valid]).T
  XX_train = sc.fit_transform(X_train_vector)
  XX_valid = sc.transform(X_valid_vector)
  return t_train_vector,X_train_vector,t_valid_vector,X_valid_vector,XX_train,XX_valid
  
RMSE_train_arr = []
RMSE_val_arr = []
#regularization
Error_val_reg_arr = []
RMSE_train_reg_arr = []


def generate_lambdas(min,max):
  lamdas_arr = []
  for i in range(min,max+1):
    lamdas_arr.append(np.exp(i))
  
  return lamdas_arr


def compute_avg():
  error = (t_valid - t_actual)**2
  return np.sqrt(np.mean(error))

def calculate_rmse(predicted_value,actual_value):
  error = (predicted_value - actual_value)**2
  return np.sqrt(np.mean(error))



def compute_loop(X,t,axs):
  first_column_vector = X
  X = np.ones((len(X),1))
  for M in range(len(X)):
  # print(i)
    if(M !=0):
      #append the column to the X for the increased degree
      X = np.hstack((X,first_column_vector**M))
    A = np.dot(X.T,X)
    A1 = np.linalg.inv(A) #inverse the matrix of A
    b = np.dot(X.T,t)
    w = np.dot(A1,b) 
    compute_val(w,M,axs)
    compute_train(w,M)
    if(M==7):
      compute_test_err(w,M)
    # compute_average_test(w,M)

def compute_test_err(w,M):
  f_M = w[0]
  for i in range(1,M+1):
    # print(i)
    f_M = f_M + w[i]* (X_valid**i)
  print(f_M) #(100,1) predicted values
  # add_to_plot(f_M,M,axs)

  print("Test error is: ",calculate_rmse(f_M,t_test))
  plot_test_err(f_M,M)
  # RMSE_val_arr.append(calculate_rmse(f_M,t_valid))



def compute_val(w,M,axs):
  f_M = w[0]
  for i in range(1,M+1):
    # print(i)
    f_M = f_M + w[i]* (X_valid**i)
  print(f_M) #(100,1) predicted values
  add_to_plot(f_M,M,axs)

  RMSE_val_arr.append(calculate_rmse(f_M,t_valid))
  
def compute_train(w,M):
  f_M = w[0]
  for i in range(1,M+1):
    # print(i)
    f_M = f_M + w[i]* (X_train**i)
  # print(f_M) #(100,1) predicted values
  # add_to_plot(f_M,M,axs)

  RMSE_train_arr.append(calculate_rmse(f_M,t_train))  
  
def compute_w_regularization(lambdas,XX_train,t,XX_valid,axs2):
  # B is the (D+1) matrix
  # X = XX_train #reuse the previous one, i am lazy to update the new name
  # first_column_vector = X

  for i in lambdas:
    X = XX_train
    first_column_vector = X
    N = len(X)
    X = np.ones((len(X),1))
    for M in range(len(XX_train)):
      B_dim = M+1
      fill = 2 * i
      B = np.zeros((B_dim,B_dim))
      np.fill_diagonal(B,fill)
      #B_00 = 0
      B[0][0] = 0
      if(M !=0):
        X = np.hstack((X,first_column_vector**M))

      if(M == 9):
        A = np.dot(X.T,X)
        A1 = np.linalg.inv(A+(N/2)*B)
        b = np.dot(X.T,t)
        w = np.dot(A1,b)
        
        
        compute_val_Reg(w,M,XX_valid.T,i,axs2,XX_train)
        compute_train_Reg(w,M,XX_train.T,i) 

#potential XX_valid error?
def compute_val_Reg(w,M,XX_valid,lambdas,axs2,XX_train):
  f_M = w[0]
  for i in range(1,M+1):
    # print(i)
    f_M = f_M + w[i]* (XX_valid**i)
  # print(f_M) #(100,1) predicted values
  lambda_1 = -8
  lambda_2 = 5
  if(lambdas == np.exp(lambda_1)):
    add_to_plot_Reg(f_M,0,axs2,XX_valid,lambda_1,XX_train)
  elif(lambdas == np.exp(lambda_2)):
    add_to_plot_Reg(f_M,1,axs2,XX_valid,lambda_2,XX_train)

  Error_val_reg_arr.append(calculate_error_reg(f_M,t_valid,lambdas,w))

def calculate_error_reg(predicted_value,actual_value,lambdas,w):
  # error = (np.mean((predicted_value - actual_value)**2))
  # regularization_term = lambdas * np.sum(w**2)
  # error = error + regularization_term
  # return error
  error = (predicted_value - actual_value)**2
  return np.sqrt(np.mean(error))



def compute_train_Reg(w,M,XX_train,lambdas):
  f_M = w[0]
  for i in range(1,M+1):
    # print(i)
    f_M = f_M + w[i]* (XX_train**i)
  # print(f_M) #(100,1) predicted values
  # add_to_plot(f_M,M,axs)

  RMSE_train_reg_arr.append(calculate_error_reg(f_M,t_train,lambdas,w))  
 
  

def add_to_plot(f_M,M,axs):
  row = int(M/2)
  col = M%2
  if(M ==0):
    axs[row,col].axhline(f_M,color  ='red',xmin=0.0, xmax=1.0, label='Regression Line')
    axs[row,col].scatter(X_train,t_train)
    axs[row,col].scatter(X_valid,t_valid)
    axs[row,col].plot(X_valid,t_actual,color='green')
    axs[row,col].set_xlabel("x")
    axs[row,col].set_ylabel("y")
    axs[row,col].title.set_text("M = " + str(M))
    axs[row,col].legend()
  else:
    axs[row,col].scatter(X_train,t_train)
    axs[row,col].scatter(X_valid,t_valid)
    axs[row,col].plot(X_valid,f_M,color='red')
    axs[row,col].plot(X_valid,t_actual,color='green')
    axs[row,col].set_ylabel("y")
    axs[row,col].set_xlabel("x")
    axs[row,col].title.set_text("M = " + str(M))
    
def add_to_plot_Reg(f_M,M,axs,XX_valid,lambdas,XX_train):
  #only work on M=9 on different lamda
  col = M%2
  axs[col].scatter(XX_train.T,t_train)
  axs[col].scatter(XX_valid.T,t_valid)
  axs[col].plot(XX_valid[0],f_M[0],color='red',label = "In λ =" +str(lambdas))
  axs[col].plot(XX_valid.T,t_actual,color='green',label = "Actual Function")
  axs[col].set_ylabel("y")
  axs[col].set_xlabel("x")
  axs[col].legend()
  # axs[col].title.set_text("M = " + str(M))    

def init_figure():
  fig, axs = plt.subplots(nrows=5, ncols=2,sharex=False)
  fig.tight_layout(pad=0.5)
  fig.legend()
  return fig,axs

def init_figure_2():
  fig, axs = plt.subplots(nrows=1, ncols=2,sharex=False)
  fig.tight_layout(pad=0.5)
  fig.legend()
  return fig,axs

def plot_test_err(f_M,M):
  plt.figure(5)
  plt.scatter(X_valid,t_test,color = 'orange',label="Test Point")
  plt.plot(X_valid,t_actual,color='green',label="Actual Function")
  plt.plot(X_valid,f_M,label="Predictor Function",color ="red")
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()  
  
def plot():
  plt.figure(3)
  M = np.linspace(0,9,10)
  plt.plot(M,RMSE_train_arr,label="Train_RMSE",color='red')
  plt.plot(M,RMSE_val_arr,label="Validation_RMSE",color='green')
  plt.axhline(compute_avg(),label="Average_RMSE",color='blue')
  plt.xlabel('M')
  plt.ylabel('E_RMS')
  # plt.title('Example Plot with Labels')
  plt.legend()

def plot_lambdas_vs_Errreg(min,max):
  plt.figure(4)
  lambdas_value = np.arange(min,max+1)
  plt.plot(lambdas_value,Error_val_reg_arr,label="E_val",color='red')
  plt.plot(lambdas_value,RMSE_train_reg_arr,label="E_train",color='green')
  plt.xlabel('In λ')
  plt.ylabel('E_RMS')
  plt.legend()



def main():
  fig,axs1 = init_figure()
  fig,axs2 = init_figure_2()
  t_train_vector,X_train_vector,t_valid_vector,X_valid_vector,XX_train,XX_valid = matrix_init()
  compute_loop(X_train_vector,t_train_vector,axs1)
  
  min = -20
  max = 5
  lambdas = generate_lambdas(min,max)
  compute_w_regularization(lambdas,XX_train,t_train_vector,XX_valid,axs2)
  plot()
  plot_lambdas_vs_Errreg(min,max)
  # plot_reg()
  plt.show()
  # plt.savefig('your_plot.png')
  # print(RMSE_train_arr)

 
if __name__ == "__main__":
  main()
  


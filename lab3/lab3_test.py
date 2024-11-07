import matplotlib.pyplot as plt
import numpy as np

MSE_arr = []
centroids_arr = []
def init():
  file_name = r"C:\Users\Richard\Documents\4SL4\Machine-Learning\lab3\duck.jpg"
  image = plt.imread(file_name)
  print(image.shape)
  row_length = image.shape[0]
  col_length = image.shape[1]
  return image,row_length,col_length

def K_means_clustering(image,row_length,col_length,random_arr,k):
  clusters = [[] for _ in range(k)]
  for i in range(row_length):
    for j in range(col_length):
      closest_point_index = calculate_shortest_distance(image[i][j],random_arr)
      clusters[closest_point_index].append(image[i][j])
      
  # print(clusters)
  centroids = []
  for i in range(len(clusters)):
    if(len(clusters[i])==0):
      # clusters[i] = random_arr[i]
      centroids.append(random_arr[i])
    else:
      centroids.append(np.mean(clusters[i],axis=0))
  # print(centroids)
  
  centroids_arr.append(centroids)
  MSE = compute_MSE(centroids,clusters)
  if len(MSE_arr)==0:
    MSE_arr.append(MSE)
    K_means_clustering(image,row_length,col_length,centroids,k)
  elif MSE < np.min(MSE_arr):
    MSE_arr.append(MSE)
    K_means_clustering(image,row_length,col_length,centroids,k)
  else:
    MSE_arr.append(MSE)

def find_clustering():
  mse_arr = np.array(MSE_arr)
  print("MSE is: ", MSE_arr[mse_arr.argmin()])
  print("# iterations", len(centroids_arr))
  return centroids_arr[mse_arr.argmin()]



def reconstruct_image(image,row_length,col_length,centroid):
  new_image = []
  for i in range(row_length):
    row = []
    for j in range(col_length):
      new_image_index = calculate_shortest_distance(image[i][j],centroid)
      row.append(centroid[new_image_index])
    new_image.append(row)
  return new_image
def compute_MSE(centroids,clusters):
  MSE = []
  for i in range(len(centroids)):
    if(len(clusters[i])!=0):
      squared_errors = [(point - centroids[i]) ** 2 for point in clusters[i]]
      mse = np.mean(squared_errors)
      MSE.append(mse)
    
  return sum(MSE)


    
def calculate_shortest_distance(pixel,clusters):
  #use vectorization for fast process
  pixel = np.array(pixel)
  # clusters = np.array(clusters)
  distances = []
  # Iterate through each cluster to compute distances
  for cluster in clusters:
      if np.isnan(cluster).any():  # If the cluster is empty or invalid
          distances.append(float('inf'))  # Assign an infinite distance
      else:
          # Calculate the Euclidean distance between the pixel and the cluster
          distances.append(np.linalg.norm(cluster - pixel))
    # print(distances)
    
  distances = np.array(distances)
  closest_point_index = distances.argmin()
  # closest_point = clusters[closest_point_index]

  return closest_point_index
def main():
  image,row_length,col_length = init()
  #k
  k = 40
  random_rows = np.random.randint(0,row_length,size=(k,1))
  random_cols = np.random.randint(0,col_length,size=(k,1))
  random_arr = np.hstack([random_rows,random_cols])
  random_centers = []
  for i in range(k):
    random_centers.append(image[random_arr[i][0]][random_arr[i][1]])
  random_centers = np.array(random_centers)
  # print("Random array indexes are:")
  # print(random_arr)
  # for i in range(k):
  #   print("Pixels are")
  #   print(image[random_arr[i][0]][random_arr[i][1]])

  random_centers.reshape(k,3)
  K_means_clustering(image,row_length,col_length,random_centers,k)
  centroid = find_clustering()
  centroid = np.array(centroid).astype(int)
  print("Centroid is",centroid)
  
  new_image = reconstruct_image(image,row_length,col_length,centroid)
  
  # print(MSE_arr)
  # print(centroids_arr)
  plt.imshow(new_image)
  plt.axis('off')
  plt.show()
  
  
if __name__ == "__main__":
  main()
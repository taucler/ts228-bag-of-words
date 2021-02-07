
import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import joblib
from scipy.cluster.vq import kmeans, vq



# Chemin vers les images d'entrainement
train_path = './train'
training_names = os.listdir(train_path)


# Stockage des images dans une liste
image_paths = []
image_classes = []
class_id = 0

# Mettre tous les fichiers dans un dossier
def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]



for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1
print(class_path)

# Détection des points clés grâce à la méthode SIFT
des_list = []


for image_path in image_paths:
    im = cv2.imread(image_path)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(im,None)
    des_list.append((image_path,descriptor))
print(des_list)
print("SIFT OK")

# Mettre tous les descripteurs dans un tableau
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
print(descriptors)
#kmeans works only on float, so convert integers to float
descriptors_float = descriptors.astype(float)


# Clustering avec Kmeans
k = 50  #nombre de clusters à modifier
voc, variance = kmeans(descriptors_float, k, 1)
print(variance)
print("Kmeans ok")

# Calculer les données en histogramme(Machine bloque quand j'essaie de tracer les histogrammes)
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1
print("Histogrammes en vecteurs ok")

# Vectorisation
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
print("Vectorisation ok")


stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)
print("Scaling ok")

# Entrainement grâce à SVM
clf = LinearSVC(max_iter=10000)  #Nombre d'itérations modifiable
clf.fit(im_features, np.array(image_classes))
print("Train algorithme ok")


# Enregistrement des données dans un fichier.pkl
joblib.dump((clf, training_names, stdSlr, k, voc), "informations.pkl", compress=3)
print("Enregistrement ok")

import cv2
import numpy as np
import os
import pylab as pl
from sklearn.metrics import confusion_matrix, accuracy_score #sreeni
import joblib
from scipy.cluster.vq import vq


# Récupérer les informations du fichier .pkl
clf, classes_names, stdSlr, k, voc = joblib.load("informations.pkl")
print("Téléchargement ok")

# Chemin vers les images test
test_path = './test'  # En essayant les images train les résultats sont exactes


testing_names = os.listdir(test_path)
print(test_path)

### Code similaire à l'entrainement ###
# Images dans une liste
image_paths = []
image_classes = []
class_id = 0


def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]



for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1
print(class_path)


des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(im,None)
    des_list.append((image_path,descriptor))
print(des_list)
print("SIFT ok")


descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))


test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1


nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')


test_features = stdSlr.transform(test_features)

### Fin ###

# Comparaison
true_class =  [classes_names[i] for i in image_classes]
# Prédictions
predictions =  [classes_names[i] for i in clf.predict(test_features)]


# Affichage dans la console
print ("true_class ="  + str(true_class))
print ("prediction ="  + str(predictions))


# Accuracy Matrice
def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()


accuracy = accuracy_score(true_class, predictions)
print ("accuracy = ", accuracy)
cm = confusion_matrix(true_class, predictions)
print (cm)

showconfusionmatrix(cm)


print ("Image =", image_paths)
print ("prediction ="  + str(predictions))

# Enregistrement des infos dans un fichier .csv
np.savetxt ('data_me.csv', np.transpose([image_paths, predictions]),fmt='%s', delimiter=',', newline='\n')

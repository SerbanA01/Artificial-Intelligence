import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm


train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt', 'int')
test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', 'int') 


class  KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels
    def classify_image(self, test_image, num_neighbors = 3, metric ='l2'):
        if metric == 'l1':
            distances = np.sum(np.abs(self.train_images - test_image),axis = 1)
        elif metric == 'l2':
            distances = np.sqrt(np.sum((self.train_images-test_image)**2,axis = 1))
        sorted_indices = np.argsort(distances)
        nearest_indices = sorted_indices[:num_neighbors]
        nearest_labels =self.train_labels[nearest_indices]
        unique, counts = np.unique(nearest_labels, return_counts=True)
        return unique[np.argmax(counts)]


Kn = KnnClassifier(train_images,train_labels)
ct = 0
for i,image in enumerate(test_images):
    predicted_image = Kn.classify_image(image)
    if predicted_image == test_labels[i]:
        ct+=1

accuracy = ct / len(test_labels)

with open('predictii_3nn_l2_mnist.txt', 'w') as file:
    file.write(str(accuracy))

acc=[]
for nn in [1,3,5,7,9]:
    ct = 0
    for i,image in enumerate(test_images):
        predicted_image = Kn.classify_image(image,nn)
        if predicted_image == test_labels[i]:
            ct+=1
    accuracy = ct / len(test_labels)
    acc.append(accuracy)

plt.plot([1,3,5,7,9], acc)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of neighbors')
plt.show()

acc2=[]
for nn in [1,3,5,7,9]:
    ct = 0
    for i,image in enumerate(test_images):
        predicted_image = Kn.classify_image(image,nn,'l1')
        if predicted_image == test_labels[i]:
            ct+=1
    accuracy = ct / len(test_labels)
    acc2.append(accuracy)

plt.plot([1,3,5,7,9], acc, label='l2')
plt.plot([1,3,5,7,9], acc2, label='l1')

plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of neighbors')
plt.legend()
plt.show()
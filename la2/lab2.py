import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt', 'int')
test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', 'int') 


num_bins = 5

bins = np.linspace(0,255,num_bins)

def values_to_bins(a,bins):
    a = np.digitize(a,bins)
    return a-1

dis_train = values_to_bins(train_images,bins)
dis_test = values_to_bins(test_images,bins)


clf = MultinomialNB()
clf.fit(dis_train,train_labels) #aici il antrenam pe cele de antrenare
predict = clf.predict(dis_test) #aici se calculeaza labels pentru teste
accuracy = accuracy_score(predict,test_labels)  #se calculeaza acuratetea pe cele de testare
print(accuracy)

#
for bin in [3,5,7,9,11]:
    bins = np.linspace(0,255,bin)
    dis2_train = values_to_bins(train_images,bins)
    dis2_test = values_to_bins(test_images,bins)
    clf = MultinomialNB()
    clf.fit(dis2_train,train_labels)
    predict = clf.predict(dis2_test)
    print(accuracy_score(predict,test_labels))

#
print("exercitiul 5")
bins = np.linspace(0,255,7)
dis3_train = values_to_bins(train_images,bins)
dis3_test = values_to_bins(test_images,bins)
clf = MultinomialNB()
clf.fit(dis3_train,train_labels)
predict3=clf.predict(dis3_test)
i = 0
ct = 0
while(True):
    if predict3[i] != test_labels[i]:#inseamna ca a fost misclasificat
        image = train_images[i,:]
        image = np.reshape(image, (28, 28)) 
        plt.imshow(image.astype(np.uint8), cmap='gray')  
        plt.title('Aceasta imagine a fost clasificata ca %d.' % predict3[i])
        plt.show()
        ct+=1
        if ct == 5:
            break
    i+=1

#ex 6
print("exercitiul 6")

def confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true))
    matrix = [[0 for _ in classes] for _ in classes]

    for true, pred in zip(y_true, y_pred):
        matrix[classes.index(true)][classes.index(pred)] += 1

    return matrix

import seaborn as sns

cm = confusion_matrix(test_labels,predict3)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
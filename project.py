from tkinter.constants import BOTH, NO, YES
from keras.datasets import mnist
from keras import layers, models
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
#import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.keras.backend import softmax
from tensorflow.python.keras.saving.save import load_model # modulo per la creazione e gestione della canvas per l'input 

#loading
#(train_X, train_y), (test_X, test_y) = mnist.load_data()

# funzione di creazione del modello di rete neurale 
def create_NN():
    global train_X,train_y, test_X, test_y

    #shape of dataset
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    
    #plotting
    # from matplotlib import pyplot

    # for i in range(9):  
    #     pyplot.subplot(330 + 1 + i)
    #     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))

    # pyplot.show()

    # ridimensiono gli array delle immagini
    train_X = train_X.reshape((60000,28*28))
    test_X = test_X.reshape((10000, 28 * 28))

    # Pixel of the image have to be normalized
    train_X = train_X.astype('float32') / 255
    test_X = test_X.astype('float32') / 255

    # cetegiorizziamo le classi risultanti
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
   
    # creazione della RN
    network = models.Sequential()
    network.add(layers.Dense(500, activation="relu", input_shape = (28*28,)))
    network.add(layers.Dense(250, activation="relu"))
    network.add(layers.Dense(90, activation="relu"))
    network.add(layers.Dense(10, activation="softmax"))

    #addestramento della RN con iperparametri per il tuning
    network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    network.fit(train_X,train_y, epochs=5, batch_size=130, validation_split=0.15)

    #salvo il modello 
    network.save("Mnist-RN-5epoch")

#funziuone per la creazione del modello con rete nuerale convoluzionale 
def create_CNN():
    global train_X,train_y, test_X, test_y

    #shape of dataset
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))

    # ridimensiono gli array delle immagini
    train_X = train_X.reshape((60000,28*28))
    test_X = test_X.reshape((10000, 28 * 28))

    # Pixel of the image have to be normalized
    train_X = train_X.astype('float32') / 255
    test_X = test_X.astype('float32') / 255

    # cetegiorizziamo le classi risultanti
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
   
    # creazione della RN
    network = models.Sequential()
    #todo
    
    #addestramento della RN con iperparametri per il tuning
    network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    network.fit(train_X,train_y, epochs=5, batch_size=130, validation_split=0.15)

    #salvo il modello 
    network.save("Mnist-CNN-5epoch")

#creo il modello di N
#create_NN()

def test_NN():
    #carico il modello
    classifier = load_model("Mnist-RN-5epoch")

    # ridimensiono gli array delle immagini
    train_X = train_X.reshape((60000,28*28))
    test_X = test_X.reshape((10000, 28 * 28))

    # Pixel of the image have to be normalized
    train_X = train_X.astype('float32') / 255
    test_X = test_X.astype('float32') / 255

    # cetegiorizziamo le classi risultanti
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    #Evaluate on the test set
    test_loss, test_acc = classifier.evaluate(test_X, test_y)
    print('Test Accuracy:', test_acc)
    print('Test Loss:', test_loss)

    #effettuo la predizione sul test-set
    pred_y = classifier.predict_classes(test_X)

    #sistemo i valori delle label del test-set per la creazione della matrice di confusione
    rounded_y = np.argmax(test_y, axis=1)

    #stampo la matrice di confusione
    conf_mat = confusion_matrix(y_true=rounded_y, y_pred=pred_y)
    df_conf_mat = pd.DataFrame(conf_mat,range(10), range(10))
    plt.figure(figsize=(10,7))
    sn.heatmap(df_conf_mat, annot=True,fmt="d")
    plt.show()
    #print(conf_mat)


# ------------------------------ gestione dell'input number ------------------------------
def input_canvas():
    #nome della finestra di input
    nameWin = "NN-Predictor"

    drawing=False
    cv2.namedWindow(nameWin)
    black_image = np.zeros((256,256,3),np.uint8)
    ix,iy=-1,-1

    #drawing molto lento e non responsivo
    def draw_line(event,x,y,flags,param):
        global ix,iy,drawing
        if event== cv2.EVENT_LBUTTONDOWN:
            drawing=True
            ix,iy=x,y
            
        elif event==cv2.EVENT_MOUSEMOVE:
            if drawing==True:
                #cv2.circle(black_image,(x,y),5,(255,255,255),-1)
                cv2.line(black_image,(x,y),(x+1,y+1),(255,255,255),14,-1)
                
        elif event==cv2.EVENT_LBUTTONUP:
            drawing = False
            
    cv2.setMouseCallback(nameWin,draw_line)

    while True:
        cv2.imshow(nameWin,black_image)

        #esc button per terminare
        if cv2.waitKey(20)==27:
            break

        #invio per eseguire la predizione sul numero disegnato
        elif cv2.waitKey(20)==13:
            print("predicting...")
            input_img = cv2.resize(black_image,(28,28))
            black_image = input_img
            input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
            #input_img = input_img.reshape(1,28,28,1)
            input_img = input_img.reshape(1,28*28)
            
            
            res = classifier.predict_classes(input_img,1,verbose=0)[0]
            print(str(res))
            #cv2.putText(black_image,text=str(res),org=(205,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2)
        
        #tasto c per eseguire il clean della canvas
        elif cv2.waitKey(20)==ord('c'):
            black_image = np.zeros((256,256,3),np.uint8)
            ix,iy=-1,-1


    cv2.destroyAllWindows()

# ------------------------------ carico immagine con numeri scritti a mano e li tento di indivudare con openCV ------------------------------

#importo il classificatore addestrato con mnist
classifier =  load_model("Mnist-RN-5epoch")

#Apro ogni immagine della directory
import os
directory = "./cvl-strings-train/train/"
correct_digits=0
correct_numbers=0
total_digits=0
total_numbers=0
for filename in os.listdir(directory):
    img_numbers = cv2.imread(os.path.join(directory, filename),0)
    
    total_numbers = total_numbers+1
    y_test_file = filename.split('-')[0]
    total_digits = total_digits+ len(y_test_file)
    
    #print(y_test_file)
    #Aumento il contrasto dell'immagine
    row_idx = 0
    for row in img_numbers:
        pixel_idx = 0
        for pixel in row:
            if pixel < 255:
                img_numbers[row_idx][pixel_idx] = 0
            pixel_idx = pixel_idx+1
        row_idx = row_idx+1

    #cv2.imshow('numeri',img_numbers)
    #creo una soglia sulla quale vengono divisi i valori dei vari pixel
    ret , img_boundary = cv2.threshold(img_numbers,128,255, cv2.THRESH_BINARY)
    #inverto i valori B/N
    img_boundary = cv2.bitwise_not(img_boundary)
    ret , img_boundary = cv2.threshold(img_boundary,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #stampo l'immagine con i 'contorni' dei numeri ottenuti
    #cv2.imshow('Contorni',img_boundary)
    #continuare prendendo il quadrato attorno ai contorni -> https://github.com/BumjunJung9287/Handwritten_Numeral_Recognition_by_CNNmodel-automatic_data_extraction/blob/master/model%2Bmain.ipynb
    #ottengo i vari contorni in base alle soglie costruite sull'immagine
    contours, hierarchy = cv2.findContours(img_boundary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #creo una seconda immagine su cui copiare i numeri individuati
    blank = np.zeros(img_numbers.shape[:2], dtype='uint8')
    #stampo i valori dei contorni ottenuti dall'immagine

    #Vado avanti solo se il numero di cifre trovate è uguale al numero reale di cifre nell'immagine
    print("Numero di contours: "+ str(len(contours)) + " numero di cifre: " + str(len(y_test_file)))
    if(len(contours) == len(y_test_file)):
        #Riordino gli spazi trovati (da sinistra a destra)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        predicted_string = ""
        
        i=0
        for contour in contours:
            area = cv2.contourArea(contour)

            #vado avanti solo per i contorni con certi limiti di grandezza
            #if area > (20*20) and area < (1300*200):
                
            x,y,w,h = cv2.boundingRect(contour)

            blank = np.zeros(img_numbers.shape[:2], dtype='uint8')

            a = max(w,h)
            #print("Contorno in "+str(x)+"-"+str(y)+" di dimensione "+str(w)+"x"+str(h))
            #disegno il rettangolo attorno al numero indiviudato attraverso i boundaries
            #img_numbers = cv2.rectangle(img_numbers, (int(x+w/2-a*1.3/2),int(y+h/2-a*1.3/2)),(int(x+w/2+a*1.3/2),int(y+h/2+a*1.3/2)),(255,0,0),2)
            #estraggo il contenuto del rettangolo in una nuova immagine
            mask = cv2.rectangle(blank, (int(x+w/2-a*1.3/2),int(y+h/2-a*1.3/2)),(int(x+w/2+a*1.3/2),int(y+h/2+a*1.3/2)),(255,0,0),-1)
            blank = cv2.bitwise_and(img_numbers, img_numbers, mask=mask)
            
            numb = blank[y:y+h, x:x+w]

            
            #inverto bianco e nero
            numb = cv2.bitwise_not(numb)
            ret , numb = cv2.threshold(numb,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            #faccio resize dell'immagine per renderla quadrata
            border_numb = cv2.copyMakeBorder(numb,5,5,5,5,cv2.BORDER_CONSTANT,value=[0,0,0])
            border_numb = cv2.resize(border_numb,(28,28))
            
            #cv2.imshow('numeri'+str(i),border_numb )
            border_numb = border_numb.reshape(1,28*28)
            #classifico con il classificatore addestrato su MNIST
            res = classifier.predict_classes(border_numb,1,verbose=0)[0]

            #print('numero'+y_test_file[i]+ ' classificato come: '+ str(res))
            #se la predizione è corretta, la conto
            predicted_string = predicted_string + str(res)
            
            
            if(y_test_file[i] == str(res)):
                correct_digits = correct_digits+1

            i=i+1
            
        if(predicted_string == y_test_file):
            correct_numbers = correct_numbers+1
            
            #cv2.imshow('numeri2',masked)
    
print("Predizioni corrette delle cifre: "+ str(correct_digits) + " su " + str(total_digits))
print("Accuracy cifre: " + str(correct_digits/total_digits))
print("Predizioni corrette dei numeri: "+ str(correct_numbers) + " su " + str(total_numbers))
print("Accuracy cifre: " + str(correct_numbers/total_numbers))

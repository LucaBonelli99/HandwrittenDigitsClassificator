
# def read_images(images_name):
#     #returns an array of flattened images
#     f = open(images_name, "rb")
#     ds_images = []
#     #Let's read the head of the file encoded in 32-bit integers in big-endian(4 bytes)
#     mw_32bit = f.read(4) #magic word
#     n_numbers_32bit = f.read(4) #number of images
#     n_rows_32bit = f.read(4) #number of rows of each image
#     n_columns_32bit = f.read(4) #number of columns of each image

#     #convert it to integers ; '&gt;i' for big endian encoding
#    # mw = struct.unpack("&gt;i",mw_32bit)[0]
#     mw = int.from_bytes(mw_32bit,'big')
#     n_numbers = int.from_bytes(n_numbers_32bit, 'big')
#     n_rows = int.from_bytes(n_rows_32bit, 'big')
#     n_columns = int.from_bytes(n_columns_32bit, 'big')
#     print("num image: "+str(n_numbers))
#     print("num rows: "+str(n_rows))
#     print("num cols: "+str(n_columns))
#     # n_numbers = struct.unpack('&gt;i',n_numbers_32bit)[0]
#     # n_rows = struct.unpack('&gt;i',n_rows_32bit)[0]
#     # n_columns = struct.unpack('&gt;i',n_columns_32bit)[0]

#     try:
#         for i in range(n_numbers):
#             image = []
#             for r in range(n_rows):
#                 for l in range(n_columns):
#                     byte = f.read(1)
#                     #print(byte)
#                     #pixel = struct.unpack('B',byte[0])[0]
#                     pixel = int.from_bytes(byte,'big')
#                     #print(pixel)
#                     image.append(pixel)
#             ds_images.append(image)

#     finally:
#         f.close()

#     return ds_images


# def read_labels(labels_name):
#     #returns an array of labels
#     f = open(labels_name, "rb")
#     ds_labels = []
#     #Let's read the head of the file encoded in 32-bit integers in big-endian(4 bytes)
#     mw_32bit = f.read(4) #magic word
#     n_numbers_32bit = f.read(4) #number of labels

#     #convert it to integers ; '&gt;i' for big endian encoding
#     #mw = struct.unpack('&gt;i',mw_32bit)[0]
#     mw = int.from_bytes(mw_32bit, 'big')
#     #n_numbers = struct.unpack('&gt;i',n_numbers_32bit)[0]
#     n_numbers = int.from_bytes(n_numbers_32bit, 'big')

#     try:
#         for i in range(n_numbers):
#             byte = f.read(1)
#             #label = struct.unpack('B',byte[0])[0]
#             label = int.from_bytes(byte, 'big')
#             ds_labels.append(label)
#     finally:
#         f.close()
    
#     return ds_labels

# def read_dataset(images_name,labels_name):
#     #reads an image-file and a labels file, and returns an array of tuples of
#     #(flattened_image, label)
#     images = read_images(images_name)
#     labels = read_labels(labels_name)
#     #print(str(len(images))+" =?= "+str(len(labels)))
#     assert len(images) == len(labels)
#     return images,labels

# test_X, test_y = read_dataset("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte")
# train_X, train_y = read_dataset("train-images-idx3-ubyte","train-labels-idx1-ubyte")


# fig = plt.figure(figsize=(10,20))
# for i in range(50):
#     sp = fig.add_subplot(10,5,i+1)
#     sp.set_title(trainingset[i][1])
#     plt.axis('off')
#     image = numpy.array(trainingset[i][0]).reshape(28,28)
#     plt.imshow(image,interpolation='none',cmap=pylab.gray(),label=trainingset[i][1])

# plt.show()


# Creazione della canvas per la predizione del numero disegnato
# lastx, lasty = 0, 0
 
 
# def xy(event):
#     "Takes the coordinates of the mouse when you click the mouse"
#     global lastx, lasty
#     lastx, lasty = event.x, event.y
 
 
# def addLine(event):
#     """Creates a line when you drag the mouse
#     from the point where you clicked the mouse to where the mouse is now"""
#     global lastx, lasty
#     canvas.create_line((lastx, lasty, event.x, event.y))
#     # this makes the new starting point of the drawing
#     lastx, lasty = event.x, event.y
 
# def paint(event):
#     color="black"
#     x1,y1 = (event.x-1),(event.y-1)
#     x2,y2 = (event.x+1),(event.y+1)
#     #canvas.create_line(x1,y1,x2,y2, fill="red",outline=color, width=400)
#     canvas.create_rectangle(x1,y1,x2,y2, fill=color,outline=color)

# def mmove(event):
#     print(event.x, event.y)

# root = tk.Tk()
# root.title("Number's Input")
# root.columnconfigure(0, weight=1)
# root.rowconfigure(0, weight=1)

# canvas = tk.Canvas(root,width=280,height=280,bg="white")

# canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
# canvas.bind("<Button-1>", xy)
# canvas.pack()
# canvas.bind("<B1-Motion>", addLine)
# #canvas.bind('<Motion>', mmove)
 
# root.mainloop()

from tkinter.constants import BOTH, NO, YES
from keras.datasets import mnist
from keras import layers, models
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
import tkinter as tk

from tensorflow.python.keras.backend import softmax
from tensorflow.python.keras.saving.save import load_model # modulo per la creazione e gestione della canvas per l'input 

#loading
(train_X, train_y), (test_X, test_y) = mnist.load_data()

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
network.add(layers.Dense(784, activation="relu", input_shape = (28*28,)))
network.add(layers.Dense(500, activation="relu"))
network.add(layers.Dense(250, activation="relu"))
network.add(layers.Dense(90, activation="relu"))
network.add(layers.Dense(10, activation="softmax"))

#addestramento della RN con iperparametri per il tuning
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
network.fit(train_X,train_y, epochs=5, batch_size=130, validation_split=0.1)

#Evaluate on the test set
test_loss, test_acc = network.evaluate(test_X, test_y)
print('Test Accuracy:', test_acc)
print('Test Loss:', test_loss)

#salvo il modello 
network.save("Mnist-RN-5epoch")
print("salvato il modello")

#carico il modello
classifier =  load_model("Mnist-RN-5epoch")

# gestione dell'input number
import numpy as np
import cv2

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
            cv2.line(black_image,(x,y),(x+1,y+1),(255,255,255),5,-1)
            
    elif event==cv2.EVENT_LBUTTONUP:
        drawing = False
        
cv2.setMouseCallback(nameWin,draw_line)

while True:
    cv2.imshow(nameWin,black_image)

    #esc button per terminare
    if cv2.waitKey(1)==27:
        break

    #invio per eseguire la predizione sul numero disegnato
    elif cv2.waitKey(1)==13:
        print("predicting...")
        input_img = cv2.resize(black_image,(28,28))
        input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
        #input_img = input_img.reshape(1,28,28,1)
        input_img = input_img.reshape(1,28*28)
        
        res = classifier.predict_classes(input_img,1,verbose=0)[0]
        print(str(res))
        #cv2.putText(black_image,text=str(res),org=(205,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2)
    
    #tasto c per eseguire il clean della canvas
    elif cv2.waitKey(1)==ord('c'):
        black_image = np.zeros((256,256,3),np.uint8)
        ix,iy=-1,-1

cv2.destroyAllWindows()
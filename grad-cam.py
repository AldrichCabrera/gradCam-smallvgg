from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential, load_model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import imutils
import h5py
import pickle

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='conv2d_3'):
    input_img = model.input
    #layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    #print("layer_dict", layer_dict)
    layer_output = layer_dict[activation_layer].output
    #print("layer_output", layer_output)
    max_output = K.max(layer_output, axis=3)
    #print("max_output", max_output)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        #new_model = VGG16(weights='imagenet')
        new_model = load_model('.../gradCam-smallvgg/pokedex.model')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 5
    #nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)

    x = input_model.output
    #x = input_model.layers[-1].output
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(x)

    #model = keras.models.Model(input_model.layers[0].input, x)
    model = keras.models.Model(inputs=input_model.input, outputs=x)
    
    #loss = K.sum(model.layers[-1].output)
    loss = K.sum(model.output)

    #compute_heatmap  
    # For VGG16
    #conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    #conv_output =  [l for l in model.layers if l.name is layer_name][0].output
    
    conv_output = model.get_layer(layer_name).output
    #print("conv_output", conv_output)

    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    #cam = cv2.resize(cam, (224, 224)) # For VGG16
    cam = cv2.resize(cam, (96, 96))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)

    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    return np.uint8(cam), heatmap

img_path = sys.argv[1]
orig = cv2.imread(img_path)
img = cv2.resize(orig, (96, 96))
#img = cv2.resize(orig, (224, 224)) # For VGG16
img = img.astype("float") / 255.0
#img = image.load_img(img_path, target_size=(224, 224)) # For VGG16
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
#x = preprocess_input(x) 

#model = VGG16(weights='imagenet') # For VGG16

print("[INFO] loading network...")
model = load_model('.../gradCam-smallvgg/pokedex.model')
lb = pickle.loads(open(".../gradCam-smallvgg/lb.pickle", "rb").read())
model.summary()

# classify the input image
print("[INFO] classifying image...")
predictions = model.predict(x)#[0]

idx = np.argmax(predictions)
label = lb.classes_[idx]
label = "{}: {:.2f}%".format(label, predictions[0][idx] * 100)
#print("idx", idx)
print("label", label)

# For VGG16
#top_1 = decode_predictions(predictions)
#(imagenetID, label, prob) = top_1[0][0]
#label = "{}: {:.2f}%".format(label, prob * 100)
#print("[INFO] {}".format(label))
#predicted_class = np.argmax(predictions)

our, heatmap = grad_cam(model, x, idx, "dropout_3")
#our, heatmap = grad_cam(model, x, predicted_class, "block5_conv3") # For VGG16
heatmap2 = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
our = cv2.resize(our, (orig.shape[1], orig.shape[0]))
cv2.rectangle(our, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(our, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')
saliency_fn = compile_saliency_function(guided_model, "dropout_3")
saliency = saliency_fn([x, 0])
#gradcam = saliency[0] * heatmap[..., np.newaxis] # For VGG16
gradcam = saliency[0] * heatmap[np.newaxis, ...]
gradcam = deprocess_image(gradcam)
gradcam = cv2.resize(gradcam, (orig.shape[1], orig.shape[0]))
#cv2.imwrite("guided_gradcam1.jpg", deprocess_image(gradcam))

our = np.vstack([orig, our, gradcam])
our = imutils.resize(our, height=900)
cv2.imshow("gradcam", our)
cv2.waitKey(0)
cv2.imwrite("dropout_3.jpg", our)
import os
import sys
import json
import numpy as np
import skimage.draw
import imgaug
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import matplotlib.patches as patches
from skimage.measure import find_contours
from tensorflow.keras.models import load_model
import random
from imgaug import augmenters as iaa
import re
from skimage.transform import resize
import os
from skimage import io, color
from joblib import Parallel, delayed
import time
import concurrent.futures
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import shutil
from skimage.filters import threshold_otsu
import warnings
from tensorflow.keras.callbacks import ReduceLROnPlateau
import imgaug.augmenters as iaa
# Configurar para usar GPU específica
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Usar GPU 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suprime advertencias de TensorFlow


warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image")

# Root directory of the project
ROOT_DIR = "/home/angarcia/datos/data/hackaton/tf24/"

# Definir las rutas de las imágenes
image_dir_train = '/home/angarcia/datos/data/hackaton/5r_premio_cervera/DATASET_5R/IMAGES/TRAIN'
#image_dir_train = '/home/angarcia/datos/data/hackaton/5r_premio_cervera/DATASET_5R/IMAGES/TRAIN_pruebas'
image_dir_val = '/home/angarcia/datos/data/hackaton/5r_premio_cervera/DATASET_5R/IMAGES/VAL'
#image_dir_val = '/home/angarcia/datos/data/hackaton/5r_premio_cervera/DATASET_5R/IMAGES/VAL'_pruebas'

image_dir_label = "/home/angarcia/datos/data/hackaton/5r_premio_cervera/DATASET_5R/LABELS/LABELME"
evaluacion = "VAL"   


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Ensure the Mask RCNN library is in the system path
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.model import log

class CustomConfig(Config):
    NAME = "orkli"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + tool
    STEPS_PER_EPOCH = 300
    DETECTION_MIN_CONFIDENCE = 0.1
    LEARNING_RATE = 0.001
    GPU_COUNT = 1

    DETECTION_MAX_INSTANCES = 235 #por default es 35
    
    #IMAGE_RESIZE_MODE = "crop"
    IMAGE_RESIZE_MODE = "pad64"

    IMAGE_MIN_DIM = 1280
    IMAGE_MAX_DIM = 2304
    
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128) #parece que 8 empieza a capturar piezas. 16 un poco peor pero captura algunas piezas
    RPN_ANCHOR_SCALES = (24, 48, 64, 88, 128) #

    
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.87 # aumentar esto ya que los objetos estan muy pegados entre si 
    DETECTION_NMS_THRESHOLD = 0.87 # antes estaba  a0.3
    
    
    TRAIN_ROIS_PER_IMAGE = 230  
    ROI_POSITIVE_RATIO = 0.96
    RPN_ANCHOR_RATIOS = [0.76, 1, 1.5]
    
    
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 300 #antes estaba a 256. Vamos a probar a x3 una epoca. IoU anterior: 0.441
    
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    # Aumentar: Retiene más ROIs antes de aplicar NMS, mejorando la detección de objetos difíciles, pero incrementa la carga computacional.
    # Decrecer: Retiene menos ROIs antes de aplicar NMS, reduciendo la carga computacional, pero puede perder algunas detecciones de objetos difíciles.
    PRE_NMS_LIMIT = 8000 #original 6000
    #PRE_NMS_LIMIT = 7000
    # ROIs kept after non-maximum suppression (training and inference)
    # Aumentar: Retiene más ROIs después de NMS durante el entrenamiento, proporcionando más datos de entrenamiento, pero incrementa la carga computacional.
    # Decrecer: Retiene menos ROIs después de NMS durante el entrenamiento, reduciendo la carga computacional, pero puede proporcionar menos datos de entrenamiento.
    POST_NMS_ROIS_TRAINING = 8000 #original 2000
    #POST_NMS_ROIS_TRAINING = 5000
    
    #lo mismo que el de arriba pero solo para la inferencia
    POST_NMS_ROIS_INFERENCE = 10000 #original 1000
    
    
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 250 #original 100
    
    A = 0.6
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2]) * A
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2]) * A



    
# Configuración para inferencia
class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.75
    
    

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset, image_dir):
        self.add_class("object", 1, "tool")

        # Determinar el assert correcto basado en el directorio
        if image_dir.endswith("TRAIN_pruebas") or image_dir.endswith("VAL_pruebas"):
            assert subset in ["TRAIN_pruebas", "VAL_pruebas"]
        else:
            assert subset in ["TRAIN", "VAL"]

        dataset_dir = os.path.join(dataset_dir, subset)

        for json_filename in os.listdir(dataset_dir):
            if json_filename.endswith('.json'):
                json_file_path = os.path.join(dataset_dir, json_filename)
                with open(json_file_path) as f:
                    data = json.load(f)
                filename = json_filename.replace('.json', '.png')
                image_path = os.path.join(image_dir, filename)
                if not os.path.exists(image_path):
                    continue
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                polygons = [r['points'] for r in data['shapes']]
                num_ids = [1] * len(polygons)
                self.add_image(
                    "object",  
                    image_id=filename,
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons,
                    num_ids=num_ids
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(CustomDataset, self).load_mask(image_id)

        num_objs = len(image_info["polygons"])
        masks = np.zeros((image_info["height"], image_info["width"], num_objs), dtype=np.uint8)
        class_ids = []
        for i, p in enumerate(image_info["polygons"]):
            rr, cc = skimage.draw.polygon([point[1] for point in p], [point[0] for point in p])
            masks[rr, cc, i] = 1
            class_ids.append(1)
        return masks.astype(bool), np.array(class_ids, dtype=np.int32)




def train(model):
    
    # Suprimir los UserWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
    
        # print("Preparando los datasets para el entrenamiento...")
        dataset_train = CustomDataset()
        dataset_train.load_custom(image_dir_label, "TRAIN", image_dir_train)
        dataset_train.prepare()
    
        dataset_val = CustomDataset()
        dataset_val.load_custom(image_dir_label, "VAL", image_dir_val)
        dataset_val.prepare()
        
        print("Datasets preparados...")
        
        # Mostrar el dispositivo de entrenamiento
        device_name = tf.test.gpu_device_name()
        if device_name != '':
            print("Entrenando en el dispositivo GPU:", device_name)
        else:
            print("Entrenando en el dispositivo CPU")
        
        # Define una secuencia de augmentations
        augmentation = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Fliplr(1.0)),  # Aplica flip horizontal con una probabilidad del 50%
            iaa.Affine(rotate=(-45, 45)),  # Rotar aleatoriamente entre -45 y 45 grados
            iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.3, 0), pad_mode="constant", pad_cval=(0, 255)))
        ])
        
        # Definir el callback de ReduceLROnPlateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_mrcnn_mask_loss', 
            factor=0.1, 
            patience=10, 
            min_lr=1e-6, 
            verbose=1
        )
        
        # Entrenar el modelo con el callback de ReduceLROnPlateau
        model.train(
            dataset_train, 
            dataset_val, 
            learning_rate=CustomConfig.LEARNING_RATE, 
            epochs=120, 
            layers='all',
            augmentation=augmentation,
            custom_callbacks=[reduce_lr]
        )        
        
        # Guarda el modelo después de completar todas las épocas de entrenamiento
        # model.keras_model.save(ruta_modelo)
        # print(f"Modelo guardado en: {ruta_modelo}")



###########################################################################################################################
#
#                   EVALUACION Y OBTENCION DE PREDICCIONES
#
############################################################################################################################   
          
def display_masks_only(image, boxes, masks, class_ids, class_names, scores=None, title="", figsize=(16, 16), ax=None, show_mask=True, colors=None, captions=None, save_path=None, iteration=1):

    N = masks.shape[-1]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Si no se provee un eje, crea uno y configura automáticamente su tamaño
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
    
    # Muestra la imagen
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i] if colors is not None else visualize.random_colors(N)[i]
        # Máscara
        mask = masks[:, :, i]
        masked_image = visualize.apply_mask(masked_image, mask, color, alpha=0.5)

    # Muestra la imagen
    ax.imshow(masked_image.astype(np.uint8))

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # Construye el nombre de archivo y guarda la imagen en la ruta especificada
        file_name = f"masked_image_{title.replace(' ', '_')}_{iteration}.png"
        full_path = os.path.join(save_path, file_name)
        plt.savefig(full_path)
        print(f"Imagen guardada en: {full_path}")

    plt.show() 


def evaluate_model(dataset, model, inference_config, output_dir):
    # Obtener las rutas de las imágenes desde el dataset
    image_paths = [dataset.image_info[i]["path"] for i in range(len(dataset.image_info))]
    
    print(f"Evaluando {len(image_paths)} imágenes del dataset.")
    
    target_size=(1242, 2204)



    for image_path in image_paths:
        image = skimage.io.imread(image_path)
        # Convertir a RGB si es necesario
        if image.shape[-1] == 4:
            image = image[..., :3]
        
        # Ajustar el tamaño de la imagen (opcionalmente, si es necesario)
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=inference_config.IMAGE_MIN_DIM,
            min_scale=inference_config.IMAGE_MIN_SCALE,
            max_dim=inference_config.IMAGE_MAX_DIM,
            mode=inference_config.IMAGE_RESIZE_MODE)

        results = model.detect([image], verbose=0)[0]

        # Crear carpeta para la imagen actual
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Quitar "_defaultImage" si existe
        if image_name.endswith("_defaultImage"):
            image_name = image_name[:-13]
        
        output_image_dir = os.path.join(output_dir, image_name)
        
        # Eliminar el contenido de la carpeta si ya existe
        if os.path.exists(output_image_dir):
            shutil.rmtree(output_image_dir)
    
        os.makedirs(output_image_dir, exist_ok=True)

        for i in range(results['masks'].shape[2]):
            mask = results['masks'][:, :, i]
            
            # Redimensionar la máscara a target_size
            resized_mask = skimage.transform.resize(mask, target_size, order=0, preserve_range=True, anti_aliasing=False)
            binary_mask = (resized_mask > 0.5).astype(np.uint8)
            mask_path = os.path.join(output_image_dir, f"mask_{i+1}.png")
            skimage.io.imsave(mask_path, binary_mask * 255)  # Guardar como imagen binaria
        #Aqui se veran las imagenes
        save_path = '/home/angarcia/datos/data/hackaton/imagenes_predichas'

        display_masks_only(image, results['rois'], results['masks'], results['class_ids'], dataset.class_names, scores=results['scores'], title="Predictions", save_path=save_path, iteration=i+1)
        print(f"Number of masks detected in {os.path.basename(image_path)}: {results['masks'].shape[-1]}")



###########################################################################################################################
#
#                   CALCULO DEL IOU
#
###########################################################################################################################

#pierde un pelin pero va mucho mejor mas rapido (15 veces mas rapido)
def resize_mask(mask, new_shape):
    """Redimensiona una máscara a una nueva forma utilizando NumPy."""
    zoom_factors = (new_shape[0] / mask.shape[0], new_shape[1] / mask.shape[1])
    rows = np.round(np.arange(0, mask.shape[0] * zoom_factors[0]) / zoom_factors[0]).astype(int)
    cols = np.round(np.arange(0, mask.shape[1] * zoom_factors[1]) / zoom_factors[1]).astype(int)
    return mask[rows[:, np.newaxis], cols].astype(bool)


def umbralizar_mascara(mask, mask_name):
    """Umbraliza una máscara utilizando el umbral de Otsu."""
    unique_values = np.unique(mask)
    if len(unique_values) == 1:
        #print(f"Máscara {mask_name} con un solo valor único: {unique_values[0]}.")
        return np.zeros_like(mask) if unique_values[0] == 0 else np.ones_like(mask)
    else:
        # Si tiene más de un valor, aplicar umbralización de Otsu
        threshold = threshold_otsu(mask)
        binary_mask = mask > threshold
        return binary_mask


def mostrar_mascaras(mask_label, mask_pred, iou):
    """Muestra las máscaras de etiquetas y predichas utilizando matplotlib."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask_label, cmap='gray')
    axes[0].set_title('Máscara Label')
    axes[1].imshow(mask_pred, cmap='gray')
    axes[1].set_title('Máscara Predicha')
    axes[2].imshow(mask_label, cmap='gray', alpha=0.5)
    axes[2].imshow(mask_pred, cmap='jet', alpha=0.5)
    axes[2].set_title(f'IoU: {iou:.4f}')
    plt.show()


def calcular_iou(mask1, mask2):
    """Calcula el IoU entre dos máscaras binarias."""
        
    if mask1.shape != mask2.shape:
        mask2 = resize_mask(mask2, mask1.shape)
        
            
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    

    return iou

def procesar_imagen(imagen, dir_IoU, dir_labels):
    """Procesa una imagen y calcula el IoU."""
    # Quitar prefijo "mask_" si existe
    if imagen.startswith("mask_"):
        imagen_sin_prefijo = imagen[5:]
    else:
        imagen_sin_prefijo = imagen

    # Obtener listas de máscaras para la imagen actual
    dir_label_image = os.path.join(dir_labels, imagen)
    dir_pred_image = os.path.join(dir_IoU, imagen_sin_prefijo)



    if not os.path.exists(dir_pred_image):
        print(f"No se encontraron máscaras predichas para {imagen}. Saltando esta imagen.")
        return 0, 1  # iou_imagen, skip_count

    mascaras_label_files = [mask for mask in os.listdir(dir_label_image) if re.search(r'\d+\.(png|jpg|jpeg)$', mask)]
    mascaras_pred_files = [mask for mask in os.listdir(dir_pred_image)]

    mascaras_label = [skimage.io.imread(os.path.join(dir_label_image, mask)) for mask in mascaras_label_files]
    mascaras_pred = [skimage.io.imread(os.path.join(dir_pred_image, mask)) for mask in mascaras_pred_files]
    
    mascaras_label = [mask[..., 0] if mask.ndim == 3 else mask for mask in mascaras_label]
    mascaras_pred = [mask[..., 0] if mask.ndim == 3 else mask for mask in mascaras_pred]
    
    # Umbralizar máscaras label
    mascaras_label = [umbralizar_mascara(mask, mascaras_label_files[idx]) for idx, mask in enumerate(mascaras_label)]


    iou_imagen = 0
    
    for mask_label_idx, mask_label in enumerate(mascaras_label):
        mejor_iou = 0
        mejor_mask_pred_idx = -1
        mejor_mask_pred = None
        for idx, mask_pred in enumerate(mascaras_pred):

            iou = calcular_iou(mask_label, mask_pred)
            #mostrar_mascaras(mask_label, mask_pred, iou)

            if iou > mejor_iou:
                mejor_iou = iou
                mejor_mask_pred_idx = idx
                mejor_mask_pred = mask_pred

        # Eliminar la máscara predicha si el IoU es mayor que 0.5 y mostrar las máscaras
        if mejor_iou > 0.5:
            # mostrar_mascaras(mask_label, mejor_mask_pred, mejor_iou)
            mascaras_pred.pop(mejor_mask_pred_idx)
        else:
            mejor_iou = 0  # Si no se encuentra ninguna máscara con IoU > 0.5, el IoU es 0

        iou_imagen += mejor_iou

    if mascaras_label:
        iou_imagen /= len(mascaras_label)

    print(f"IoU para la imagen {imagen}: {iou_imagen}")
    return iou_imagen, 1
    


def Obtener_IoU(dir_IoU, dir_labels, max_workers=4):
    """Calcula el IoU promedio entre máscaras predichas y etiquetas."""
    imagenes_labels = [name for name in os.listdir(dir_labels) if os.path.isdir(os.path.join(dir_labels, name))]
    num_imagenes_inicial = len(imagenes_labels)

    iou_total = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_imagen = {executor.submit(procesar_imagen, imagen, dir_IoU, dir_labels): imagen for imagen in imagenes_labels}
        for future in concurrent.futures.as_completed(future_to_imagen):
            iou_imagen, _ = future.result()  # Extrae el primer elemento de la tupla
            iou_total += iou_imagen

    if num_imagenes_inicial > 0:
        iou_promedio = iou_total / num_imagenes_inicial
    else:
        iou_promedio = 0

    print(f"Total de imágenes procesadas: {num_imagenes_inicial}")
    print(f"IoU promedio: {iou_promedio}")
    return iou_promedio








if __name__ == "__main__":
    
    use_gpu = False  # Set this to False to use CPU
    #use_gpu = True  # Set this to False to use CPU


    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Fuerza el uso de CPU
        print("Usando CPU")
    else:
        device_name = tf.test.gpu_device_name()
        if not device_name:
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))
        print("Versión de CUDA:", tf.sysconfig.get_build_info()["cuda_version"])
        print("Versión de cuDNN:", tf.sysconfig.get_build_info()["cudnn_version"])


    config = CustomConfig() 
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

    if not os.path.exists(COCO_WEIGHTS_PATH):
        utils.download_trained_weights(COCO_WEIGHTS_PATH)

    #Por si queremos comenzar con los pesos de COCO, pero estos tienen un POOL_SIZE Y MASK_POOL_SIZE predeterminados
    #loa cuales coinciden con MASK_SHAPE = [28, 28], por lo que si queremos cambair estos parametros deberemos entrenar desde 0

    print("Cargando Pesos de Coco")
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    
    #Esto es para continuar entrenamientos (si queremos continuar por una epoca especifica por ejemplo)
    model_path = "orkli20240527T0920/mask_rcnn_orkli_0004.h5" #0.28 iou. FIN

    print("COmenzando entrenamiento!")
    #train(model)
    print("Entrenamiento finalizado!")

     
    # Preparar configuración de inferencia y modelo para evaluación
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=DEFAULT_LOGS_DIR)

    
    # Cargar los pesos del modelo entrenado
    #model_path = "orkli20240520T1117/mask_rcnn_orkli_.h5"
    model_path = os.path.join(DEFAULT_LOGS_DIR, model_path)
    
    
    model.load_weights(model_path, by_name=True)


    if evaluacion in ["VAL", "VAL_pruebas"]:
        image_dir = image_dir_val
    elif evaluacion in ["TRAIN", "TRAIN_pruebas"]:
        image_dir = image_dir_train



    # Cargar dataset para usar class_names en visualización
    dataset_eval = CustomDataset()
    dataset_eval.load_custom(image_dir_label, evaluacion, image_dir)
    dataset_eval.prepare()



    dir_IoU = os.path.join("/home/angarcia/datos/data/hackaton/Calcular_IoU/Mascaras_Predichas", evaluacion)
    dir_labels = os.path.join("/home/angarcia/datos/data/hackaton/Calcular_IoU/Mascaras_Label", evaluacion)

    # Evaluación del modelo
    evaluate_model(dataset_eval, model, inference_config, dir_IoU)

    # Medir el tiempo de ejecución
    start_time = time.time()
    iou_promedio = Obtener_IoU(dir_IoU, dir_labels, max_workers=5)  # cuidado no explote la cpu
    end_time = time.time()
    
    # Calcular el tiempo transcurrido
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos") 
    

    
    
    
    
    
    
    
    
    
    

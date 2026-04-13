
# Importar librerías
from numpy import rint
from ultralytics import YOLO
import os
import torch
import glob
import matplotlib.pyplot as plt
import shutil

# ============================================================================
# CELDA 1: CONFIGURACIÓN INICIAL
# ============================================================================

print("🔍 Verificando GPU...")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ GPU no disponible. El entrenamiento será MUY lento.")

# ============================================================================
# CELDA 2: VERIFICAR DIRECTORIOS LOCALES DE DATOS Y MODELOS
# ============================================================================

data_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', '5_Fold_CV_Dataset_Physical'))
models_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

os.makedirs(models_base_path, exist_ok=True)

if os.path.exists(data_base_path):
    print(f"✅ Dataset encontrado correctamente en:\n   {data_base_path}")
else:
    print(f"❌ Error: La carpeta de datos no existe en {data_base_path}")

print(f"\n📂 Los modelos resultantes se guardarán en:\n   {models_base_path}")

# ============================================================================
# CELDA 4: CREAR ARCHIVOS YAML PARA CADA FOLD
# ============================================================================

folds = ['fold_0']

for fold in folds:
    yaml_path = os.path.join(data_base_path, fold, 'data.yaml')
    fold_path = os.path.join(data_base_path, fold)


    yaml_content = f"""# Configuración del dataset - {fold}
path: {fold_path}
train: train/images
val: valid/images

# Clases
nc: 1
names:
  0: huevo
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✅ {yaml_path} actualizado de colab a ruta local")

MODEL_SIZE = 'yolov8n.pt' 
EPOCHS = 30
BATCH_SIZE = 4  
IMG_SIZE = 640

folds = ['fold_0', 'fold_1'] 
results_all = []

for fold in folds:
    print(f"\n🚀 Entrenando {fold}...\n")

    data_yaml = os.path.join(data_base_path, fold, 'data.yaml')

    model = YOLO(MODEL_SIZE)

    train_project_path = os.path.join(models_base_path, 'train_runs')

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=f'train_{fold}',
        project=train_project_path,
        patience=50,
        save=True,
        plots=True,
        device=0,
        workers=2,
        degrees=45.0,  
        scale=0.6,     
        fliplr=0.5,    
        flipud=0.2    
    )

    results_all.append(results)

print("\n✅ Entrenamiento completado!")


# ============================================================================
# CELDA 6: VISUALIZAR RESULTADOS DE LOS FOLDS 
# ============================================================================

folds = ['fold_0', 'fold_1']

for fold in folds:
    print(f"\n📊 Resultados del entrenamiento - {fold}:\n")

    
    train_base_path = os.path.join(models_base_path, 'train_runs')
    actual_train_dirs = glob.glob(os.path.join(train_base_path, f'train_{fold}*'))
    
    if not actual_train_dirs:
        print(f"❌ No se encontró ningún directorio de entrenamiento para {fold}.")
        continue
        
    actual_train_dirs.sort()
    resultados_carpeta = actual_train_dirs[-1]

    best_model_path = os.path.join(resultados_carpeta, 'weights', 'best.pt')
    valid_images = os.path.join(data_base_path, fold, 'valid', 'images')

    if not os.path.exists(best_model_path):
        print(f"❌ Modelo {best_model_path} no encontrado.")
        continue

    # Cargar modelo entrenado
    model = YOLO(best_model_path)

    predict_base_dir = os.path.join(models_base_path, 'predict_runs')
    os.makedirs(predict_base_dir, exist_ok=True)

    preds = model.predict(source=valid_images, save=True, project=predict_base_dir, name=f'preds_{fold}', exist_ok=True)

    if preds and hasattr(preds[0], 'save_dir') and os.path.isdir(preds[0].save_dir):
        pred_folder = preds[0].save_dir
        pred_images = [f for f in os.listdir(pred_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if pred_images:
            pred_img_path = os.path.join(pred_folder, pred_images[0])
            print(f"1️⃣ Ejemplo de predicciones en validación guardado en: {pred_img_path}")
        else:
            print("❌ No se encontraron imágenes en los resultados.")
    else:
        print("❌ No pudimos identificar la carpeta de predicción.")

    results_csv_path = os.path.join(resultados_carpeta, 'results.csv')
    if os.path.exists(results_csv_path):
        print(f"\n2️⃣ Métricas finales guardadas en: {results_csv_path}")
    else:
        print("❌ results.csv no encontrado en el directorio de entrenamiento.")

# ============================================================================
# CELDA 7: PROBAR MODELO EN IMAGEN
# ============================================================================

print("🔍 Usando imagen de prueba local automática...")
imagen_prueba = os.path.join(data_base_path, 'fold_0', 'valid', 'images', 'damaged_1_jpg.rf.ecece902a402ee5be485902e7cb62418.jpg')

if os.path.exists(imagen_prueba):
    modelo_entrenado = YOLO(f'{resultados_carpeta}/weights/best.pt')

    resultados = modelo_entrenado.predict(
        imagen_prueba,
        conf=0.7,
        save=True,
        project=os.path.join(models_base_path, 'predict_runs'),
        name='test_img',
        exist_ok=True,
        show_labels=True,
        show_conf=True
    )
    
    num_detecciones = len(resultados[0].boxes) if resultados[0].boxes is not None else 0    
    print(f"   🥚 Huevos detectados: {num_detecciones}")


# ============================================================================
# CELDA: PROCESAR VIDEO Y GENERAR RESULTADOS
# ============================================================================

video_prueba = os.path.join(data_base_path, 'video_test.mp4')

if os.path.exists(video_prueba):
    print(f"\n🎬 Procesando video local: {video_prueba}")
    
    modelo_entrenado = YOLO(f'{resultados_carpeta}/weights/best.pt')
    save_dir_video = os.path.join(models_base_path, 'predict_video')

    resultado = modelo_entrenado.predict(
        source=video_prueba,
        conf=0.8,          
        imgsz=640,         
        save=True,         
        project=save_dir_video,
        name='predict_video',
        exist_ok=True,
        stream=True        
    )

    print(f"\n✅ Video procesado guardado en: {save_dir_video}")
else:
    print("⚠️ Nota: Para procesar un video, asegúrate de colocar un 'video_test.mp4' en tu carpeta data/.")

# ============================================================================
# CELDA 9: MODELO ENTRENADO LISTO
# ============================================================================

modelo_path = os.path.join(resultados_carpeta, 'weights', 'best.pt')
print(f"📥 El modelo entrenado puede ser importado desde: {modelo_path}")


# ============================================================================
# CELDA 10: EXPORTAR MODELO A OTROS FORMATOS
# ============================================================================


modelo_entrenado = YOLO(f'{resultados_carpeta}/weights/best.pt')

print("🔄 Exportando modelo a diferentes formatos...\n")

print("1️⃣ Exportando a ONNX...")
modelo_entrenado.export(format='onnx')
print("   ✅ Exportado a ONNX\n")

print("2️⃣ Exportando a TorchScript...")
modelo_entrenado.export(format='torchscript')
print("   ✅ Exportado a TorchScript\n")

print("3️⃣ Exportando a TFLite...")
modelo_entrenado.export(format='tflite')
print("   ✅ Exportado a TFLite\n")

print("✅ Todos los modelos exportados!")
print(f"📁 Ubicación: {resultados_carpeta}/weights/")


# ============================================================================
# CELDA 11: LIMPIAR Y HACER BACKUP
# ============================================================================

print("📦 Comprimiendo resultados para backup...")

backup_dir = os.path.join(models_base_path, 'BACKUPS')
os.makedirs(backup_dir, exist_ok=True)

nombre_zip = os.path.join(backup_dir, 'modelo_huevos_backup')
shutil.make_archive(nombre_zip, 'zip', resultados_carpeta)

zip_file_path = f"{nombre_zip}.zip"

print("✅ Backup creado exitosamente en tu carpeta de proyecto:")
print(f"   Ubicación: {zip_file_path}")
if os.path.exists(zip_file_path):
    print(f"   Tamaño: {os.path.getsize(zip_file_path) / 1e6:.2f} MB")
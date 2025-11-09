"""
LocationClassifier - Sistema de Clasificación de Ubicaciones
Librería para entrenar y predecir categorías de ubicaciones basadas en características urbanas
Versión 2.0 - Soporte para múltiples características dinámicas
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict, List, Optional


class LocationClassifier:
    """
    Clasificador de ubicaciones basado en redes neuronales.
    
    Atributos:
        model_dir (str): Directorio donde se guardan modelo y preprocesadores
        model: Modelo de red neuronal
        scaler: Normalizador MinMaxScaler
        encoder: Codificador de etiquetas
        feature_columns (list): Lista de columnas usadas como características
    """
    
    def __init__(self, model_dir: str):
        """
        Inicializa el clasificador.
        
        Args:
            model_dir: Directorio para guardar/cargar modelo y preprocesadores
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_columns = None
        
        # Rutas de archivos
        self.model_path = os.path.join(model_dir, "modelo_entrenado.keras")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.encoder_path = os.path.join(model_dir, "encoder.pkl")
        self.features_path = os.path.join(model_dir, "feature_columns.pkl")
        
        # Crear directorio si no existe
        os.makedirs(model_dir, exist_ok=True)
    
    def _calcular_pesos_clase(self, y_encoded, weight_factor=0.5):  # ✅ Agregar valor por defecto
        """
        Calcula pesos suavizados para balancear clases.
        
        Args:
            y_encoded: Array con las etiquetas codificadas
            weight_factor: Factor de suavizado (0-1). 
                        0 = sin pesos, 1 = pesos completos
                        Por defecto 0.5 (suavizado al 50%)
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Calcular pesos balanceados
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        
        # Suavizar los pesos hacia 1.0
        # Peso_final = 1 + (Peso_calculado - 1) * weight_factor
        class_weights = 1 + (class_weights - 1) * weight_factor
        
        return dict(enumerate(class_weights))
    
    def _limpiar_datos(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y preprocesa el DataFrame.
        
        Args:
            data: DataFrame con datos crudos
            
        Returns:
            DataFrame limpio
        """
        # Eliminar columnas duplicadas
        duplicadas = data.columns[data.columns.duplicated()]
        if len(duplicadas) > 0:
            print(f"⚠️  Columnas duplicadas eliminadas: {list(duplicadas)}")
            data = data.loc[:, ~data.columns.duplicated()]
        
        # Limpiar nombres de columnas
        data.columns = [col.strip() for col in data.columns]
        
        # Eliminar filas con valores nulos en columnas críticas
        antes = len(data)
        data = data.dropna(subset=['tipo', 'lat', 'lon'])
        despues = len(data)
        
        if antes != despues:
            print(f"⚠️  Eliminadas {antes - despues} filas con valores nulos")
        
        # Rellenar NaN en características numéricas con 0
        feature_cols = [col for col in data.columns if col not in ['nombre', 'tipo']]
        data[feature_cols] = data[feature_cols].fillna(0)
        
        return data
    
    def _construir_modelo(self, input_shape: int, num_classes: int) -> Sequential:
        """
        Construye la arquitectura de la red neuronal adaptada al número de características.
        
        Args:
            input_shape: Número de características de entrada
            num_classes: Número de clases a predecir
            
        Returns:
            Modelo Sequential de Keras
        """
        # Arquitectura adaptativa basada en el número de características
        if input_shape < 20:
            layer_sizes = [32, 64, 32]
        elif input_shape < 100:
            layer_sizes = [64, 128, 64, 32]
        else:
            layer_sizes = [128, 256, 128, 64, 32]
        
        model = Sequential([Dense(layer_sizes[0], activation='relu', input_shape=(input_shape,), name='entrada')])
        
        # Agregar capas ocultas con dropout
        for i, size in enumerate(layer_sizes[1:], 1):
            model.add(Dense(size, activation='relu', name=f'oculta_{i}'))
            if i % 2 == 0:  # Agregar dropout cada 2 capas
                model.add(Dropout(0.3, name=f'dropout_{i}'))
        
        # Capa de salida
        model.add(Dense(num_classes, activation='softmax', name='salida'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def entrenar(self, 
                 csv_path: str,
                 sep: str = ',',
                 target_column: str = 'tipo',
                 exclude_columns: List[str] = None,
                 test_size: float = 0.2,
                 epochs: int = 100,
                 batch_size: int = 16,
                 validation_split: float = 0.15,
                 early_stopping: bool = True,
                 patience: int = 15) -> Dict[str, float]:
        """
        Entrena el modelo con datos desde un archivo CSV.
        
        Args:
            csv_path: Ruta al archivo CSV de entrenamiento
            sep: Separador del CSV (por defecto ',')
            target_column: Nombre de la columna objetivo (por defecto 'tipo')
            exclude_columns: Columnas a excluir como características (nombre, id, etc.)
            test_size: Proporción de datos para prueba (0.0-1.0)
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
            validation_split: Proporción de datos para validación
            early_stopping: Si usar early stopping
            patience: Paciencia para early stopping
            
        Returns:
            Diccionario con métricas de evaluación
        """
        print("=" * 70)
        print("🚀 INICIANDO ENTRENAMIENTO - LocationClassifier v2.0")
        print("=" * 70)
        
        # 1. Cargar datos
        print(f"\n📂 Cargando datos desde: {csv_path}")
        data = pd.read_csv(csv_path, sep=sep)
        print(f"   ✓ Registros cargados: {len(data)}")
        print(f"   ✓ Columnas totales: {len(data.columns)}")
        
        # 2. Limpiar datos
        print("\n🧹 Limpiando datos...")
        data = self._limpiar_datos(data)
        print(f"   ✓ Registros válidos: {len(data)}")
        
        # 3. Identificar columnas a excluir
        if exclude_columns is None:
            exclude_columns = ['nombre']  # Por defecto excluir nombre
        
        exclude_columns.append(target_column)  # Siempre excluir la columna objetivo
        
        # 4. Seleccionar características
        self.feature_columns = [col for col in data.columns 
                               if col not in exclude_columns]
        
        print(f"\n📊 Características seleccionadas: {len(self.feature_columns)}")
        print(f"   Primeras 10: {self.feature_columns[:10]}")
        print(f"   Últimas 10: {self.feature_columns[-10:]}")
        
        # Guardar lista de características
        joblib.dump(self.feature_columns, self.features_path)
        print(f"   ✓ Lista de características guardada en: {self.features_path}")
        
        # 5. Separar características y etiquetas
        X = data[self.feature_columns]
        y = data[target_column]
        
        # Convertir todas las características a numérico
        print("\n🔢 Convirtiendo características a formato numérico...")
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"\n📈 Distribución de clases en '{target_column}':")
        for clase, count in y.value_counts().items():
            print(f"   - {clase}: {count} ({count/len(y)*100:.1f}%)")
        
        # Verificar balance de clases
        min_samples = y.value_counts().min()
        if min_samples < 10:
            print(f"\n⚠️  ADVERTENCIA: La clase con menos muestras tiene solo {min_samples} ejemplos.")
            print("   Considera agregar más datos para mejorar el rendimiento.")
        
        # 6. Normalizar características
        print("\n🔧 Normalizando características (MinMaxScaler)...")
        self.scaler = MinMaxScaler()
        X_normalized = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"   ✓ Scaler guardado en: {self.scaler_path}")
        print(f"   ✓ Rango de normalización: [0, 1]")
        
        # 7. Codificar etiquetas
        print("\n🏷️  Codificando etiquetas...")
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y)
        class_weights = self._calcular_pesos_clase(y_encoded,1)
        print(f"\n⚖️  Pesos de clase calculados:")
        for idx, peso in class_weights.items():
            print(f"      {self.encoder.classes_[idx]}: {peso:.2f}")
        joblib.dump(self.encoder, self.encoder_path)
        print(f"   ✓ Encoder guardado en: {self.encoder_path}")
        print(f"   ✓ Mapeo de clases:")
        for idx, clase in enumerate(self.encoder.classes_):
            print(f"      {idx} → {clase}")
        
        # 8. Dividir datos
        print(f"\n✂️  Dividiendo datos ({int((1-test_size)*100)}% entrenamiento, {int(test_size*100)}% prueba)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y_encoded, 
            test_size=test_size, 
            random_state=42,
            stratify=y_encoded
        )
        print(f"   ✓ Entrenamiento: {len(X_train)} registros")
        print(f"   ✓ Prueba: {len(X_test)} registros")
        
        # 9. Construir modelo
        print("\n🏗️  Construyendo red neuronal...")
        self.model = self._construir_modelo(
            input_shape=X_train.shape[1],
            num_classes=len(np.unique(y_encoded))
        )
        
        print(f"   ✓ Características de entrada: {X_train.shape[1]}")
        print(f"   ✓ Clases de salida: {len(np.unique(y_encoded))}")
        print(f"\n   Resumen de la arquitectura:")
        for layer in self.model.layers:
            if hasattr(layer, 'units'):
                print(f"      • {layer.name}: {layer.units} neuronas ({layer.activation.__name__})")
            else:
                print(f"      • {layer.name}: {layer.__class__.__name__}")
        
        # 10. Configurar callbacks
        callbacks = []
        if early_stopping:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            )
            callbacks.append(early_stop)
            print(f"\n   ✓ Early stopping activado (patience={patience})")
        
        # 11. Entrenar
        print(f"\n🎯 Entrenando modelo (máximo {epochs} épocas)...")
        print("=" * 70)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weights,  # ✅ AGREGAR ESTO
            verbose=1
        )
            
        # 12. Evaluar
        print("\n" + "=" * 70)
        print("📈 EVALUACIÓN FINAL EN DATOS DE PRUEBA")
        print("=" * 70)
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predicciones detalladas
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Matriz de confusión simplificada
        print("\n📊 Rendimiento por clase:")
        for idx, clase in enumerate(self.encoder.classes_):
            mask = y_test == idx
            if mask.sum() > 0:
                clase_accuracy = (y_pred_classes[mask] == idx).mean()
                print(f"   - {clase}: {clase_accuracy*100:.2f}% ({mask.sum()} muestras)")
        
        print(f"\n   📉 Pérdida (loss):      {loss:.4f}")
        print(f"   ✅ Precisión (accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 13. Guardar modelo
        self.model.save(self.model_path)
        print(f"\n💾 Modelo guardado en: {self.model_path}")
        
        print("\n" + "=" * 70)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'epochs_trained': len(history.history['loss']),
            'num_features': X_train.shape[1],
            'num_classes': len(np.unique(y_encoded)),
            'classes': list(self.encoder.classes_)
        }
    
    def cargar_modelo(self) -> bool:
        """
        Carga el modelo y preprocesadores previamente entrenados.
        
        Returns:
            True si la carga fue exitosa, False en caso contrario
        """
        try:
            print("📥 Cargando modelo entrenado...")
            
            # Verificar que existan los archivos
            archivos_requeridos = {
                'Modelo': self.model_path,
                'Scaler': self.scaler_path,
                'Encoder': self.encoder_path,
                'Características': self.features_path
            }
            
            for nombre, ruta in archivos_requeridos.items():
                if not os.path.exists(ruta):
                    raise FileNotFoundError(f"{nombre} no encontrado en: {ruta}")
            
            # Cargar
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.encoder = joblib.load(self.encoder_path)
            self.feature_columns = joblib.load(self.features_path)
            
            print("   ✓ Modelo cargado correctamente")
            print(f"   ✓ Clases disponibles: {list(self.encoder.classes_)}")
            print(f"   ✓ Número de características: {len(self.feature_columns)}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error al cargar modelo: {e}")
            return False
    
    def predecir(self,
             datos: Dict[str, float],
             mostrar_probabilidades: bool = False) -> Dict:
        """
        Predice la categoría de una ubicación a partir de sus características.
        
        Args:
            datos: Diccionario con las características (campos faltantes se rellenan con 0)
            mostrar_probabilidades: Si mostrar probabilidades de todas las clases
            
        Returns:
            Diccionario con la predicción y (opcionalmente) probabilidades
        """
        # Verificar que el modelo esté cargado
        if self.model is None or self.scaler is None or self.encoder is None:
            raise RuntimeError("Modelo no cargado. Ejecuta cargar_modelo() primero.")
        
        # Verificar características faltantes
        missing = set(self.feature_columns) - set(datos.keys())
        
        if missing:
            print(f"⚠️  Advertencia: Faltan las siguientes características: {missing}")
            print(f"📝 Agregando campos faltantes con valor por defecto 0...")
            
            # Agregar características faltantes con valor 0
            for feature in missing:
                datos[feature] = 0
            
            print(f"✅ Se agregaron {len(missing)} campos faltantes")
        
        # Crear DataFrame con las características en el orden correcto
        df = pd.DataFrame([datos])[self.feature_columns]
        
        # Convertir a numérico y rellenar NaN
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Normalizar
        X_scaled = self.scaler.transform(df)
        
        # Predecir
        probabilidades = self.model.predict(X_scaled, verbose=0)[0]
        clase_idx = np.argmax(probabilidades)
        clase_predicha = self.encoder.inverse_transform([clase_idx])[0]
        confianza = probabilidades[clase_idx]
        
        resultado = {
            'prediccion': clase_predicha,
            'confianza': float(confianza),
            'lat': datos.get('lat', None),
            'lon': datos.get('lon', None)
        }
        
        if mostrar_probabilidades:
            resultado['probabilidades'] = {
                clase: float(prob)
                for clase, prob in zip(self.encoder.classes_, probabilidades)
            }
        
        # Agregar información sobre campos completados automáticamente
        if missing:
            resultado['campos_autocompletados'] = list(missing)
        
        return resultado
    
    def predecir_desde_dataframe(self, df: pd.DataFrame, 
                                  inplace: bool = False) -> pd.DataFrame:
        """
        Predice para múltiples ubicaciones desde un DataFrame.
        
        Args:
            df: DataFrame con las características (debe tener todas las columnas del modelo)
            inplace: Si modificar el DataFrame original o retornar una copia
            
        Returns:
            DataFrame con columnas 'Prediccion' y 'Confianza' agregadas
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Ejecuta cargar_modelo() primero.")
        
        if not inplace:
            df = df.copy()
        
        # Verificar columnas
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Faltan las siguientes columnas: {missing}")
        
        # Seleccionar y normalizar características
        X = df[self.feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predecir
        probabilidades = self.model.predict(X_scaled, verbose=0)
        clases_idx = np.argmax(probabilidades, axis=1)
        clases = self.encoder.inverse_transform(clases_idx)
        confianzas = np.max(probabilidades, axis=1)
        
        # Agregar resultados
        df['Prediccion'] = clases
        df['Confianza'] = confianzas
        
        return df
    
    def predecir_desde_objeto(self, ubicacion_obj) -> object:
        """
        Predice y actualiza un objeto con atributos de ubicación.
        Compatible con objetos que tengan atributos con los nombres de las características.
        
        Args:
            ubicacion_obj: Objeto con atributos correspondientes a las características del modelo
            
        Returns:
            El mismo objeto con atributos 'Prediccion' y 'Confianza' agregados
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Ejecuta cargar_modelo() primero.")
        
        # Extraer características del objeto
        datos = {}
        for feature in self.feature_columns:
            if hasattr(ubicacion_obj, feature):
                datos[feature] = getattr(ubicacion_obj, feature)
            else:
                datos[feature] = 0  # Valor por defecto
        
        # Predecir
        resultado = self.predecir(datos, mostrar_probabilidades=False)
        
        # Actualizar objeto
        ubicacion_obj.Prediccion = resultado['prediccion']
        ubicacion_obj.Confianza = resultado['confianza']
        
        return ubicacion_obj

# CareerPath AI - Quick Start Guide

## 🚀 Cómo Ejecutar y Testear el Proyecto

### Ubicación del Proyecto

```
C:\Users\Steven Paredes\Documents\Universidad\7mo semestre\Inteligencia Artificial\ProyectoFinalIA\temp_repo
```

---

## Método 1: Ejecución Rápida (Todo está listo)

### 1. Navega al proyecto

```bash
cd "C:\Users\Steven Paredes\Documents\Universidad\7mo semestre\Inteligencia Artificial\ProyectoFinalIA\temp_repo"
```

### 2. Verifica que todo funciona

```bash
python test_project.py
```

Deberías ver:
```
[SUCCESS] All core tests passed!
```

### 3. Lanza la aplicación web

```bash
streamlit run web/app.py
```

**¡Eso es todo!** La aplicación se abrirá en tu navegador en `http://localhost:8501`

---

## Método 2: Instalación Desde Cero

Si trabajas desde otra computadora o quieres empezar limpio:

### 1. Clona el repositorio

```bash
git clone https://github.com/jjjulianleon/ProyectoFinalIA.git
cd ProyectoFinalIA
```

### 2. Crea entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Instala dependencias

```bash
pip install -r requirements.txt
```

### 4. Genera datos (si no existen)

```bash
# Opción A: Usar datos sintéticos
python src/data/generate_sample_data.py

# Opción B: Descargar datos reales de Kaggle (requiere cuenta)
kaggle datasets download -d utkarshshrivastav07/career-prediction-dataset -p data/raw --unzip
python src/data/prepare_real_data.py
python src/data/augment_real_data.py
```

### 5. Preprocesa datos

```bash
python src/data/preprocess.py
```

### 6. Entrena modelos

```bash
python src/models/train.py
```

### 7. Lanza la aplicación

```bash
streamlit run web/app.py
```

---

## 🧪 Testing del Proyecto

### Test Completo

```bash
python test_project.py
```

Esto verifica:
- ✅ Estructura del proyecto
- ✅ Dataset cargado correctamente
- ✅ Modelos ML funcionando
- ✅ Predicciones funcionando
- ✅ MCP Server operativo
- ✅ OpenAI configurado
- ✅ Streamlit instalado

### Test Individual de Componentes

#### 1. Test MCP Server

```bash
python src/mcp/career_data_server.py
```

Deberías ver:
```
[SUCCESS] All MCP server tests completed!
```

#### 2. Test OpenAI Integration

```bash
python src/models/openai_integration.py
```

Deberías ver descripciones de carreras generadas por AI.

#### 3. Test Preprocesamiento

```bash
python src/data/preprocess.py
```

#### 4. Test Entrenamiento

```bash
python src/models/train.py
```

---

## 📱 Usando la Aplicación Web

### 1. Lanza la app

```bash
streamlit run web/app.py
```

### 2. En el navegador

La app se abre automáticamente en `http://localhost:8501`

### 3. Ingresa tu perfil

**Lado izquierdo (Sidebar):**

**Personality Traits (1-10):**
- Openness: 8.5
- Conscientiousness: 7.0
- Extraversion: 6.0
- Agreeableness: 7.5
- Neuroticism: 4.0

**Aptitude Scores (0-10):**
- Numerical: 8.5
- Spatial: 7.5
- Perceptual: 8.0
- Abstract: 8.2
- Verbal: 7.0

### 4. Haz clic en "Predict My Career Path"

Verás:
- 📊 Top 5 predicciones de carreras con probabilidades
- 📈 Gráfico de barras interactivo
- 👤 Radar chart de tu perfil
- 🤖 Insights generados por AI (3 pestañas):
  - Personalized Advice
  - Top Career Details
  - Explanation
- 🔍 Feature Importance

---

## 🎯 Ejemplos de Perfiles para Testear

### Perfil 1: Tech-Oriented (Software Engineer)

```
Openness: 8.5, Conscientiousness: 7.0, Extraversion: 6.0
Agreeableness: 7.5, Neuroticism: 4.0
Numerical: 8.5, Spatial: 7.5, Perceptual: 8.0
Abstract: 8.2, Verbal: 7.0
```

**Resultado esperado:** Software Engineer, Engineer, Data Analyst

### Perfil 2: Creative (Graphic Designer)

```
Openness: 9.0, Conscientiousness: 6.0, Extraversion: 7.0
Agreeableness: 8.0, Neuroticism: 5.0
Numerical: 5.0, Spatial: 9.0, Perceptual: 9.0
Abstract: 7.0, Verbal: 7.0
```

**Resultado esperado:** Graphic Designer, Architect, Creative Professional

### Perfil 3: Healthcare (Healthcare Professional)

```
Openness: 7.0, Conscientiousness: 9.0, Extraversion: 6.0
Agreeableness: 9.5, Neuroticism: 3.0
Numerical: 6.0, Spatial: 5.0, Perceptual: 8.0
Abstract: 6.0, Verbal: 8.0
```

**Resultado esperado:** Healthcare Professional, Teacher, Psychologist

### Perfil 4: Business (Marketing Manager)

```
Openness: 7.5, Conscientiousness: 7.5, Extraversion: 9.0
Agreeableness: 7.0, Neuroticism: 4.0
Numerical: 7.0, Spatial: 5.0, Perceptual: 7.0
Abstract: 7.0, Verbal: 9.0
```

**Resultado esperado:** Marketing Manager, Sales Representative, Business Analyst

---

## 🔧 Troubleshooting

### Problema: "Streamlit not found"

```bash
pip install streamlit
```

### Problema: "No module named 'openai'"

```bash
pip install openai python-dotenv
```

### Problema: "Models not found"

```bash
# Entrena los modelos
python src/data/preprocess.py
python src/models/train.py
```

### Problema: "Dataset not found"

```bash
# Genera el dataset
python src/data/generate_sample_data.py
```

### Problema: OpenAI no funciona

- Verifica que `.env` existe y tiene tu API key
- El proyecto funciona sin OpenAI (sin insights de AI)

### Problema: Puerto 8501 ocupado

```bash
# Usa otro puerto
streamlit run web/app.py --server.port 8502
```

---

## 📊 Verificar Resultados de los Modelos

### Ver métricas del modelo

Después de entrenar, verás:

```
Random Forest - Performance Metrics
====================================
Test Accuracy:       0.7161
Precision:           0.7355
Recall:              0.7161
F1-Score:            0.6998
Cross-Val Accuracy:  0.7287 (+/- 0.0389)
```

### Ver visualizaciones

Las gráficas se guardan en:
```
models/plots/
├── confusion_matrix_random_forest.png
├── confusion_matrix_logistic_regression.png
└── feature_importance.png
```

---

## 🎥 Demo Para Presentación

### Script de Demo (5 minutos)

**1. Intro (30 seg)**
- Mostrar README en GitHub
- Explicar el objetivo del proyecto

**2. Arquitectura (1 min)**
```bash
tree -L 2  # Mostrar estructura
```

**3. Test Rápido (1 min)**
```bash
python test_project.py
```

**4. Demo Aplicación (2 min)**
```bash
streamlit run web/app.py
```
- Ingresar perfil de ejemplo
- Mostrar predicciones
- Mostrar AI insights
- Mostrar visualizaciones

**5. MCP Server (30 seg)**
```bash
python src/mcp/career_data_server.py
```

---

## 📝 Comandos Útiles

### Ver dataset

```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/career_data.csv'); print(df.head()); print('\n', df['Career'].value_counts())"
```

### Ver clases del modelo

```bash
python -c "import joblib; le = joblib.load('models/label_encoder.joblib'); print(le.classes_)"
```

### Ver accuracy del modelo

```bash
python -c "import joblib; import pandas as pd; m = joblib.load('models/best_model.joblib'); X = pd.read_csv('data/processed/X_test.csv'); y = pd.read_csv('data/processed/y_test.csv').values.ravel(); print(f'Accuracy: {m.score(X, y):.2%}')"
```

---

## 🌐 Deployment (Opcional)

### Deploy en Streamlit Cloud (GRATIS)

1. Sube tu proyecto a GitHub (ya está)
2. Ve a https://share.streamlit.io
3. Conecta tu repositorio
4. Selecciona `web/app.py`
5. Click "Deploy"

¡Tu app estará online en minutos!

---

## ✅ Checklist Pre-Presentación

- [ ] `python test_project.py` pasa todos los tests
- [ ] `streamlit run web/app.py` se abre correctamente
- [ ] Probaste con 2-3 perfiles diferentes
- [ ] AI insights funcionan (o sabes que son opcionales)
- [ ] Tienes screenshots de backup
- [ ] Conoces la accuracy del modelo (71.6%)
- [ ] Puedes explicar las 25 categorías de carreras
- [ ] Sabes que usas datos reales de Kaggle

---

## 📞 Ayuda

Si algo no funciona:

1. Revisa `test_project.py` para diagnóstico
2. Lee los mensajes de error completos
3. Verifica que instalaste todas las dependencias
4. Chequea que estás en el directorio correcto

---

**¡Listo para presentar!** 🎓🚀

**URL GitHub:** https://github.com/jjjulianleon/ProyectoFinalIA

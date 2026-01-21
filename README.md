# 🐱🐶 Cat vs Dog - Image Classification

Projeto de **Visão Computacional** utilizando **Deep Learning** para classificar imagens de **gatos** e **cachorros** em tempo real via webcam.

---

### 🚀 Tecnologias

- **Python 3**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**

---

### 📂 Estrutura do Projeto

- `train.py` → Treina o modelo usando o dataset Cats vs Dogs  
- `webcam_predict.py` → Classifica imagens em tempo real via webcam  
- `model/cat_dog_model.h5` → Modelo pré-treinado  
- `requirements.txt` → Dependências do projeto  

---

### 📝 Pré-requisitos

- Python 3.10+  
- Dataset **[Cat and Dog](https://www.kaggle.com/datasets/tongpython/cat-and-dog)** (necessário apenas para treinar o modelo)  
- Webcam (para classificação em tempo real)

#### 📂 Dataset

O projeto utiliza o dataset **[Cat and Dog](https://www.kaggle.com/datasets/tongpython/cat-and-dog)**, disponível no Kaggle.  
- Necessário apenas se você for treinar o modelo.  
- Caso use o modelo pré-treinado, não é necessário baixar o dataset.  

---

###  Como Executar

```bash
# 1️⃣ Criar e ativar virtualenv
python3 -m venv venv
source venv/bin/activate

# 2️⃣ Instalar dependências
pip install -r requirements.txt

# 3️⃣ Treinar o modelo (opcional, requer dataset)
python train.py

# 4️⃣ Executar classificação via webcam
python webcam_predict.py

```
---
💡 Dica: Para rodar apenas a classificação, use o modelo pré-treinado disponível em model/cat_dog_model.h5.

### 📈 Observações

O modelo é binário: Gato ou Cachorro

As imagens são redimensionadas para 224x224 pixels antes da classificação

Caso opte por baixar o dataset e treinar, o processo pode levar tempo dependendo do hardware.

## 👨‍💻 Autor

Marcos Gustavo

Projeto desenvolvido para fins acadêmicos.

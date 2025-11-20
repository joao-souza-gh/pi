import os
import random
import numpy as np
import tensorflow as tf
import pathlib
import gc
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURAÇÕES ---
# Seeds sequenciais para validação estatística + Seeds "culturais" famosas
SEEDS_SEQUENCIAIS = list(range(0, 5))
SEEDS_FAMOSAS = [0, 7, 42, 2025, 314159, 777, 1337, 12345, 271828, 52, 40028922]

# Combina as listas e remove duplicatas, mantendo a ordem
SEEDS_PARA_TESTAR = sorted(list(set(SEEDS_SEQUENCIAIS + SEEDS_FAMOSAS)))

ARQUIVO_LOG = "ranking_seeds_metricas_compostas.txt"
MELHOR_MODELO_NOME = "melhor_modelo_seed_{}.keras"

# Configurações do Dataset
IMG_HEIGHT = 176
IMG_WIDTH = 208
# Reduzi levemente o batch_size para garantir que rode em GPUs menores
# Se sua GPU for forte (ex: RTX 3060 ou superior), pode tentar voltar para 32
BATCH_SIZE = 16
DIR_DADOS = pathlib.Path('dataset')


def reset_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def criar_modelo():
    modelo = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])
    modelo.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return modelo


def isolar_luminancia(imagem, rotulo):
    imagem = tf.image.rgb_to_grayscale(imagem)
    imagem = tf.cast(imagem, tf.float32) / 255.0
    return imagem, rotulo


# Loop Principal
melhor_score_global = -float('inf')
melhor_seed_global = -1
melhores_metricas = (0.0, 0.0)

# Limpa o arquivo de log
with open(ARQUIVO_LOG, "w") as f:
    f.write("SEED | SCORE (Acc - Loss) | Val_Acc | Val_Loss | STATUS\n")

print(f"Iniciando busca nas seguintes seeds: {SEEDS_PARA_TESTAR}")

for seed_atual in SEEDS_PARA_TESTAR:
    print(f"\n--- TESTANDO SEED: {seed_atual} ---")

    # 1. Resetar memória agressivamente antes de começar
    tf.keras.backend.clear_session()
    gc.collect()
    reset_seeds(seed_atual)

    # 2. Carregar Dataset
    try:
        dataset_completo = tf.keras.utils.image_dataset_from_directory(
            DIR_DADOS,
            seed=seed_atual,
            shuffle=True,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=None
        )

        dataset_completo = dataset_completo.shuffle(10000, seed=seed_atual)

        ds_size = len(dataset_completo)
        train_size = int(0.7 * ds_size)
        val_size = int(0.15 * ds_size)

        train_ds = dataset_completo.take(train_size)
        val_ds = dataset_completo.skip(train_size).take(val_size)

        # ALTERAÇÃO IMPORTANTE: Removi o .cache() para evitar estouro de memória no loop
        # Mantive o prefetch para performance
        train_ds = train_ds.batch(BATCH_SIZE).map(isolar_luminancia).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(BATCH_SIZE).map(isolar_luminancia).prefetch(tf.data.AUTOTUNE)

        # 3. Treinar
        modelo = criar_modelo()
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

        modelo.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            callbacks=[early_stop],
            verbose=0
        )

        # 4. Avaliar
        val_loss, val_acc = modelo.evaluate(val_ds, verbose=0)

        # Score = Acurácia - Loss
        score_atual = val_acc - val_loss

        print(f"Seed {seed_atual} -> Score: {score_atual:.5f} (Acc: {val_acc:.5f}, Loss: {val_loss:.5f})")

        with open(ARQUIVO_LOG, "a") as f:
            f.write(f"{seed_atual} | {score_atual:.8f} | {val_acc:.8f} | {val_loss:.8f} | \n")

        if score_atual > melhor_score_global:
            melhor_score_global = score_atual
            melhor_seed_global = seed_atual
            melhores_metricas = (val_acc, val_loss)

            print(f"NOVO RECORDE! Salvando modelo da seed {seed_atual}...")
            modelo.save(MELHOR_MODELO_NOME.format(seed_atual))

            with open(ARQUIVO_LOG, "a") as f:
                f.write(f"--> NOVO RECORDE (Salvo)\n")

    except Exception as e:
        print(f"ERRO na seed {seed_atual}: {e}")
        # Tenta limpar memória mesmo se der erro
        pass

    # 5. Limpeza Final da Iteração (CRÍTICO PARA EVITAR OOM)
    # Deletamos explicitamente os objetos pesados
    if 'modelo' in locals(): del modelo
    if 'dataset_completo' in locals(): del dataset_completo
    if 'train_ds' in locals(): del train_ds
    if 'val_ds' in locals(): del val_ds

    # Força o Garbage Collector do Python e limpa sessão do TF
    gc.collect()
    tf.keras.backend.clear_session()

print("\n" + "=" * 30)
print(f"BUSCA FINALIZADA")
print(f"Melhor Seed: {melhor_seed_global}")
print(f"Melhor Score: {melhor_score_global:.8f}")
print(f"Métricas Finais -> Acc: {melhores_metricas[0]:.8f}, Loss: {melhores_metricas[1]:.8f}")
print("=" * 30)
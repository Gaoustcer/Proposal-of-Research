"""
对于MNIST数据集的MIA攻击实现
"""
import numpy as np

from sklearn.model_selection import train_test_split
from absl import app
from absl import flags
from tensorflow.keras import layers
import tensorflow as tf



from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data


NUM_CLASSES = 10
WIDTH = 28
HEIGHT = 28
CHANNELS = 3
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 12, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 12, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 3, "Number of shadow models")


def get_data():
    """从MNIST数据集到训练集和测试集"""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_test=np.stack((X_test, X_test, X_test) , axis=-1)
    X_train=np.stack((X_train, X_train , X_train) , axis=-1)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    return (X_train, y_train), (X_test, y_test)


def target_model_fn():
    """目标模型的结构"""
    model = tf.keras.models.Sequential()
    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS),
        )
    )
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def attack_model_fn():
    """攻击模型的结构 """
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(128, activation="relu", input_shape=(NUM_CLASSES,)))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def demo(argv):
    del argv
    (X_train, y_train), (X_test, y_test) = get_data()
    # 训练目标模型
    print("Training the target model...")
    target_model = target_model_fn()
    target_model.fit(
        X_train, y_train, epochs=FLAGS.target_epochs, validation_split=0.1, verbose=True
    )
    # 训练影子模型
    smb = ShadowModelBundle(
        target_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=FLAGS.num_shadows,
    )
    # 使用目标模型的测试集来训练影子模型
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        X_test, y_test, test_size=0.1
    )
    print(attacker_X_train.shape, attacker_X_test.shape)
    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=FLAGS.target_epochs,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
        ),
    )
    # 影子模型的输出用来训练攻击模型
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)
    # 训练攻击模型
    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True)
    )
    # 测试攻击模型
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )
    # 计算攻击命中率
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)
    print(attack_accuracy)

if __name__ == "__main__":
    app.run(demo)
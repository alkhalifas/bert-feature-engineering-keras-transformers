import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs
from keras_bert import Tokenizer
import matplotlib.pyplot as plt

sentence_pairs = [
    [
        ['covid', 'was', 'a', 'terrible', 'illness'], 
        ['this', 'year', 'was', 'not', 'fun']
    ],
    [
        ['camels', 'are', 'fun', 'creatures'], 
        ['that', 'eat', 'alot']
    ],
    [
        ['computers', 'are', 'very', 'expensive'], 
        ['but', 'help', 'create', 'cool', 'websites']
    ],
]

## Build Token Dictionary
token_dict = get_base_dict()
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())

print("token_dict size: ", len(token_dict))

## Build Keras Model
model = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
)
compile_model(model)
model.summary()

## Train Keras Model
def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )

history = model.fit_generator(
    generator=_generator(),
    steps_per_epoch=1000,
    epochs=50,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
)


## Use the trained Keras model
inputs, output_layer = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
    training=False,
    trainable=False, 
    output_layer_num=4, 
)

print(history.history.keys())

plt.subplot(131)
plt.plot(history.history['MLM_loss'])
plt.plot(history.history['val_MLM_loss'])
plt.title('Masked LM')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.subplot(132)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.subplot(133)
plt.plot(history.history['NSP_loss'])
plt.plot(history.history['val_NSP_loss'])
plt.title('Next Sentence Prediction (NSP) Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.savefig('transformers_train_results.png')
plt.show()


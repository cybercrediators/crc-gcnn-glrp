# glrp.model package

## Subpackages


* [glrp.model.callbacks package](glrp.model.callbacks.md)


    * [Submodules](glrp.model.callbacks.md#submodules)


    * [glrp.model.callbacks.auc module](glrp.model.callbacks.md#module-glrp.model.callbacks.auc)


    * [Module contents](glrp.model.callbacks.md#module-glrp.model.callbacks)


## Submodules

## glrp.model.gcnn module


### _class_ glrp.model.gcnn.GCNNModel(config, data)
Bases: [`BaseModel`](glrp.base.md#glrp.base.base_model.BaseModel)

The GCNN model class.


#### _class_ Net(\*args, \*\*kwargs)
Bases: `Model`

Subclassed gcnn model from a given config file.


#### add_dropout(layer, dropout)
add a dropout layer to the current model.


#### call(inputs)
Calls the model on new inputs and returns the outputs as tensors.

In this case call() just reapplies
all ops in the graph to the new inputs
(e.g. build a new computational graph from the provided inputs).

Note: This method should not be called directly. It is only meant to be
overridden when subclassing tf.keras.Model.
To call a model on an input, always use the __call__() method,
i.e. model(inputs), which relies on the underlying call() method.


* **Parameters**

    
    * **inputs** – Input tensor, or dict/list/tuple of input tensors.


    * **training** – Boolean or boolean scalar tensor, indicating whether to run
    the Network in training mode or inference mode.


    * **mask** – A mask or list of masks. A mask can be either a boolean tensor or
    None (no mask). For more details, check the guide

    > [here]([https://www.tensorflow.org/guide/keras/masking_and_padding](https://www.tensorflow.org/guide/keras/masking_and_padding)).




* **Returns**

    A tensor if there is a single output, or
    a list of tensors if there are more than one outputs.



#### model()

#### add_callbacks()
add tensorflow callbacks to the custom training loop


#### build_model()
Return the current model as a keras Model.


#### conv_inputs(inp)
prepare inputs with coarsened adj matrix


#### evaluate(loader)
evaluate the current model with the given data from
the spektral loader.


#### load(fname='gcnn_checkpoint.ckpt')
Load the current model.


#### save(fname='gcnn_checkpoint.ckpt')
Save the current model.


#### train(train_loader, test_loader, val_loader)
Train the current model on the given train, test
and validation loader.


#### train_on_batch(inputs, target)
## glrp.model.mnist module


### _class_ glrp.model.mnist.MnistModel(config, data)
Bases: [`BaseModel`](glrp.base.md#glrp.base.base_model.BaseModel)

testing spektral model with the spektral example model for gcn with chebconv


#### _class_ Net(\*args, \*\*kwargs)
Bases: `Model`


#### add_dropout(layer, dropout)

#### call(inputs)
Calls the model on new inputs and returns the outputs as tensors.

In this case call() just reapplies
all ops in the graph to the new inputs
(e.g. build a new computational graph from the provided inputs).

Note: This method should not be called directly. It is only meant to be
overridden when subclassing tf.keras.Model.
To call a model on an input, always use the __call__() method,
i.e. model(inputs), which relies on the underlying call() method.


* **Parameters**

    
    * **inputs** – Input tensor, or dict/list/tuple of input tensors.


    * **training** – Boolean or boolean scalar tensor, indicating whether to run
    the Network in training mode or inference mode.


    * **mask** – A mask or list of masks. A mask can be either a boolean tensor or
    None (no mask). For more details, check the guide

    > [here]([https://www.tensorflow.org/guide/keras/masking_and_padding](https://www.tensorflow.org/guide/keras/masking_and_padding)).




* **Returns**

    A tensor if there is a single output, or
    a list of tensors if there are more than one outputs.



#### build_model()
Build a new model


#### evaluate(loader)
Evaluate the current model


#### load()
Load a saved model


#### required_params()
return the required parameters needed from a config file


#### save()
Save the current model


#### train(train_loader, test_loader, val_loader)
Train the model


#### train_on_batch(inputs, target)
## glrp.model.testing module


### _class_ glrp.model.testing.TestingModel(config, data)
Bases: [`BaseModel`](glrp.base.md#glrp.base.base_model.BaseModel)

testing spektral model with the spektral example model for gcn with chebconv


#### add_dropout(layer, dropout)

#### build_model()
Build a new model


#### load()
Load a saved model


#### required_params()
return the required parameters needed from a config file


#### save()
Save the current model

## Module contents

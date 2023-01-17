# glrp.trainer package

## Submodules

## glrp.trainer.gcnn_runner module


### _class_ glrp.trainer.gcnn_runner.GCNNRunner(config, model, data)
Bases: [`BaseTrainer`](glrp.base.md#glrp.base.base_trainer.BaseTrainer)


#### run()
run the given model


#### summarize()
log current values to e.g. tensorboard

## glrp.trainer.mnist_runner module


### _class_ glrp.trainer.mnist_runner.MnistRunner(config, model, data)
Bases: [`BaseTrainer`](glrp.base.md#glrp.base.base_trainer.BaseTrainer)


#### run()
run the given model


#### summarize()
log current values to e.g. tensorboard

## glrp.trainer.test_runner module


### _class_ glrp.trainer.test_runner.Net(\*args, \*\*kwargs)
Bases: `Model`


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



### _class_ glrp.trainer.test_runner.TestingRunner(config, model, data)
Bases: [`BaseTrainer`](glrp.base.md#glrp.base.base_trainer.BaseTrainer)


#### evaluate(loader)

#### run()
run the given model


#### summarize()
log current values to e.g. tensorboard


#### train_step(inputs, target)
## Module contents

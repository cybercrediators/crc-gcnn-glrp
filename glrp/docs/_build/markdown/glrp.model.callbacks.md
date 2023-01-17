# glrp.model.callbacks package

## Submodules

## glrp.model.callbacks.auc module


### _class_ glrp.model.callbacks.auc.roc_callback(training_data, validation_data)
Bases: `Callback`


#### on_batch_begin(batch, logs={})
A backwards compatibility alias for on_train_batch_begin.


#### on_batch_end(batch, logs={})
A backwards compatibility alias for on_train_batch_end.


#### on_epoch_begin(epoch, logs={})
Called at the start of an epoch.

Subclasses should override for any actions to run. This function should only
be called during TRAIN mode.


* **Parameters**

    
    * **epoch** – Integer, index of epoch.


    * **logs** – Dict. Currently no data is passed to this argument for this method
    but that may change in the future.



#### on_epoch_end(epoch, logs={})
Called at the end of an epoch.

Subclasses should override for any actions to run. This function should only
be called during TRAIN mode.


* **Parameters**

    
    * **epoch** – Integer, index of epoch.


    * **logs** – Dict, metric results for this training epoch, and for the

        validation epoch if validation is performed. Validation result keys
        are prefixed with val_. For training epoch, the values of the

    Model’s metrics are returned. Example

        0.7}\`.




#### on_train_begin(logs={})
Called at the beginning of training.

Subclasses should override for any actions to run.


* **Parameters**

    **logs** – Dict. Currently no data is passed to this argument for this method
    but that may change in the future.



#### on_train_end(logs={})
Called at the end of training.

Subclasses should override for any actions to run.


* **Parameters**

    **logs** – Dict. Currently the output of the last call to on_epoch_end()
    is passed to this argument for this method but that may change in
    the future.


## Module contents

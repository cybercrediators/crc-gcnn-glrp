# glrp.base package

## Submodules

## glrp.base.base_explainer module


### _class_ glrp.base.base_explainer.BaseExplainer(config, model, data)
Bases: `object`

Abstract base class for a gcnn explainer


#### explain()
Explain a given model.


#### plot_things()
do some plots


#### visualize()
do visualizations

## glrp.base.base_model module


### _class_ glrp.base.base_model.BaseModel(config)
Bases: `object`

Abstract base class for a tf model


#### build_model()
Build a new model


#### evaluate()
Evaluate the current model


#### load()
Load a saved model


#### save()
Save the current model


#### train()
Train the model

## glrp.base.base_trainer module


### _class_ glrp.base.base_trainer.BaseTrainer(config, model, data)
Bases: `object`

Abstract class for the model runner


#### run()
run the given model


#### summarize()
log current values to e.g. tensorboard

## Module contents

## GLRP test runs
+ most likely too few datasets currently (only ~250) -> integrate more possible datasets
+ runtimes: ~7-9 minutes
+ maybe integrate tensorboard graphs
+ Net:
  + 2 graph convolutional layers (32 filters, neighborhood size 7)
  + max-pooling (2)
  + 2 fully-connected (512/128)
+ 10-fold cross validation

## Deployment and Class diagram

![Deployment](../text/master/Data/diagrams/deployment_arch.png)
![Class](../text/master/Data/diagrams/simple_class.png)

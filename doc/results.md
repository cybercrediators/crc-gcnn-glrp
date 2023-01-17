# Results
## New results
+ Trained models can be downloaded here:
  + [HPRD GE](https://megastore.rz.uni-augsburg.de/get/sfPzUyZmVz/)
  + [HPRD CRC](https://megastore.rz.uni-augsburg.de/get/2DcHk5GQL0/)
  + [Reactome CRC](https://megastore.rz.uni-augsburg.de/get/Oiq_21dfa0/)
  + [StringDB CRC](https://megastore.rz.uni-augsburg.de/get/nXbG0Kt8ko/)

### Time comparisons
+ Time comparisons were done with the HPRD PPI network only on the breast/colorectal (small) cancer datsets

#### Colorectal Cancer dataset

| Type               | Original    | Spektral       |
|--------------------|-------------|----------------|
| ChebNet (Training) | 525s - 554s | 414s - 474.4s |
| GLRP (ChebNet)     | 422s - 434s | 523s - 541s    |

#### Breast Cancer dataset

| Type               | Original            | Spektral     |
|--------------------|---------------------|--------------|
| ChebNet (Training) | 697s - 758s, ~10min | ~320-390s      |
| GLRP (ChebNet)     | 465s- 502s          | ~498s - 565s |

### Model Performance

#### Original with CRC dataset

  | Network           | AUC            | F1-Score       | Accuracy       | Time       |
  |-------------------|----------------|----------------|----------------|------------|
  | HPRD PPI          | 0.985 +- 0.072 | 0.945 +- 0.051 | 0.944 +- 0.052 | ~14.39 min |
  | Reactome Pathways | 0.988 +- 0.063 | 0.932 +- 0.024 | 0.932 +- 0.056 | ~67.34 min |
  | StringDB          | 0.975 +- 0.031 | 0.932 +- 0.014 | 0.932 +- 0.04  | ~275.6 min |

#### Spektral on: Breast Cancer dataset

  | Network             | Accuracy             | Precision            | AUC(100)             | Recall               | F1-Score | MetaRelVis Link                                                                                      | Ndex Link                                                                            | Time                        | Config                                                     |
  |---------------------|----------------------|----------------------|----------------------|----------------------|----------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|-----------------------------|------------------------------------------------------------|
  | (ChebNet) HPRD PPI  | 0.90138              | 0.90361              | 0.89284              | 0.84746              | 0.87464  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=eaa29509-1e6f-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/viewer/networks/eaa29509-1e6f-11ed-ac45-0ac135e8bacf) | 32 - 75s, Glrp: 498s - 565s | [Config ChebNet](../glrp/config/ge/ge_gcnn.json)           |
  |                     | (Validation) 0.79381 | (Validation) 0.74359 | (Validation) 0.78559 | (Validation) 0.74359 |          |                                                                                                      |                                                                                      |                             |                                                            |
  | (ChebNet2) HPRD PPI | 0.94725              | 0.95294              | 0.94218              | 0.91525              | 0.93372  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=1e84232c-1e70-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/v2/network/1e84232c-1e70-11ed-ac45-0ac135e8bacf)      | 460s - 490s, GLRP: 521s - 558s | [Config ChebNet2](../glrp/config/ge/ge_gcnn_chebnet2.json) |
  |                     | (Validation) 0.80412 | (Validation) 0.76316 | (Validation) 0.79421 | (Validation) 0.74359 |          |                                                                                                      |                                                                                      |


#### Spektral on: Colorectal Cancer dataset (733 samples, 659/74 training/test samples)


  | Network             | Accuracy              | Precision                       | AUC (100)              | Recall                 | F1-Score | MetaRelVis Link                                                                                      | Ndex Link                                                                            | Training Time                           | Configuration                                                            |
  |---------------------|-----------------------|---------------------------------|------------------------|------------------------|----------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|-----------------------------------------|--------------------------------------------------------------------------|
  | (ChebNet) HPRD PPI  | 0.99852               | 0.99736                         | 0.99868                | 0.99737                | 0.99868  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=058a3dc6-1ed0-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/v2/network/058a3dc6-1ed0-11ed-ac45-0ac135e8bacf)      | 414s - 474.4s, GLRP: 523s - 541s       | [Config CRC ChebNet](../glrp/config/crc/crc_gcnn.json)                   |
  |                     | (Validation) 0.95946  | (Validation) 0.97959            | (Validation) 0.95917   | (Validation) 0.96000   |          |                                                                                                      |                                                                                      |                                         |                                                                          |
  | (ChebNet) Reactome  | 0.99515               | 0.99736                         | 0.99558                | 0.99474                | 0.99605  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=879847b5-1ed3-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/v2/network/879847b5-1ed3-11ed-ac45-0ac135e8bacf)      | 744s - 769s, GLRP: 487.3s - 492.8s    | [Config CRC ChebNet Reactome](../glrp/config/crc/crc_gcnn_reactome.json) |
  |                     | (Validation) 0.95946  | (Validation) 0.98000            | (Validation) 0.96917   | (Validation) 0.94000   |          |                                                                                                      |                                                                                      |                                         |                                                                          |
  | (ChebNet) String    | 0.95367               | 0.96783                         | 0.95349                | 0.95000                | 0.95883  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=47dd78bb-1ed4-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/v2/network/47dd78bb-1ed4-11ed-ac45-0ac135e8bacf)      | 1853s - 1908s, GLRP: 6985s - 7015.95s | [Config CRC ChebNet String](../glrp/config/crc/crc_gcnn_string.json)     |
  |                     | (Validation)  0.93243 | (Validation)   0.97872          | (Validation)   0.93917 | (Validation)   0.92000 |          |                                                                                                      |                                                                                      |                                         |                                                                          |
  | (ChebNet2) HPRD PPI | 0.98222               | 0.98950                         | 0.98231                | 0.97922                | 0.98433  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=a8d5fb9e-1ed0-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/viewer/networks/a8d5fb9e-1ed0-11ed-ac45-0ac135e8bacf) | 434s - 523.3s, GLRP: 531s - 581.8s       | [Config CRC ChebNet2](../glrp/config/crc/crc_gcnn_cn2.json)              |
  |                     | (Validation) 0.94595  | (Validation) 0.93617 | (Validation): 0.93716  | (Validation): 0.97778  |          |                                                                                                      |                                                                                      |
  | (ChebNet2) Reactome | 0.99556               | 0.99736                         | 0.99558                | 0.99474                | 0.99605  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=c5a15298-1ed3-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/viewer/networks/c5a15298-1ed3-11ed-ac45-0ac135e8bacf) | 881 - 927s, GLRP: 487.1s - 499.1s       | [Config CRC ChebNet2 Reactome](../glrp/config/crc/crc_gcnn_reactome_cn2.json)              |
  |                     | (Validation) 0.95946  | (Validation) 0.96078 | (Validation): 0.94833  | (Validation): 0.98000  |          |                                                                                                      |                                                                                      |
  | (ChebNet2) String   | 0.94074               | 0.95699                         | 0.93975                | 0.93684                | 0.94681  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=dcc45e3e-1ed4-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/viewer/networks/dcc45e3e-1ed4-11ed-ac45-0ac135e8bacf) | 1793s - 1819s, GLRP: 10487.8s       | [Config CRC ChebNet2 String](../glrp/config/crc/crc_gcnn_string_cn2.json)              |
  |                     | (Validation) 0.97297  | (Validation) 0.96154 | (Validation): 0.95833  | (Validation): 0.94000  |          |                                                                                                      |                                                                                      |

#### Spektral on: Colorectal Cancer dataset (1431 samples, 1281/150 training/test samples)


  | Network             | Accuracy             | Precision              | AUC (100)             | Recall                | F1-Score | MetaRelVis Link                                                                                      | Ndex Link                                                                            | Training Time                               | Configuration                                                                              |
  |---------------------|----------------------|------------------------|-----------------------|-----------------------|----------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------|--------------------------------------------------------------------------------------------|
  | (ChebNet) HPRD PPI  | 0.89494              | 0.89086                | 0.89004               | 0.92734               | 0.90874  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=e96b4b3d-1eda-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/v2/network/e96b4b3d-1eda-11ed-ac45-0ac135e8bacf)      | 840s - 1010s, GLRP: 544s - 561s             | [Config CRC 1431 ChebNet](../glrp/config/crc_ext/crc_ext_gcnn.json)                        |
  |                     | (Validation) 0.71333 | (Validation) 0.80822   | (Validation) 0.72232  | (Validation) 0.67045  |          |                                                                                                      |                                                                                      |                                             |                                                                                            |
  | (ChebNet) Reactome  | 0.82400              | 0.80698                | 0.81034               | 0.90838               | 0.85468  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=93f36601-1ed6-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/v2/network/93f36601-1ed6-11ed-ac45-0ac135e8bacf)      | 1558 - 1616s, GLRP: 498.5s - 496.3s         | [Config CRC 1431 ChebNet Reactome](../glrp/config/crc_ext/crc_ext_reactome_gcnn.json)      |
  |                     | (Validation) 0.66000 | (Validation) 0.67857   | (Validation) 0.65620  | (Validation) 0.70370  |          |                                                                                                      |                                                                                      |                                             |                                                                                            |
  | (ChebNet) String    | 0.8188               | 0.81266                | 0.81546               | 0.90190               | 0.85496  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=32e873e7-1ed7-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/v2/network/32e873e7-1ed7-11ed-ac45-0ac135e8bacf)      | 4412.3s - 4440.5s, GLRP: 6031.3s - 6120.27s | [Config CRC 1431 ChebNet String](../glrp/config/crc_ext/crc_ext_string_gcnn.json)          |
  |                     | (Validation) 0.71735 | (Validation) 0.71277   | (Validation) 0.65886  | (Validation) 0.82716  |          |                                                                                                      |                                                                                      |                                             |                                                                                            |
  | (ChebNet2) HPRD PPI | 0.85375              | 0.84360                | 0.83106               | 0.90489               | 0.87317  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=bfe4f745-1ede-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/viewer/networks/bfe4f745-1ede-11ed-ac45-0ac135e8bacf) | 948s - 1023.5s, GLRP: 523s - 541s           | [Config CRC 1431 ChebNet2](../glrp/config/crc_ext/crc_ext_gcnn_cn2.json)                   |
  |                     | (Validation) 0.74595 | (Validation) : 0.73617 | (Validation): 0.73716 | (Validation): 0.77778 |          |                                                                                                      |                                                                                      |
  | (ChebNet2) Reactome | 0.88008              | 0.88020                | 0.87448               | 0.91361               | 0.89660  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=64ad7a01-213f-11ed-ac45-0ac135e8bacf) | [Link](https://www.ndexbio.org/viewer/networks/64ad7a01-213f-11ed-ac45-0ac135e8bacf) | 1490s - 1599s, GLRP: 501.8s - 512.6s        | [Config CRC 1431 ChebNet2 Reactome](../glrp/config/crc_ext/crc_ext_reactome_gcnn_cn2.json) |
  |                     | (Validation) 0.67333 | (Validation) : 0.71622 | (Validation): 0.67499 | (Validation): 0.77778 |          |                                                                                                      |                                                                                      |
  | (ChebNet2) String   | 0.85516              | 0.84795                | 0.84723               | 0.90630               | 0.87615  | [Link](https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=a5488b0a-1ed7-11ed-ac45-0ac135e8bacf)  | [Link](https://www.ndexbio.org/v2/network/a5488b0a-1ed7-11ed-ac45-0ac135e8bacf)      | 3716s - 3832s, GLRP: 10364s - 10490.6s      | [Config CRC 1432 ChebNet2 String](../glrp/config/crc_ext/crc_ext_string_gcnn_cn2.json)     |
  |                     | (Validation) 0.7000  | (Validation) 0.75000   | (Validation) 0.70894  | (Validation) 0.85185  |          |                                                                                                      |                                                                                      |                                             |                                                                                            |

#### Spektral on: Colorectal Cancer Dataset (1431 samples) visualised on the trained model for the CRC dataset 733 samples
+ https://frankkramer-lab.github.io/MetaRelSubNetVis?uuid=d0fcc8a4-16f1-11ed-ac45-0ac135e8bacf
+ https://www.ndexbio.org/viewer/networks/d0fcc8a4-16f1-11ed-ac45-0ac135e8bacf
  
  
## Old results
## Performance

  | Network           | AUC            | F1-Score       | Accuracy       | Time       |
  |-------------------|----------------|----------------|----------------|------------|
  | HPRD PPI          | 0.985 +- 0.072 | 0.945 +- 0.051 | 0.944 +- 0.052 | ~14.39 min |
  | Reactome Pathways | 0.988 +- 0.063 | 0.932 +- 0.024 | 0.932 +- 0.056 | ~67.34 min |
  | StringDB          | 0.975 +- 0.031 | 0.932 +- 0.014 | 0.932 +- 0.04  | ~275.6 min |

+ Updated training times: 
  + mnist:
  + orig: ~60-70s
  + crc: ~55-60s

## Visualization
### Most relevant gene occurences

+ HPRD most relevant gene occurences

![HPRD Gene occurences](./text/Data/hprd_occ.png)
+ Reactome most relevant gene occurences

![Reactome Gene occurences](./text/Data/reactome_occ.png)
+ String most relevant gene occurences

![String Gene occurences](./text/Data/string_occ.png)

### Selected patient subnetworks

| Patient   | (assigned) Subtype | Cancer 0/1 |
|-----------|--------------------|------------|
| GSM452660 | CMS1               | cancer     |
| GSM58845  | CMS2               | cancer     |
| GSM58873  | CMS1               | normal     |
| GSM916761 | CMS4               | normal     |

#### HPRD PPI

HPRD GSM452660, CMS1, cancer
![HPRD GSM452660, CMS1, cancer](./text/Data/hprdsel1.png)
HPRD GSM58845, CMS2, cancer
![HPRD GSM58845, CMS2, cancer](./text/Data/hprdsel2.png)
HPRD GSM58873, CMS1, normal
![HPRD GSM58873, CMS1, normal](./text/Data/hprdsel3.png)
HPRD GSM916761, CMS4, normal
![HPRD GSM916761, CMS4, normal](./text/Data/hprdsel4.png)

### Reactome
Reactome GSM452660, CMS1, cancer
![Reactome GSM452660, CMS1, cancer](./text/Data/reactomesel1.png)
Reactome GSM58845, CMS2, cancer
![Reactome GSM58845, CMS2, cancer](./text/Data/reactomesel2.png)
Reactome GSM58873, CMS1, normal
![Reactome GSM58873, CMS1, normal](./text/Data/reactomesel3.png)
Reactome GSM916761, CMS4, normal
![Reactome GSM916761, CMS4, normal](./text/Data/reactomesel4.png)


### String
String GSM452660, CMS1, cancer
![String GSM452660, CMS1, cancer](./text/Data/stringsel1.png)
String GSM58845, CMS2, cancer
![String GSM58845, CMS2, cancer](./text/Data/stringsel2.png)
String GSM58873, CMS1, normal
![String GSM58873, CMS1, normal](./text/Data/stringsel3.png)
String GSM916761, CMS4, normal
![String GSM916761, CMS4, normal](./text/Data/stringsel4.png)

# pL-UFG
Before running our code:
1. Please download the packages from: https://github.com/guoji-fu/pGNNs
2. Please merge "src" from https://github.com/guoji-fu/pGNNs and pL-UFG
3. Since we made a modification on the "data_proc.py" and "main.py", please using ours for running the experiment.

## Framelet Structure:

![framelet](https://user-images.githubusercontent.com/54494470/203444583-b69fcd4d-9a4e-44e5-a10d-a00a7e9b82f2.png)
Fig.1 The above figure shows the framelet framework by giving a graph with structure (adjacency matrix) and feature information.

## Instruction of Adjustable p value:

![regularization_new](https://user-images.githubusercontent.com/54494470/212585216-873a5702-43fa-4a0f-9075-81b897a295e5.png)

Fig.2 The figure above shows the working flow of the $p$-Laplacian regularized framelet. The input graph data is first filtered and reconstructed by the framelet model; then, the result is further regularized by a sequence of graph convolution and diagonal rescaling induced by the $p$-Laplacian, which is generated based on the graph gradient information, serving as an implicit layer of the model. By adjusting the $p$ value, node features resulting from this implicit layer can be smoothed or sharpened accordingly, thus making the model adopt both homophily and heterophilic graphs. Lastly, the layer output will then be either forwarded to the task objective function or to the next framelet and $p$-Laplacian layers before the final prediction task. 



## More experiment results are presented as follows:
![image](https://user-images.githubusercontent.com/54494470/195551109-7209f63f-934d-4180-a428-afa617e06ce0.png)

![image](https://user-images.githubusercontent.com/54494470/195551259-39be6510-83e9-44bf-91ef-6166371a9131.png)

Noise tolerancy test:

<img width="862" alt="image" src="https://user-images.githubusercontent.com/54494470/212585034-2b8b958b-0884-4359-9460-30d4f14de877.png">
Denoising power on heterophilic graph (Chameleon)

<img width="827" alt="image" src="https://user-images.githubusercontent.com/54494470/212585637-debb9f83-e61a-4e18-bb74-45d1eccd9df9.png">
Denoising power on homophilic graph (Cora)




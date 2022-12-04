# pL-UFG
Before running our code:
1. Please download the packages from: https://github.com/guoji-fu/pGNNs
2. Please merge "src" from https://github.com/guoji-fu/pGNNs and pL-UFG
3. Since we made a modification on the "data_proc.py" and "main.py", please using ours for running the experiment.

## Framelet Structure:

![framelet](https://user-images.githubusercontent.com/54494470/203444583-b69fcd4d-9a4e-44e5-a10d-a00a7e9b82f2.png)
Fig.1 The above figure shows the framelet framework by giving a graph with structure (adjacency matrix) and feature information.

## Instruction of Adjustable p value:

![regularization](https://user-images.githubusercontent.com/54494470/205472541-b0f6ef48-8de8-4c5c-b2d8-26ad94d62e1f.png)

Fig2. The above figure shows that the $p$-Laplacian based regularization framework interacts with the framelet when the value of $p$ is 1 and 2. The left side of the figure shows how the penalty term  S<sub>p</sub><sup>φ</sup> is built based on graph gradient information, and the right side of the figure shows the differences in terms of the range of solution space (presented as the level set of F) due to different values of p. One can clearly see that a higher penalty (i.e., p=2, presented as the bold circle in the middle) term intersects the framelet solution space at the inner circle of its level set. In contrast, a lower penalty term (i.e., p=1, presented as a shaded square at the middle ) maintains a higher variation of the framelet solution space by only touching the out circles of the framelet solution level sets. 

## More experiment results are presented as follows:
![image](https://user-images.githubusercontent.com/54494470/195551109-7209f63f-934d-4180-a428-afa617e06ce0.png)

![image](https://user-images.githubusercontent.com/54494470/195551259-39be6510-83e9-44bf-91ef-6166371a9131.png)

Noise tolerancy test:

![image](https://user-images.githubusercontent.com/54494470/196026683-14b5ab8a-a949-42b6-bfbb-65462f972782.png)


![image](https://user-images.githubusercontent.com/54494470/196026700-c3f0de47-ca47-4097-b3cb-f03446f7c768.png)


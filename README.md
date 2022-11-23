# pL-UFG
Before running our code:
1. Please download the packages from: https://github.com/guoji-fu/pGNNs
2. Please merge "src" from https://github.com/guoji-fu/pGNNs and pL-UFG
3. Since we made a modification on the "data_proc.py" and "main.py", please using ours for running the experiment.

## Framelet Structure:

![framelet](https://user-images.githubusercontent.com/54494470/203444583-b69fcd4d-9a4e-44e5-a10d-a00a7e9b82f2.png)
Fig.1 The above figure shows the framelet framework by giving a graph with structure (adjacency matrix) and feature information.

## Instruction of Adjustable p value:

<img width="1350" alt="regular" src="https://user-images.githubusercontent.com/54494470/203444665-f6563ab3-7a34-43a8-aacc-d6bf26d1ba06.png">
Fig2. The above figure shows the p-Laplacian based regularization framework and provides a double filtering with quantities of p (p = 1 and p = 2). The
first (left) and second (middle) graphs show how the penalty term   S<sub>p</sub><sup>φ</sup> is built based on graph gradient information and the third (right) graph shows the differences in terms of the range of solution space (presented as the level set of F) due to different choices of p. It is clear to see that when the model is regularized with a lower penalty term (p = 1), one can only obtain the solution of the model from the outer-ring level set compared to its higher penalty term counterpart.


## More experiment results are presented as follows:
![image](https://user-images.githubusercontent.com/54494470/195551109-7209f63f-934d-4180-a428-afa617e06ce0.png)

![image](https://user-images.githubusercontent.com/54494470/195551259-39be6510-83e9-44bf-91ef-6166371a9131.png)

Noise tolerancy test:

![image](https://user-images.githubusercontent.com/54494470/196026683-14b5ab8a-a949-42b6-bfbb-65462f972782.png)


![image](https://user-images.githubusercontent.com/54494470/196026700-c3f0de47-ca47-4097-b3cb-f03446f7c768.png)


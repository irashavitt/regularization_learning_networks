First off, thanks for your interest in Regularization Learning Networks!

I hope you will find this code useful. If you do use it, and improve it along the way, I would very much appreciate it if you take the time and contribute these improvements. Here are some ways in which the current implementation could be improved:
* Implementation of RLN on more platforms: TensorFlow would be a good start.
* Optimization: Currently, the code doesn't access the gradients directly, and has to infer them from the current weights and the previous weights. Modifying the code to access the gradients while the weights are calculated would improve the code greatly.

Ira.

# Is There a Standard Heuristic for Model Tuning?


* Training error should steadily decrease, steeply at first, and should eventually plateau as training converges.  
* If the training has not converged, try running it for longer. 
* If the training error decreases too slowly, increasing the learning rate may help it decrease faster.
    * But sometimes the exact opposite may happen if the learning rate is too high.
* If the training error varies wildly, try decreasing the learning rate.
    * Lower learning rate plus larger number of steps or larger batch size is often a good combination.
* Very small batch sizes can also cause instability.  First try larger values like 100 or 1000, and decrease until you see degradation.

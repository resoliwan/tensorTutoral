### steps
which is the total number of training iterations. One step calculates the loss from one batch and uses that value to modify the model's weights once.

### batch size
which is the number of examples (chosen at random) for a single step. For example, the batch size for SGD is 1.
\\[ \text{# of trained examples} = \text{batch size} * steps \\]

### periods
which controls the granularity of reporting. For example, if periods is set to 7 and steps is set to 70, then the exercise will output the loss value every 10 steps (or 7 times). Unlike hyperparameters, we don't expect you to modify the value of periods. Note that modifying periods does not alter what your model learns.

\\[ \text{# of training examples in each  period}  = \frac{\text{batch size} * steps}{periods} \\] 



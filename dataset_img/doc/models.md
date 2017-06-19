# Working with models

A batch class might also include model definitions.

## Model definition
A model definition method
- is marked with a `@model` decorator
- does not take any arguments (even `self`)
- returns a model descriptor.

```python
class MyArrayBatch(ArrayBatch):
    ...
    @model()
    def basic_model():
        input_data = tf.placeholder('float', [None, 28])
        model_output = ...
        cost = tf.reduce_mean(tf.square(data - model_output))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        return [input_data, optimizer]
```
It is for you to decide what the model descriptor is. It might be:
- a list of TensorFlow placeholders, optimizers and other variables you need (e.g. a loss function value or a graph).
- a Keras model
- an mxnet module

or anything else.

Later you will get back this descriptor in a model-based actions method. So you have to include in it everything you need to train and evaulate the model.

Important notes:
1. Do not forget parenthesis in a decorator - `@model()`, not `@model`.
1. You should never call model definition methods. They are called internally.

## Model-based actions
After a model is defined, you might use it to train, evaluate or predict.

```python
class MyArrayBatch(ArrayBatch):
    ...
    @action(model='basic_model')
    def train_model(self, model, session):
        input_data, optimizer = model
        session.run([optimizer], feed_dict={input_data: self.data})
        return self
```
You add to an `@action` decorator an argument `model` with a model definition method name.

Later, you just put this action into a pipeline:
```python
full_workflow = my_dataset.p
                          .load('/some/path')
                          .some_preprocessing()
                          .some_augmentation()
                          .train_model(session=sess)
```
You do not need to pass a model into this action. The model is saved in an internal model directory and then passed to all actions based on this model.

You might have several actions based on the very same model.
```python
class MyArrayBatch(ArrayBatch):
    ...
    @action(model='basic_model')
    def train_model(self, model, session):
        ...

    @action(model='basic_model')
    def evaluate_model(self, model, session):
        ...

full_workflow = my_dataset.p
                          .load('/some/path')
                          .some_preprocessing()
                          .some_augmentation()
                          .train_model(session=sess)
                          .evaluate_model(session=sess)
```

### Parallel training with TensorFlow
If you [prefetch](prefetch.md) with actions based on Tensorflow models you might encounter that your model hardly learns anything. The reason is that TF variables do not update concurrently. To solve this problem make an action `a singleton` which allows only one concurrent execution:
```python
class MyBatch:
    ...
    @action(model='some_model', singleton=True)
    def train_it(self, model, sess):
        input_images, input_labels = model[0]
        optimizer, cost, accuracy = model[1]
        _, loss = sess.run([optimizer, cost], feed_dict={input_images: self.images, input_labels: self.labels})
        return self
```

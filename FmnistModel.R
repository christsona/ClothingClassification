library(keras)

fmnist = dataset_fashion_mnist()
train_images = fmnist$train$x
train_labels = fmnist$train$y
test_images = fmnist$test$x
test_labels = fmnist$test$y

dim(train_images)
dim(test_images)

str(train_images)
str(test_images)

train_images = array_reshape(train_images, c(60000, 28 * 28))
train_images = train_images/255

test_images = array_reshape(test_images, c(10000, 28 * 28))
test_images = test_images/255

train_labels = to_categorical(train_labels, num_classes = 11)
test_labels = to_categorical(test_labels, num_classes = 11)

network = keras_model_sequential() %>% 
  layer_dense(units=16, activation="selu", initializer_he_normal(), input_shape=c(28*28)) %>% 
  layer_dense(units=16, activation = "selu", initializer_he_normal()) %>% 
  layer_dense(units=11, activation="softmax")

network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

network %>% fit(train_images, train_labels, epochs=10, batch_size=256, validation_split=.2)

metrics = network %>% evaluate(test_images, test_labels)
metrics

network %>% predict_classes(test_images[1:10,])
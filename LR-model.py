import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt

x = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.00,1.10])
y = np.array([0.096,0.117,0.135,0.182,0.196,0.216,0.241,0.286,0.313,0.351])

x_predict = np.array([1.2,1.3,1.40,1.50,1.60,1.70,1.80,1.90,2.00])

model = Sequential()
model.add(Dense(1, input_dim=1, kernel_initializer='random_uniform'))
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=200)

#model.save('fine_LR_model.h5')

prediction = model.predict(x=x_predict, batch_size=10)
print(prediction)
plt.scatter(x_predict, prediction)
plt.show()
#aaa

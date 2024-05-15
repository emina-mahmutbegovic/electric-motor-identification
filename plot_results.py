from keras.saving import load_model
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def plot_results(var1, var2, label1='Variable id_k', label2='Variable iq_k'): 
        chunk_size = 100
        # Determine the number of chunks
        num_chunks = len(var1) // chunk_size
        
        for i in range(num_chunks):
            # Define the range for the current chunk
            start = i * chunk_size
            end = start + chunk_size

            time = np.linspace(start, 10, chunk_size)  # 100 time points from 0 to 10

            # Create a plot
            plt.figure(figsize=(10, 5))  # Set the figure size

            # Plot both variables
            plt.plot(time, var1[start:end], label=label1, color='blue')
            plt.plot(time, var2[start:end], label=label2, color='red')

            # Adding title and labels
            plt.title('The first 100 target variables over time')
            plt.xlabel('Time')
            plt.ylabel('Value')

            # Add a legend to the plot
            plt.legend()

            # Show the plot
            plt.show()

            # Ask the user whether to continue
            answer = input("Continue plotting the next chunk? (y/n): ")
            if answer.lower() != 'y':
                print("Plotting stopped.")
                break

model = load_model('rmsprop_lstm.h5')  # If saved as HDF5
X = np.load('X.npy')
y = np.load('y.npy')
# Load the scaler from the file
scaler = joblib.load('scaler_lstm.save')

# Initialize scores variable
scores = []
#visualkeras.layered_view(model).show() # display using your system viewer

# Predict results
pred = model.predict(X)
#pred = np.reshape(pred,(400,2))
y = np.reshape(y,(400,2))
pred = np.reshape(pred, (400,2))
 # Perform inverse transformation
pred = scaler.inverse_transform(pred)
gtruth = scaler.inverse_transform(y)
mse = mean_squared_error(pred, gtruth)
rmse = np.sqrt(mse)
print("Broj parametara u modelu:", model.count_params())
print(rmse) 

id_k1 = [sublist[0] for sublist in gtruth]
id_k_real = [sublist[0] for sublist in X]
id_k_1_real = [sublist[0] for sublist in y]
iq_k1 = [sublist[1] for sublist in gtruth]

id_k_pred = [sublist[0] for sublist in pred]
iq_k_pred = [sublist[1] for sublist in pred]

# Plot results
plot_results(id_k1, id_k_pred, 'Variable id_k+1', 'Predicted variable id_k+1')
plot_results(iq_k1, iq_k_pred, 'Variable iq_k+1', 'Predicted variable iq_k+1')



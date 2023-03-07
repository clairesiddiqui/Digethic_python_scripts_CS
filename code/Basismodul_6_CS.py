#================================================================#
#                EXERCISE 3 - PYTHON TUTORIAL
#================================================================#

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from matplotlib import cm
from pylab import plot,show


# CREATE ARRAY FILE:
studying = np.array([84,1323,282,957,1386,810,396,474,501,660,1260,1005,1110,1290])
result = np.array([44,97,30,51,95,51,44,41,21,40,90,83,61,92])
data = np.column_stack((studying, result))


# CREATE ARRAYS FOR w AND b:
w = np.linspace(0, 0.1, 5, endpoint= True)   # values:  0, 0.025, 0.05, 0.075,  0.1
b = np.linspace(15, 25, 5, endpoint= True)   # values: 15,  17.5,   20,  22.5,   25


# CREATE DATAFRAMES FOR DIFFERENT COMBINATIONS OF w AND b TO ITERATE OVER:
model = np.add(np.zeros(len(result)*5).reshape((len(result)),5),-100)
data_0 = np.column_stack((data, model))
data_1 = np.column_stack((data, model))
data_2 = np.column_stack((data, model))
data_3 = np.column_stack((data, model))
data_4 = np.column_stack((data, model))
data_5 = np.column_stack((data, model))

i = 0
x = 0
for i in range(0,len(result)):
    for x in range(0, len(w)):
                   data_0[i,2+x]   = data_0[i,0] * w[x]
                   data_1[i,2+x]   = data_1[i,0] * w[0] + b[x]
                   data_2[i,2+x]   = data_2[i,0] * w[1] + b[x]
                   data_3[i,2+x]   = data_3[i,0] * w[2] + b[x]
                   data_4[i,2+x]   = data_4[i,0] * w[3] + b[x]
                   data_5[i,2+x]   = data_5[i,0] * w[4] + b[x]
                   #print("value w:", w[x], "for row:", i)
print(data_0)                
print(data_1)
print(data_2)



# CALCULATE ERROR TERMS FOR DIFFERENT COMBINATIONS OF w AND b:
def mse(y_modelled,y):
    return np.mean((y_modelled-y)**2)

data_2_2_error = mse(data_2[:,2],data_0[:,1])
data_3_2_error = mse(data_3[:,2],data_0[:,1])
data_4_2_error = mse(data_4[:,2],data_0[:,1])
data_5_2_error = mse(data_5[:,2],data_0[:,1])

data_2_3_error = mse(data_2[:,3],data_0[:,1])
data_3_3_error = mse(data_3[:,3],data_0[:,1])
data_4_3_error = mse(data_4[:,3],data_0[:,1])
data_5_3_error = mse(data_5[:,3],data_0[:,1])

data_2_4_error = mse(data_2[:,4],data_0[:,1])
data_3_4_error = mse(data_3[:,4],data_0[:,1])
data_4_4_error = mse(data_4[:,4],data_0[:,1])
data_5_4_error = mse(data_5[:,4],data_0[:,1])

data_2_5_error = mse(data_2[:,5],data_0[:,1])
data_3_5_error = mse(data_3[:,5],data_0[:,1])
data_4_5_error = mse(data_4[:,5],data_0[:,1])
data_5_5_error = mse(data_5[:,5],data_0[:,1])

fig1 = plt.figure()
ax = fig1.add_subplot(221)
fig1.suptitle("Linear Regression")
ax.scatter(data_0[:,0], data_0[:,1], color = "grey", label = "target")
ax.plot(data_2[:,0], data_2[:,2], color = "red",  label =  'w=0.025, b=15' + '  ' + str('{:.3f}'.format(data_2_2_error)) )
ax.plot(data_3[:,0], data_3[:,2], color = "blue", label = "w=0.05, b=15"   + '  ' + str('{:.3f}'.format(data_3_2_error)))
ax.plot(data_4[:,0], data_4[:,2], color = "green",label = "w=0.075, b=15"  + '  ' + str('{:.3f}'.format(data_4_2_error)))
ax.plot(data_5[:,0], data_5[:,2], color = "purple",label = "w=0.1, b=15"   + '  ' + str('{:.3f}'.format(data_5_2_error)))
ax.set_ylabel('result')
ax.set_xlabel('studying')
ax.legend(loc=2, frameon=False)

ax1 = fig1.add_subplot(222)
ax1.scatter(data_0[:,0], data_0[:,1], color = "grey", label = "target")
ax1.plot(data_2[:,0], data_2[:,3], color = "red",  label = "w=0.025, b=17.5" + '  ' + str('{:.3f}'.format(data_2_3_error)))
ax1.plot(data_3[:,0], data_3[:,3], color = "blue", label = "w=0.05, b=17.5"  + '  ' + str('{:.3f}'.format(data_3_3_error)))
ax1.plot(data_4[:,0], data_4[:,3], color = "green",label = "w=0.075, b=17.5" + '  ' + str('{:.3f}'.format(data_4_3_error)))
ax1.plot(data_5[:,0], data_5[:,3], color = "purple",label = "w=0.1, b=17.5"  + '  ' + str('{:.3f}'.format(data_5_3_error)))
ax1.set_ylabel('result')
ax1.set_xlabel('studying')
ax1.legend(loc=2, frameon=False)

ax2 = fig1.add_subplot(223)
ax2.scatter(data_0[:,0], data_0[:,1], color = "grey", label = "target")
ax2.plot(data_2[:,0], data_2[:,4], color = "red",  label = "w=0.025, b=20"  + '  ' + str('{:.3f}'.format(data_2_4_error)))
ax2.plot(data_3[:,0], data_3[:,4], color = "blue", label = "w=0.05, b=20"   + '  ' + str('{:.3f}'.format(data_3_4_error)))
ax2.plot(data_4[:,0], data_4[:,4], color = "green",label = "w=0.075, b=20"  + '  ' + str('{:.3f}'.format(data_4_4_error)))
ax2.plot(data_5[:,0], data_5[:,4], color = "purple",label = "w=0.1, b=20"   + '  ' + str('{:.3f}'.format(data_5_4_error)))
ax2.set_ylabel('result')
ax2.set_xlabel('studying')
ax2.legend(loc=2, frameon=False)

ax3 = fig1.add_subplot(224)
ax3.scatter(data_0[:,0], data_0[:,1], color = "grey", label = "target")
ax3.plot(data_2[:,0], data_2[:,5], color = "red",  label = "w=0.025, b=22.5"  + '  ' + str('{:.3f}'.format(data_2_5_error)))
ax3.plot(data_3[:,0], data_3[:,5], color = "blue", label = "w=0.05, b=22.5"   + '  ' + str('{:.3f}'.format(data_3_5_error)))
ax3.plot(data_4[:,0], data_4[:,5], color = "green",label = "w=0.075, b=22.5"  + '  ' + str('{:.3f}'.format(data_4_5_error)))
ax3.plot(data_5[:,0], data_5[:,5], color = "purple",label = "w=0.1, b=22.5"   + '  ' + str('{:.3f}'.format(data_5_5_error)))
ax3.set_ylabel('result')
ax3.set_xlabel('studying')
ax3.legend(loc=2, frameon=False)

#plt.show()




# IMPLEMENT POLYNOMIAL REGRESSION MODEL (Degree 1-7):
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.scatter(studying, result)
ax.set_title("Polynomial Regression")
ax.set_ylabel("result")
ax.set_xlabel("studying")
x_axis = np.linspace(0, 1500)

for i in range(1, 8):
    poly = PolynomialFeatures(i)
    x_poly = poly.fit_transform(studying.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly,result)
    prediction = model.predict(poly.transform(studying.reshape(-1, 1)))
    error = mse(prediction, result)
    
    predicted_plot = model.predict(poly.transform(x_axis.reshape(-1, 1)))
    ax.plot(x_axis, predicted_plot, label="Polynomgrad={}, Fehler: {}".format(i, round(error,3)))
ax.legend()

#plt.show()




# IMPLEMENT LOGISTIC REGRESSION MODEL (for classified data):
studying = np.array([84,1323,282,957,1386,810,396,474,501,660,1260,1005,1110,1290])
result = np.array([44,97,30,51,95,51,44,41,21,40,90,83,61,92])
result_binary = np.array([1 if i >= 50 else 0 for i in result])

def plot_original_data(ax, x, y):
    ax.scatter(x[np.argwhere(y>=50)].ravel(),y[np.argwhere(y>=50)].ravel(),c="green",label="Bestanden")
    ax.scatter(x[np.argwhere(y<50)].ravel(),y[np.argwhere(y<50)].ravel(),c="red",label="Nicht bestanden")

fig3 = plt.figure()
ax = fig3.add_subplot(3, 3, 1)
ax.set_title("Original data")
ax.scatter(studying, result)
ax = fig3.add_subplot(3, 3, 2)
ax.set_title("Binary data")
ax.scatter(studying, result_binary)
for i in range(1, 8):
    
    # Plot the original data
    ax = fig3.add_subplot(3, 3, i+2)
    ax.set_title("Polynomgrad {}".format(i))
    plot_original_data(ax, studying, result)
    
    # Train the model
    poly = PolynomialFeatures(i)
    x_poly = poly.fit_transform(studying.reshape(-1, 1))
    model = LogisticRegression()
    model.fit(x_poly, result_binary)
    
    # Predict the studying   --> "predict the results based on studying"
    predicted = model.predict(poly.transform(studying.reshape(-1, 1)))
    score = model.score(x_poly, result_binary)
    print("Genauigkeit für Polynomgrad {}: {} Prozent.".format(i,score*100))

    # Visualise the model        
    x_axis, y_axis = np.linspace(0, 1400), np.linspace(0, 100)
    x1, x2 = np.meshgrid(x_axis, y_axis)
    z = np.array([model.predict_proba(poly.transform(x_axis.reshape(-1, 1))).T[1] for y in y_axis])  
    ax.contourf(x1,x2,z.reshape(50,50),cmap=cm.RdYlGn,levels=50,alpha=0.3)

plt.show()
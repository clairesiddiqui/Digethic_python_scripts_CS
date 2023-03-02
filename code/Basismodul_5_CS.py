#================================================================#
#    BASISMODUL #5 - STATISTISCHE GRUNDLAGEN - PYTHON TUTORIAL
#================================================================#

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import plot,show


# LOAD .CSV FILE
census_file = '/Users/csi/private/Data_Scientist/Digethic/full-stack-machine-learning/data/census.csv'
census = pd.read_csv(census_file)
#print(census)

import pdb; pdb.set_trace()


# GET INFO ON FILE
print(census.describe())

# DECLARE NUMERICAL VALUES 
sex_dict = {' Male': 1, ' Female': 0}
census["sex"] = census["sex"].replace(sex_dict)

race_dict = {' Asian-Pac-Islander': 4, ' White': 3, ' Black': 2, ' Amer-Indian-Eskimo':1, ' Other': 0}
census["race"] = census["race"].replace(race_dict)

target_dict = {' >50K': 1, ' <=50K': 0}
census["target"] = census["target"].replace(target_dict)

print("Census head:", census.head())
#['age', 'workclass', 'education', 'marital-status','occupation','relationship','race', 'sex', 'capital-gain',
# 'capital-loss', 'hours-per-week', 'native-country', 'target']


# CREATE NUMPY ARRAY FILE
census_data = census[['age','race','sex','capital-gain','capital-loss','hours-per-week','target']].to_numpy()
print("Shape of census file:", np.shape(census_data))

census_data_female = np.where(census_data[:,2] == 0) [0]
census_data_male   = np.where(census_data[:,2] == 1) [0]


# MORE STATISTICS
# number of male vs female people earning more than 50k

total_female = len(census_data[census_data_female,:])
total_male   = len(census_data[census_data_male,:])

census_female_rich = np.where(census_data[census_data_female,6] == 1) [0]
census_male_rich   = np.where(census_data[census_data_male,6]   == 1) [0]

rich_female = len(census_data[census_female_rich,:])
rich_male   = len(census_data[census_male_rich,:])

print("PERCENTAGE OF RICH FEMALES:", rich_female / (total_female/100))
print("PERCENTAGE OF RICH MALES:", rich_male / (total_male/100))




# PLOT HISTOGRAMS
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,6))

axs[0,0].hist(census_data[census_data_female,0], histtype='step', color='red')
axs[0,0].hist(census_data[census_data_male,0],   histtype='step', color='blue')
axs[0,0].set_xlabel('age', color='black', fontsize=11)
axs[0,0].set_ylabel('frequency', color='black', fontsize=11)

axs[0,1].hist(census_data[census_data_female,1], histtype='step', color='red')
axs[0,1].hist(census_data[census_data_male,1],   histtype='step', color='blue')
axs[0,1].set_xlabel('race', color='black', fontsize=11)
#axs[0,1].set_ylabel('frequency', color='black', fontsize=11)

axs[0,2].hist(census_data[census_data_female,3], histtype='step', color='red')
axs[0,2].hist(census_data[census_data_male,3],   histtype='step', color='blue')
axs[0,2].set_xlabel('capital-gain', color='black', fontsize=11)
#axs[0,2].set_ylabel('frequency', color='black', fontsize=11)

axs[1,0].hist(census_data[census_data_female,4], histtype='step', color='red')
axs[1,0].hist(census_data[census_data_male,4],   histtype='step', color='blue')
axs[1,0].set_xlabel('capital-loss', color='black', fontsize=11)
axs[1,0].set_ylabel('frequency', color='black', fontsize=11)

axs[1,1].hist(census_data[census_data_female,5], histtype='step', color='red')
axs[1,1].hist(census_data[census_data_male,5],   histtype='step', color='blue')
axs[1,1].set_xlabel('hours-per-week', color='black', fontsize=11)
#axs[1,1].set_ylabel('frequency', color='black', fontsize=11)

axs[1,2].hist(census_data[census_data_female,6], histtype='step', color='red')
axs[1,2].hist(census_data[census_data_male,6],   histtype='step', color='blue')
axs[1,2].set_xlabel('target', color='black', fontsize=11)
#axs[1,2].set_ylabel('frequency', color='black', fontsize=11)





# PLOT BOXPLOTS
fig1 = plt.figure()
ax1 = fig1.add_subplot(221)
box = ax1.boxplot([census_data[census_data_female,3], census_data[census_data_male,3]], labels = ['female', 'male'], patch_artist=True, showmeans=True)
ax1.set_ylabel('capital-gain')
colors = ['red', 'blue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)


ax2 = fig1.add_subplot(223)
box = ax2.boxplot([census_data[census_data_female,4], census_data[census_data_male,4]], labels = ['female', 'male'], patch_artist=True, showmeans=True)
ax2.set_ylabel('capital-loss')
colors = ['red', 'blue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

ax3 = fig1.add_subplot(222)
box = ax3.boxplot([census_data[census_data_female,5], census_data[census_data_male,5]], labels = ['female', 'male'], patch_artist=True, showmeans=True)
ax3.set_ylabel('hours-per-week')
colors = ['red', 'blue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

ax4 = fig1.add_subplot(224)
box = ax4.boxplot([census_data[census_data_female,6], census_data[census_data_male,6]], labels = ['female', 'male'], patch_artist=True, showmeans=True)
ax4.set_ylabel('target')
colors = ['red', 'blue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.show()





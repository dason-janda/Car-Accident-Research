import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_excel("data/AccidentData21-26.csv.xlsx")

df.columns = df.columns.str.strip()
print(df.columns.tolist())

d = {'Fatal': 1, 'Suspected Serious Injury': 1}
df['Crash Severity'] = df['Crash Severity'].map(d).fillna(0)
y = df['Crash Severity']

features = [
    'Roadway Junction Type', 
    'Light Condition', 
    'Weather Condition', 
    'Roadway Surface Condition', 
    'Number of Vehicles Involved', 
    'Route Type', 
    'Intersection Involved', 
    'Aggressive Driving Involved', 
    'Bicycle Involved', 
    'Collision With Fixed Object', 
    'Commercial Motor Vehicle Involved', 
    'Crash Type', 
    'Disregard Traffic Control Device Involved', 
    'Distracted Driving Involved', 
    'Drowsy Driving Involved', 
    'DUI Involved', 
    'Estimated Travel Speed', 
    'Hit and Run', 
    'Manner of Collision', 
    'Motorcycle Involved', 
    'Pedestrian Involved', 
    'Posted Speed', 
    'Roadway Departure Crash', 
    'Roadway Geometry Involved', 
    'Roadway Type', 
    'Speed Related', 
    'Unrestrained Involved', 
    'Work Zone Involved', 
    'Wrong Way Driving Involved'
]

X = df[features]

X = pd.get_dummies(X, drop_first=True)

dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree = dtree.fit(X, y)

plt.figure(figsize=(24, 12))
tree.plot_tree(
    dtree, 
    feature_names=X.columns, 
    class_names=['Not Severe', 'Severe/Fatal'], 
    filled=True, 
    rounded=True, 
    fontsize=10
)
plt.savefig("crashFactorsTree.png")
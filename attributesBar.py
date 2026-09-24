import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('./data/AccidentData21-26.csv.xlsx')

counts = df["Crash Attributes"].dropna().str.split(",").explode().str.strip('[]"').value_counts()

print(counts)

plt.barh(counts.index, counts.values)

plt.title("Crash Attributes in Crash Data")
plt.xlabel("Quantity")
plt.ylabel("Values")
plt.tight_layout()

plt.savefig("CrashAttributesBar.png")
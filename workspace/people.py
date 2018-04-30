import os, pandas as pd
import _tkinter
import matplotlib.pyplot as plt
csv_path = os.path.join("./", "people.csv")
people = pd.read_csv(csv_path)
print(people[["last_name"]])


print(people[people.age > 20][["height","weight"]])

print(people.plot(kind="scatter",x="height",y="weight"))
plt.show(block=False)


print(people[["height","weight", "age"]].corr())

firstname = input("First Name ")
lastname = input("Last Name ")
print(people[(people.first_name == firstname) & (people.last_name == lastname)][["age"]])

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

flights = sns.load_dataset("flights")
print(flights)

may_flights = flights.query("month == 'May'")
# sns.lineplot(data=may_flights, x="year", y="passengers")
# sns.lineplot(data=flights, x="month", y="passengers")
x = [i for i in range(100)]
y = [i for i in range(100)]
x = np.concatenate([x, x])
y2 = [i + np.random.randint(100) for i in range(100)]
y = np.concatenate([y, y2])
sns.lineplot(x, y)
plt.show()

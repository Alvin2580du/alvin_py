import json
import matplotlib.pyplot as plt
import requests

save = []
method = 'network'  # change the method where you want to get data

if method == 'network':
    url = "https://think.cs.vt.edu/corgis/json/aids/aids.json"
    data = requests.get(url).json()
else:
    with open("aids.json", 'r') as fr:
        data = json.load(fr)

for x in data:
    print(x['Data'].keys())
    res = x['Data']['HIV Prevalence']['Adults']
    save.append(res)

plot_data = save[:100]
print(plot_data)

y = range(len(plot_data))

plt.figure()
plt.plot(y, plot_data)
plt.savefig("line plot.png")
plt.show()
plt.close()

plt.figure()
plt.scatter(y, plot_data)
plt.savefig("scatter plot.png")
plt.show()
plt.close()

plt.figure()
plt.boxplot(y, plot_data)
plt.savefig("scatter plot.png")
plt.show()
plt.close()


m = sum(plot_data) / len(plot_data)
print("mean of data:{}".format(m))

plot_data.sort()
median = plot_data[len(plot_data) // 2]
print("median:{}".format(median))

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 

gas = pd.read_csv('./matplotlib/Example/gas_prices.csv')

plt.figure(figsize=(8,5))

plt.title('Gas Prices over Time (in USD)', fontdict={'fontweight':'bold', 'fontsize': 18})


plt.plot(gas.Year, gas.USA, 'b.-', label='United States')
plt.plot(gas.Year, gas.Canada, 'r.-')
plt.plot(gas.Year, gas['South Korea'], 'g.-')
plt.plot(gas.Year, gas.Australia, 'y.-')

# Another Way to plot many values!
countries_to_look_at = ['Australia', 'USA', 'Canada', 'South Korea']
for country in gas:
    if country != 'Year':
        plt.plot(gas.Year, gas[country], marker='.')

# 3 nam 1 lan 
plt.xticks(gas.Year[::3].tolist()+[2011])

plt.xlabel('Year')
plt.ylabel('US Dollars')

plt.legend()

plt.savefig('Gas_price_figure.png', dpi=300)

plt.show()
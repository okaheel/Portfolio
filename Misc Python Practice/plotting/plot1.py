import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("company_sales_data.csv")

profitList = df['total_profit'].tolist()
monthList  = df ['month_number'].tolist()

plt.plot(monthList, profitList, label = 'Month-wise Profit data of last year')
plt.show()
plt.xlabel('Month number')
plt.ylabel('Profit in dollar')
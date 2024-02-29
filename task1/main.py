import pandas as pd

from neural_network.task1.GainRation import GainRation

if __name__ == "__main__":
    laptop_df = pd.read_csv('./resources/laptop_price.csv')

    laptop_df['Price'] = pd.cut(laptop_df["Price"].astype(float),
                                bins=[0, 10750, 29000, float('inf')],
                                labels=["Inexpensive", "Medium", "Expensive"])

    laptop_df['Weight'] = pd.cut(laptop_df["Weight"].astype(float),
                                 bins=[0.0, 2.95, 4.0, float('inf')],
                                 labels=["Easy", "Medium", "Heavy"])

    laptop_df['Screen_Size'] = pd.cut(laptop_df["Screen_Size"].astype(int),
                                      bins=[0, 12, 14, float('inf')],
                                      labels=["Small", "Medium", "Large"])

    laptop_df['Processor_Speed'] = pd.cut(laptop_df["Processor_Speed"].astype(float),
                                          bins=[0, 2.3, 3.15, float('inf')],
                                          labels=["Small", "Medium", "Fast"])

    print('gain ration weight: ', GainRation.gain_ration(laptop_df, 'Weight', 'Price'))
    print('gain ration RAM size: ', GainRation.gain_ration(laptop_df, 'RAM_Size', 'Price'))
    print('gain ration storage capacity: ', GainRation.gain_ration(laptop_df, 'Storage_Capacity', 'Price'))
    print('gain ration screen size: ', GainRation.gain_ration(laptop_df, 'Screen_Size', 'Price'))
    print('gain ration processor speed: ', GainRation.gain_ration(laptop_df, 'Processor_Speed', 'Price'))
    print('gain ration brand: ', GainRation.gain_ration(laptop_df, 'Brand', 'Price'))

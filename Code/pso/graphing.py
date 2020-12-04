import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("GPU_sharedmem.csv")

df = df.iloc[:, 1:]

columns = ['totalTime', 'totalInitTime', 'totalFitnessTime', 
            'totalUpdateTime', 'fitnessPerParticle', 'updatePerParticle']

for col in columns:
    for i in df['n'].unique():
        dftemp = df[df['n'] == i]
        plt.plot(dftemp['particles'], dftemp[col], label = i)

    plt.xlabel('particles')
    plt.ylabel(col)
    plt.title(col + ' vs particles for different n. Shared Memory')
    plt.legend()
    plt.show()
    
    
# for col in columns:
    # for i in df['particles'].unique():
        # dftemp = df[df['particles'] == i]
        # plt.plot(dftemp['n'], dftemp[col], label = i)

    # plt.xlabel('n')
    # plt.ylabel(col)
    # plt.title(col + ' vs n for different particles')
    # plt.legend()
    # plt.show()



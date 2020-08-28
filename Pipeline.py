from Preprocessing import Pipeline
import pandas as pd
import Config

data=pd.read_csv('houseprice.csv')
data=data[Config.features+[Config.target]].copy()
pipeline=Pipeline(target=Config.target,
                    features=Config.features,
                    data=data,split_pct=Config.split_pct)

pipeline.fit(data)
print('Training Completed')
pipeline.evaluate()

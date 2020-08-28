
categorical_variables=['BldgType', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'CentralAir', 'ExterQual', 'GarageFinish', 'GarageType', 'HeatingQC', 'KitchenQual',
 'LotShape', 'PavedDrive', 'SaleCondition']

categories_to_Impute=['BsmtFinType1', 'BsmtExposure', 'BsmtQual', 'GarageFinish', 'GarageType']

Numerics_to_impute=[]

Years_toBe_Transformed=['YearBuilt', 'YearRemodAdd']

features=['YrSold','OverallQual', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'GrLivArea','BsmtFullBath', 'Fireplaces', 'GarageCars',
 'LotShape', 'BldgType','ExterQual', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'HeatingQC','CentralAir', 'KitchenQual',
  'GarageType', 'GarageFinish', 'PavedDrive','SaleCondition']


LogTransormFeatures=['GrLivArea', '1stFlrSF']

split_pct=0.1

target='SalePrice'
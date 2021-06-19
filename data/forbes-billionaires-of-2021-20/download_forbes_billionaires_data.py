# os.system('pip install kaggle')

import zipfile 
os.system('kaggle datasets download alexanderbader/forbes-billionaires-of-2021-20/forbes_billionaires.csv')
archive = zipfile.ZipFile('forbes-billionaires-of-2021-20.zip')
filename = archive.filelist[0].filename
xlfile = archive.open(filename)
df = pd.read_csv(xlfile)


import pandas as pd
df_new = pd.read_csv("https://raw.githubusercontent.com/srhoads/learn/main/data/forbes-billionaires-of-2021-20/forbes_billionaires.csv")


import pycountry_convert as pc #os.system("pip install pycountry-convert")
import re
# df_new = df.copy().apply(lambda y: y.str.lower() if str(y.dtype)=='object' else y).assign(
    #children=lambda d: [recode_onehot_if(x, 'children') for x in d.Description],
    # purchase_price = lambda d: d.UnitPrice * d.Quantity,
    # invoice_month = lambda d: pd.DatetimeIndex(d.InvoiceDate).month,
    # invoice_year = lambda d: pd.DatetimeIndex(d.InvoiceDate).year,
    # invoice_day_of_month = lambda d: pd.DatetimeIndex(d.InvoiceDate).day,
    # n_words_description = lambda d: [len(str(x).split()) for x in d.Description],
    #noun=lambda d: [str(x).split()[-1].strip() for x in d.Description],
    # material=lambda d: d.Description.str.extract(r'(ceramic|metal|wood|plastic|cloth|felt|fabric|silk|polyester|linen|cotton|compostable|mahogany|canvas|cashmere|chiffon|denim|viscose|wool|fur|lace|leather|diamond|crystal|rhinestone|jewel|birch|rubber|wax|vintage|organic|pashmina|satin|spandex|suede|cement|marble)', expand=False, flags=re.IGNORECASE),
    # color=lambda d: extract_color(d.Description).apply(lambda s: s.split(' ')[0]),
#     ).apply(lambda y: y.str.lower().str.strip().replace('', np.nan) if str(y.dtype)=='object' else y)
# print('df_new.shape # (531283, 11)', df_new.shape) # (531283, 11))

new_list = []
new_list_country_codes = []
for countrystr in df_new.Country.str.title().astype(str).replace({' \\(.*':'', 'St\\.':'Saint'}, regex=True):
    try:
        new_boob_country_code = pc.country_name_to_country_alpha2(countrystr, cn_name_format="default")
        new_boob = pc.country_alpha2_to_continent_code(new_boob_country_code)
    except:
        new_boob = ''
        countrystr = re.sub('.*Nevis.*', 'Cayman Islands', countrystr)
        new_boob_country_code = pc.country_name_to_country_alpha2(countrystr, cn_name_format="default")
        new_boob = pc.country_alpha2_to_continent_code(new_boob_country_code)
    new_list.append(new_boob)
    new_list_country_codes.append(new_boob_country_code)
df_new['continent'] = new_list
df_new['country_code'] = new_list_country_codes
df_new


df_new[df_new.continent.isnull()|df_new.continent==""]
df_new[df_new.continent==""]
df_new.continent.value_counts()
df_new.Name.replace(' .*', '', regex=True).value_counts()
df_new.Country.value_counts()

continent_dummies = pd.get_dummies(df_new.continent, prefix='Continent')
df_new = pd.concat([df_new, continent_dummies], axis=1)

country_dummies = pd.get_dummies(df_new.country_code, prefix='Country')
df_new = pd.concat([df_new, country_dummies[['Country_US', 'Country_CN']]], axis=1)
df_new.continent = df_new.continent.replace('NA', 'NAm', regex=False)

df_new.to_csv("data/forbes-billionaires-of-2021-20/forbes_billionaires_new.csv", index=False)

# df_new = pd.read_csv("https://raw.githubusercontent.com/srhoads/learn/main/data/forbes-billionaires-of-2021-20/forbes_billionaires.csv")
df_new.Age.describe()
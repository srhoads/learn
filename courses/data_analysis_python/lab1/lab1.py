# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Lab #1
# Due No Due Date Points 100 Submitting a website url or a file upload
# Dear all --
# 
# Answer the following questions:
# 
# 1-- Choose one variable, look at its distribution, and plot a histogram of it.  Explain what you take away from looking at the variable.
# 
# 2-- Choose some continuous-ish variable, and calculate its mean and standard deviation by some grouping variable.  Explain what conclusion you draw from this analysis.
# 
# 3-- Choose two categorical-ish variables, and cross-tabulate them.  Explain what conclusion you draw from this analysis.
# 

# %%
USE_GSS = False
from __future__ import division
import os, re
import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/srhoads/srhoads/master/python/functions.py').read())
pkg('pandas'); import pandas as pd
pkg('numpy'); import numpy as np
pkg('statsmodels'); import statsmodels.api as sm
import statsmodels.formula.api as smf
pkg('colour'); from colour import Color
import matplotlib.pyplot as plt

def printurn(x):
    print(x)
    return(x)

def recode_onehot_if(s='CHILDRENS COAT', pattern='CHILDREN'):
    result = '1' if pattern in str(s) else '0'
    return(result)

def check_color(color):
    try:
        Color(color)
        return True
    except ValueError:
        return False

def extract_color(pdcolumn):
    pdcolumn = pd.Series(pdcolumn.copy()) if type(pdcolumn)==str else pd.Series(pdcolumn) if type(pdcolumn)==list else pdcolumn
    newcolumn = pdcolumn.copy().apply(lambda s: ' '.join([i for i in str(s).split(' ') if check_color(i)]))
    return(newcolumn)

def compute_percentage(x, my_crosstab):
      pct = float(x/my_crosstab['count'].sum()) * 100
      return round(pct, 2)

# extract_color(d.head().Description)
    # _color = [i for i in s.split(' ') if check_color(i)]
# s = 'light green'
# _color = [i for i in s.split(' ') if check_color(i)]

# %% [markdown]
# ### Reading data below! We're using our own custom dataset (not the GSS data!). I found this sample retail sales dataset at this link: #https://www.kaggle.com/carrie1/ecommerce-data. I'm using this dataset because it aligns with my personal research interests and I wanted to make this lab as utilitarian as possible!

# %%
# # AREND CUSTOM (bc I personally hate downloading data files to my disk...):
if USE_GSS: # I'm not using GSS data for this lab, but if we wanted to, then we could grab the data straight from a URL (demo below).
    # try:
    #     import gssapi
    # except:
    #     os.system('pip install gssapi'); import gssapi
    from io import BytesIO
    from zipfile import ZipFile
    from urllib.request import urlopen # or: requests.get(url).content
    resp = urlopen('http://gss.norc.org/documents/stata/2006_stata.zip')
    zipfile = ZipFile(BytesIO(resp.read()))
    dtafile = zipfile.namelist()[0]
    print(dtafile) # ['GSS2006.dta']
    d = pd.read_stata(zipfile.open(dtafile), convert_categoricals=False)
else:
    try: # NOTE: trying to read the csv file with basic default args first, but if error, we're adding the encoding argument. Since this dataset has text data, the encoding has some quirks and some Python & Pandas versions don't support it with default UTF-8 encoding.
        d = pd.read_csv('data.csv') # dataset is from #https://www.kaggle.com/carrie1/ecommerce-data
    except:
        d = pd.read_csv('data.csv', encoding="ISO-8859-1") #https://www.kaggle.com/carrie1/ecommerce-data
d

# %% [markdown]
# ## I'm recoding some variables/creating new ones below! I'm also filtering the data so that it only has the observations we care about (valid UnitPrices and valid Quantity sold).

# %%
## df.loc[df.ID == 103, ['FirstName', 'LastName']] = 'Matt', 'Jones'
d_ = d.copy().apply(lambda y: y.str.lower() if str(y.dtype)=='object' else y).assign(
    children=lambda d: [recode_onehot_if(x, 'children') for x in d.Description],
    noun=lambda d: [str(x).split()[-1].strip() for x in d.Description],
    material=lambda d: d.Description.str.extract(r'(ceramic|metal|wood|plastic|cloth|felt|fabric|silk|polyester|linen|cotton|compostable|mahogany|canvas|cashmere|chiffon|denim|viscose|wool|fur|lace|leather|diamond|crystal|rhinestone|jewel|birch|rubber|wax|vintage|organic|pashmina|satin|spandex|suede|cement|marble)', expand=False, flags=re.IGNORECASE),
    # color=lambda d: d.Description.str.extract('\\b(pink|red|blue|white|green|yellow|purple|orange)\\b', expand=False, flags=re.IGNORECASE),
    color=lambda d: extract_color(d.Description),
    ).query('Quantity>=0 and UnitPrice>=0')
## d.assign(noun=lambda d: [str(x).split()[-1] for x in d.Description])
d_

# %% [markdown]
# ### Visualizing value counts & other descriptive statistics for the factors of our variable of interest (material)

# %%
# Checking value counts (frequencies) of words in the description field
pd.Series(' '.join(d_.Description.astype(str).unique().tolist()).split()).value_counts().head(20)


# %%
# print(d_.noun.value_counts().head(20))
print(d_.value_counts().head(20))
# sorted(d_.noun.unique().tolist())


# %%
d_.material.value_counts()


# %%
summary = d_.material.describe()
summary.transpose()


# %%
## Crosstabulation (like in Eirich's example!)
my_crosstab = pd.crosstab(index=d_["material"], columns="count")     
my_crosstab['percentage'] = my_crosstab.apply(lambda y: compute_percentage(y, my_crosstab), axis=1)
my_crosstab.sort_values('percentage', ascending=False)

# %% [markdown]
# Below is a histogram of material! It's just demoing the value counts (which I explain below).

# %%
d_['material'].value_counts().plot(kind='bar')

# %% [markdown]
# # Answer to Q 1:
# 
# <h3>
# 1-- The variable upon which I'm focusing is one that I custom-created from a text variable. Using the "Description" variable of the retail item in my sales dataset, I found that many of the described items include their materials (ie: metal, ceramic, wood, etc...). I thought this variable would be interesting because different materials of retail items might have different sales patterns (price, quantity, or even date/location purchased). 
# 
# I'm most interested in fashion-related retail items, so I defined a list of materials that I wanted to extract from the description column. I also included the code "vintage" even though it's not a specific material because it connotes a type of quality. This is a very rough first-go at defining a material list, but it's useful for now. The value counts are listed in the cell above. The most common codes are vintage, wood, metal, felt, and ceramic. Less common materials listed in item descriptions are diamond, fur, suede, wax.
# 
# As shown in the crosstabulation table above, 30.05% of the records have the "vintage" code. This isn't really a material, so I think I might recode it as a separate variable. Vintage must be a popular description in retail sales though!
# The other factors with substantial percentages of the total are "wood" at 23.02%, "metal" at 18.97%, "felt" at 8.74%,"ceramic" at 8.17%, "lace" at 3.39%, "jewel" at 1.72%, and "crystal" at 1.17% of the observations. The other material factors make up under 1%. I could be more specific with some of these codes. For example, I could look for text including specific types of metal instead of "metal" at large.
#     
# </h3>
# %% [markdown]
# ### In the cell below, I am removing the "vintage" code from the "material" variable.

# %%
d_['material'] = d_['material'].replace({'vintage': np.NaN}, regex=False)
d_['material'].value_counts()


# %%


# %% [markdown]
# # Answer to Q2:
# *Question: 2-- Choose some continuous-ish variable, and calculate its mean and standard deviation by some grouping variable.  Explain what conclusion you draw from this analysis.*
# 
# ### For the histograms, I cropped the min and max X values to be more intuitive to the range of possibilities in the data itself. Quantity of item purchased can't ever be negative, so I restricted the X min to a quantity of 0.

# %%



# %%
print(d_.Quantity.describe())
d_.hist(column='Quantity', range=(0, 100), bins=20)


# %%
print(d_.UnitPrice.describe())
d_.hist(column='UnitPrice', range=(0, 30))


# %%
print('Mean of material x UnitPrice:\n\n', d_.groupby(['material'])['UnitPrice'].mean().sort_values(ascending=False), '\n\n')
print('Median of material x UnitPrice:\n\n', d_.groupby(['material'])['UnitPrice'].median().sort_values(ascending=False), '\n\n')
print('Standard Deviation of material x UnitPrice:\n\n', d_.groupby(['material'])['UnitPrice'].std().sort_values(ascending=False), '\n\n')
print('Count of material x UnitPrice (meaning unique unit prices per each material):\n\n', d_.groupby(['material'])['UnitPrice'].count().sort_values(ascending=False), '\n\n')


# %%
print('Mean of material x Quantity:\n\n', d_.groupby(['material'])['Quantity'].mean(), '\n\n')
print('Median of material x Quantity:\n\n', d_.groupby(['material'])['Quantity'].median(), '\n\n')
print('Standard Deviation of material x Quantity:\n\n', d_.groupby(['material'])['Quantity'].std(), '\n\n')
print('Count of material x Quantity (meaning unique quantity levels purchased per each material):\n\n', d_.groupby(['material'])['Quantity'].count(), '\n\n')

# %% [markdown]
# ### Let's try grouping these variables by material! Material could be an interesting metric alongside price and quantity of items.

# %%
## Crosstabulation (like in Eirich's example!)
# pd.crosstab(d_.material, [d_.UnitPrice, d.Quantity], rownames=['material'], colnames=['UnitPrice', 'Quantity'])
pd.crosstab(d_.material, [d_.UnitPrice], rownames=['material'], colnames=['UnitPrice'])
# my_crosstab['percentage'] = my_crosstab.apply(lambda y: compute_percentage(y, my_crosstab), axis=1)
# my_crosstab.sort_values('percentage', ascending=False)


# %%



# %%
pd.crosstab(d_.material, [d_.Quantity], rownames=['material'], colnames=['Quantity'])


# %%
my_crosstab

# %% [markdown]
# <h2>
# Analysis
# </h2>
# 
# <h4>
# Based on the descriptive statistics above, when analysing the mean price grouped by the categorical variable material, we observe that fur on average is the most expensive item purchased. On the contrary, plastic is the cheapest item. 
# 
# When analysing the standard deviation of the categorical variable material grouped by its unit price, denim is the most consistent material purchased at a similar price, whilst diamond is the most erratic material purchased when a consumer purchases an item from the company in terms of its unit pricing. There could be several reasons for this: denim is perhaps the most common, and most casual material used when an individual dresses or wants to go out. As a consequence, since it is such an ubiquitous material and item used amongst consumers, there is more supply from vendors which standardises the pricing of items with denim material. Conversely, with diamond, a luxury material exacerbated by a scattered industry where prices are not simply quoted, it is not surprising to data analysts to observe these items be inconsistently priced. 
# 
# There are confounds here because some of these items do not have the construct validity that is rightfully supposed to have. For example, cloth encompasses a wide range of material, possibly entailing materials listed in the same data set i.e. cotton, lace etc. They are all cloth material but they are treated as mutually exclusive entities.  We also aren’t accounting for variations in quality. Diamond, for example, has different grades of material, whereas denim is a lot more uniform. Our data set printed does not account for these different grades of the same material. In some ways, we are comparing apples and oranges, and if we have to report on these numbers, we should try to compile statistics that have more construct validity, and fewer issues with collinearity. 
#  </h4>

# %%


# %% [markdown]
# # Answer to Q3:
# *3-- Choose two categorical-ish variables, and cross-tabulate them.  Explain what conclusion you draw from this analysis.
# *
# 
# ### material by country

# %%
pd.crosstab(d_.material, [d_.Country], rownames=['material'], colnames=['Country'])

# %% [markdown]
# Based on the descriptive statistics above (cross-tabulating two categorical variables, country and material), there are some obvious conclusions that we can draw. First, since this is a UK retailer, it is not uncommon to see the most purchases(regardless of material), come from local consumers. Swiss consumers purchased the most amount of wood in comparison to other countries like Denmark, Saudi Arabia. We could think that it is logical for there to be a higher demand of wood in Swiss consumers than Saudi Arabian consumers because the climate is not optimal using wood for building, since wood is in prime condition in humid areas and at average temperatures. The dry climate in Saudia Arabia would not suit wood for building purposes. Based on the data set above, a conclusion that we could draw is that we require a narrowed down data set of countries so as to really analyze the material purchased based off geographical region. A way to do this could be by recoding the country variable as a new variable (continent for example).
# 

# %%
#os.system("pip install pycountry-convert")
import pycountry_convert as pc
country_code = pc.country_name_to_country_alpha2("United Kingdom", cn_name_format="default")
print(country_code)


# %%
continent_name = pc.country_alpha2_to_continent_code(country_code)
print(continent_name)


# %%
# d_.Country

new_list = []
for boob in d_.Country.str.title():
    try:
        new_boob = pc.country_name_to_country_alpha2(boob, cn_name_format="default")
        new_boob = pc.country_alpha2_to_continent_code(new_boob)
#         print(new_boob)
    except:
        new_boob = ''
    new_list.append(new_boob)


# %%
d_['continent'] = new_list


# %%
## For the sake of continent-code clarification, here's which codes are associated with which continents:
# Code	Continent name
# AF	Africa
# AN	Antarctica
# AS	Asia
# EU	Europe
# NA	North america
# OC	Oceania
# SA	South america

# And here are our new continent value counts!
d_.continent.value_counts()


# %%
pd.crosstab(d_.material, [d_.continent], rownames=['material'], colnames=['continent'])

# %% [markdown]
# ### Based on the above data set grouping these materials by continent, we have a much clearer picture. For starters, wood, like we've already mentioned in the first analysis, is most heavily purchased in Europe in comparison to Asia and South America. And this is rightfully so with the more suitable climate conditions in Europe for wood. There are other interesting observations such as diamond being more heavily purchased in Europe than in Asia. This could be because most luxury fashion brands are stationed in Europe, like Pandora for example, who have built a reputation as a reliable diamond maker for jewellery.  

# %%



# %%


# %% [markdown]
# # IGNORE THE FOLLOWING STUFF.....

# %%
try:
    from census import Census
    from us import states
except:
        os.system('pip install census, us, CensusData')
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
c = Census(CENSUS_API_KEY)
# c.acs5.get(('NAME', 'B25034_010E'),{'for': 'state:{}'.format(states.MD.fips)})
# c.acs5.state(('NAME', 'B25034_010E'), states.MD.fips)
# import requests
# s = requests.session()
# s.headers.update({'User-Agent': 'census-demo/0.0'})

# c = Census(CENSUS_API_KEY, session=s)
# c.acs5.state('B01001_004E', Census.ALL)
os.environ
c.sf1.get('NAME', geo={'for': 'tract:*', 'in': 'state:{} county:170'.format(states.AK.fips)})


# %%
# print(d.columns.tolist())
os.system('pip install kaggle')
'kaggle datasets download -d PromptCloudHQ/innerwear-data-from-victorias-secret-and-others'

import requests
url="https://www.kaggle.com/carrie1/ecommerce-data/download"
r = requests.get(url)
r.headers


# %%



# %%



# %%



# %%
# import requests, csv
# from contextlib import closing
# url = 'https://www.kaggle.com/carrie1/ecommerce-data/download'
# with closing(requests.get(url, stream=True)) as r:
#     reader = csv.reader(r.iter_lines(), delimiter=',', quotechar='"')
# #     for row in reader:
# #         print(row)
#     [print(row) for row in reader]
import csv
import requests
CSV_URL = 'https://www.kaggle.com/carrie1/ecommerce-data/download'
with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    for row in my_list:
        print(row)


# %%



# %%


# %% [markdown]
# ## Sample answer to Q 1:
#     
# 1-- I chose to look at a variable that investigates one of the Big Five dimensions of personality, as developed by psychologists. This variable, ​big5a1​, is the respondent's level of agreement to the following statement: I see myself as someone who is reserved; so it is basically a measure of introversion.
# 
# As Table 1 indicates, valid answers range from 1 to 5, where 1 represents strongagreement and 5 indicates strong disagreement (3=neither agreement or disagreement).
# 
# The below table, Table 2, shows that a large plurality of Americans (42%) agree that they are reserved, while only roughly a quarter of Americans assert they are not reserved (21% disagreeand only 7% strongly disagree). It is somewhat surprising how few Americans think of themselves as outgoing or bold, and yet few think of themselves as extremely shy (i.e., only 13% of people ​strongly​ agree they are reserved).
# 
# There is a histogram of this variable as well, which represents the same results of what I was talking about.
# 
# From the codebook, Table 1 is:
# 
# 1019) I see myself as someone who is reserved. (BIG5A1) TOTAL % 1) Strongly agree 200 13.2 2) Agree 632 41.6 3) Neither agree nor disagree 253 16.7 4) Disagree 317 20.9 5) Strongly disagree 102 6.7 8) Can't choose 12 0.8 9) No answer 2 0.1 Missing 2992 TOTAL 1518 100

# %%



# %%



# %%



# %%
# os.chdir('C:/Users/gme2101/Desktop/Data Analysis Data') # change working directory

# ## I am using the codebook at http://www.thearda.com/Archive/Files/Codebooks/GSS2006_CB.asp ##

# d = pd.read_csv("GSS.2006.csv")
# d.head()


# %%



# %%




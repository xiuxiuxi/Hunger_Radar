# Hungry Radar #
CSE 6242 DVA Spring 2021  
Team DNA Reformation: Xiyuan Dong, Qingrong Lu,Ou Xi, Lixin Jin, Dahai Fu, Binny Tsai
![Screenshot](https://github.gatech.edu/ljin77/CSE6242Sp21TeamDNA/blob/master/Screenshot.PNG)

# Description #
The Local Consumer Review Survey shows 87% of customers read reviews on online review sites such as Yelp and Zomato and 72% of people trust online reviews as much as personal recommendations . Also, the study shows a better understanding of customers' opinions and sentiments would recommend better business strategies . There is a need for an efficient way for customers to explore restaurants and for restaurant business owners to get insight into their target customers’ consumer behavior. Hungry Radar will provide an interactive map for users to choose cities and restaurant categories to check various business statistics and find the recommended restaurants in the Boston Metropolitan Area.
 
# Datasets #
### 1. Yelp Open Dataset ###
> https://www.yelp.com/dataset

10,550 restaurants with 1,365,551 reviews in the Boston Metropolitan Area are filtered from 7GB business and review raw data.
Each restaurant has basic geolocation and category information.
Each review has review text, reviewer id, business id, date of the review, and the stars given by the reviewer.

### 2. Yelp Filter Review Dataset (Labeled Yelp Fake Review Dataset): ###
We thank for Prof Bing Liu for authorizing us to use this dataset for our Fake Review Detection machine learning classifier training:
> https://gtvault-my.sharepoint.com/:u:/g/personal/xdong77_gatech_edu/ETimOj7oWyNDkM9mukuarCsBJPjMwXwyhQca7X5iLAbF7g?e=5sfPdH

Provided by the authors of What Yelp Fake Review Filter Might Be Doing.
8,303 fake reviews from a total of 67,019 reviews
Has the same data structure as review data in Yelp Open Dataset.

# Installation #
The recommended method is to install and run the application inside the a virtual environments.

### Dependencies ###
Install Python
Please make sure you have download and install the _Python 3.x_
> https://www.python.org/downloads/

### Setup Virtual Environment ###
At the project root directory:

Create vitual Environment
#### macOS/Linux ####
> python3 -m venv .venv  
> You may need to run sudo apt-get install python3-venv first

#### Windows ####
> python -m venv .venv  
> You can also use py -3 -m venv .venv

A virtual environment will be created and started.  

In case of if the virtual environment does not start automatically, please run the following scripts to activate virtual environment.

Unix Bash (Linux, Mac, etc.):

> source .\venv\Scripts\activate.bat 

Windows CMD:

> .\venv\Scripts\activate.bat  

Windows PowerShell:

> Set-ExecutionPolicy -ExecutionPolicy Unrestricted;  
> .\venv\Scripts\activate.ps1


__Install Required Packages__
> pip install -r .\requirements.txt  

# Start Application #

Unix Bash (Linux, Mac, etc.):

> export FLASK_APP=hungryradar;
> export FLASK_ENV=development;
> flask run

Windows CMD:

> set FLASK_APP=hungryradar;
> set FLASK_ENV=development;
> flask run

Windows PowerShell:

> Set-ExecutionPolicy -ExecutionPolicy Unrestricted;  
> $env:FLASK_APP = "hungryradar";
> $env:FLASK_ENV = "development";
> flask run



# Data Cleaning & Analysis #
The data behind our front-end web have been included in this project. If you're interested in how they are generated, please refer to the following steps

Please download the following datasets and store under this directory: __./data/__
> https://www.yelp.com/dataset  

> https://gtvault-my.sharepoint.com/:u:/g/personal/xdong77_gatech_edu/ETimOj7oWyNDkM9mukuarCsBJPjMwXwyhQca7X5iLAbF7g?e=5sfPdH

When downloading the Yelp Dataset, only the compressed JSON file is required. Please note that to correctly extract the Yelp dataset, you may need to extract twice. Thus, please rename the extracted file to end with ".tar" and extract the file again.

After successfully extracted the datasets, please check if the following files under __./data/__ before running the scripts.
> yelp_academic_dataset_business.json  
> yelp_academic_dataset_review.json  
> labeled_review.csv

1. Business Data Cleaning: data/DataCleaning_business.py
> python data/DataCleaning_business.py
2. Review Data Cleaning & Sentiment Analysis & Fake Rreview Detection: data/review.py
> python data/review.py
3. Fake Review Counts Grouping: data/group_fake_reviews.py
> python data/group_fake_reviews.py

After finishing running the scripts above, please copy all the .csv files under __./data/result/__ to __./static/csv/__. Then, restart the application and proform a hard refresh "__CTRL+F5__" on the web browser. 

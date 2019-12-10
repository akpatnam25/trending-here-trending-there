## imports
import sys
import pandas as pd
import numpy as np
import random
import requests
import json
import matplotlib.pyplot as plt
import time
from nltk.book import *
from nltk.corpus import stopwords
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import seaborn as sns
import pyLDAvis
import pyLDAvis.gensim
sns.set_context('notebook')
import warnings
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from nltk.corpus import stopwords
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
import random
from gensim import corpora
import gensim
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
spacy.load('en')
from spacy.lang.en import English
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import bokeh.plotting as bp
from bokeh.models import HoverTool, ColorBar, NumeralTickFormatter, LinearColorMapper, LassoSelectTool, ResetTool, PanTool, BoxSelectTool, TapTool, PolySelectTool
from bokeh.palettes import plasma
from bokeh.plotting import figure
from bokeh.transform import transform
import bokeh.io
import bokeh.plotting as bpl
import bokeh.models as bmo
from bokeh.palettes import d3
from bokeh.transform import linear_cmap
from bokeh.models import SingleIntervalTicker, LinearAxis
from bokeh.layouts import gridplot
from nltk.stem.wordnet import WordNetLemmatizer
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.palettes import Category20
from bokeh.models import Legend
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, CDSView, GroupFilter
import bokeh
from bokeh.palettes import Category20c, Plasma
from bokeh.models import HoverTool
from bokeh.io import show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import HoverTool, ColorBar, NumeralTickFormatter, LinearColorMapper, LassoSelectTool, ResetTool, PanTool, BoxSelectTool, TapTool, PolySelectTool
from bokeh.io import output_file
from bokeh.models.widgets import Tabs, Panel


warnings.filterwarnings('ignore')

## written by Jeremy Tan
## inserts category field into each dataframe from given json file
def insert_category_field(df):
    # add a category column using the category id from the json file
    id_to_category = {}
    with open("data/US_category_id.json", 'r') as f: # the other category json files have missing category ids. We decided to just use the US file since it contained all of them and since they are standard internationally.
        data = json.load(f)
        for category in data['items']:
            id_to_category[category['id']] = category['snippet']['title']
        categories = []
        for id in list(df['category_id']):
            categories.append(id_to_category[str(id)])
        df.insert(4, 'category', categories)

### do not run this unless you have about 1 hour and 10 Youtube API keys and a lot of time!!! Use above already generated datasets for testing.

## for each videoId, find a related video
## written by Aravind Patnam
def do_search_youtube_request(videoId):
    f = open("apiKey", "r")
    key = f.read()
    url = "https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=50&order=relevance&relatedToVideoId={}&type=video&videoDefinition=any&key={}".format(videoId, key)
    r = requests.get(url)
    return r

## given a set of videoIds, find insights (statistics, tags, etc)
## written by Aravind Patnam
def find_video_insights(videoIds):
    f = open("apiKey", "r")
    key = f.read()
    print(videoIds)
    url = 'https://www.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id={}&key={}'.format(videoIds, key)
    r = requests.get(url)
    return r

## call this with 1 country at a time
## written by Aravind Patnam
def process_youtube_requests(videoIds):
    f = open("apiKey", "r")
    key = f.read()
    df = pd.DataFrame(columns=['video_id', 'title', 'channel_title', 'category_id', 'publish_time', 'tags',
                               'views', 'likes', 'dislikes', 'comment_count', 'description'])
    relatedVideoIds = []
    for videoId in videoIds:
        try:
            response = do_search_youtube_request(videoId)
            time.sleep(2)
            if (response.status_code == 200):
                r1 = response.json()
                relatedVideoIdItems = r1['items']
                for id in relatedVideoIdItems:
                    relatedVideoId = id['id']['videoId']
                    relatedVideoIds.append(str(relatedVideoId))
            else:
                print(response.status_code)
        except:
            print ("Something went wrong here! 2")
    random.shuffle(relatedVideoIds)
    videoIdsForInsights = []
    for i in range(0, len(relatedVideoIds), 50):
        videoIdsForInsights.append(relatedVideoIds[i:i + n])
    for videoIdList in videoIdsForInsights:
        videoIdsStr = '%2C'.join([str(elem) for elem in videoIdList])
        r2 = find_video_insights(videoIdsStr)
        time.sleep(2)
        if (r2.status_code == 200):
            r = r2.json()
            i = 0
            while (i < len(videoIdList)):
                try:
                    id = videoIdList[i]
                    title = (r['items'][i]['snippet']['title'])
                    channel_title = (r['items'][i]['snippet']['channelTitle'])
                    category_id = (r['items'][i]['snippet']['categoryId'])
                    publish_time = (r['items'][i]['snippet']['publishedAt'])
                    tags = '|'.join((r['items'][i]['snippet']['tags']))
                    views = (r['items'][i]['statistics']['viewCount'])
                    likes = (r['items'][i]['statistics']['likeCount'])
                    dislikes = (r['items'][i]['statistics']['dislikeCount'])
                    comment_count = (r['items'][i]['statistics']['commentCount'])
                    description = (r['items'][i]['snippet']['description'])
                    data = {'video_id': id, 'title': title, 'channel_title': channel_title, 'category_id' : category_id,
                           'publish_time' : publish_time, 'tags' : tags, 'views' : views, 'likes' : likes, 'dislikes' : dislikes,
                           'comment_count' : comment_count, 'description' : description}
                    df = df.append(data, ignore_index = True)
                except:
                    print("Something went wrong! 3")
                i = i + 1
        else:
            print("Something went wrong! 4")
            print (r2.status_code)
            print(r2.text)
    return df


## method to find stats about a given country df
## written by Aravind Patnam
def find_stats(df):
    views = df['views']
    likes = df['likes']
    dislikes = df['dislikes']
    comment_count = df['comment_count']
    tags = list(df['tags'])
    tagsList = []
    for t in tags:
        tagsList.append(len(t.split('|')))
    viewsStats = [views.sum(), views.sum() / len(views), views.min(), views.max()]
    likesStats = [likes.sum(), likes.sum() / len(likes), likes.min(), likes.max()]
    dislikesStats = [dislikes.sum(), dislikes.sum() / len(dislikes), dislikes.min(), dislikes.max()]
    commentsStats = [comment_count.sum(), comment_count.sum() / len(comment_count), comment_count.min(), comment_count.max()]
    tagsStats = [sum(tagsList), sum(tagsList) / len(tagsList), min(tagsList), max(tagsList)]
    statsDf = pd.DataFrame({'views': viewsStats, 'likes': likesStats, 'dislikes': dislikesStats, 'comment_count': commentsStats, 'tags': tagsStats}, index = ['count', 'mean', 'min', 'max'])
    return statsDf

## written by Aravind Patnam

###get most common tags from a country df
## written by Aravind Patnam
def get_most_common_tags(country_df):
    tags = country_df['tags'].to_string(index=False, header=False)
    split_tags = [i.replace('"', '') for i in tags.split("|")]
    stop_words = stopwords.words('english')
    filtered_tags = [word for word in split_tags if word not in stop_words]
    fdist = FreqDist(split_tags)
    most_popular_tags = fdist.most_common(1000)
    return dict(most_popular_tags)


## written by Aravind Patnam
def extract_features(words):
    return dict([(word, True) for word in words.split()])

## written by Aravind Patnam
def build_sentiment_analysis_model():
    ## do sentiment analysis on each of the tags
    ## return classifications on each of the tags in both trending and non-trending per country
    positive_words_df = pd.read_fwf('data/positivewords.txt')
    negative_words_df = pd.read_fwf('data/negativewords.txt')
    positive_words = positive_words_df['positivewords'].values.tolist()
    negative_words = negative_words_df['negativewords'].values.tolist()
    pos_feats = [(extract_features(f), 'positive') for f in positive_words ]
    neg_feats = [(extract_features(f), 'negative') for f in negative_words ]
    dataset = pos_feats + neg_feats
    random.shuffle(dataset)
    cutoff = int(0.80 * len(dataset))
    train_data = dataset[:cutoff]
    test_data = dataset[cutoff:]

    classifier = NaiveBayesClassifier.train(train_data)
    print("Accuracy is:", classify.accuracy(classifier, test_data))
    # print(classifier.show_most_informative_features(10))
    return classifier

## written by Aravind Patnam
def execute_model(tags):
    classifications = {}
    classifier = build_sentiment_analysis_model()
    # classifier.show_most_informative_features(5)
    for tag in tags:
        classified = classifier.classify(extract_features(tag))
        classifications[tag] = classified
    return classifications

## written by Aravind Patnam

## pulls actual statistics from classifications for visualization
def get_sentiment_stats(classification, country):
    sentiments = list(classification.values())
    sentiments_df = pd.DataFrame(sentiments, columns=['Sentiment'])
    negatives = len(sentiments_df[sentiments_df['Sentiment'] =='negative'])
    positives = len(sentiments_df[sentiments_df['Sentiment'] =='positive'])
    total_len = len(sentiments_df)
    percentage_of_negative = negatives / total_len * 100
    percentage_of_positive = positives / total_len * 100
    ratio = positives/negatives
    # ratioStr = "{} positive/negative ratio: {} -----> {}% positives of total , {}% negatives of total".format(country, ratio, percentage_of_positive, percentage_of_negative)
    # print (ratioStr)
    return country, positives, negatives

## written by Aravind Patnam
## lda model prep for topic model inference from youtube tags and description accross trending and nontrending videos
def tokenize(text):
    parser = English()
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


## written by Aravind Patnam
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


## written by Aravind Patnam
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))


def prepare_text_for_lda(text):
    tokens = tokenize(str(text))
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


## written by Aravind Patnam

## method to execute LDA model and clean the LDA input one more time
def do_LDA(lda_input):
    text_data = []
    for line in lda_input:
        tokens = prepare_text_for_lda(line)
        if random.random() > .99:
            text_data.append(tokens)
    topics, corpus, dictionary = execute_LDA(text_data)
    return topics, corpus, dictionary


def execute_LDA(text_data):
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('data/corpus.pkl', 'wb'))
    dictionary.save('data/dictionary.gensim')
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
    ldamodel.save('data/model5.gensim')
    topics = ldamodel.print_topics(num_words=10)
    return topics, corpus, dictionary


def clean_lda_input(input):
    l = []
    for a in input:
        text = re.sub(r"http\S+", "", str(a))
        l.append(text)
    return l


## written by Aravind Patnam

## visualize LDA using pyLDAvis -> this might be only visible on nbviewer depending on your notebook viewing settings

def visualize_LDA(start, corpus, dictionary):
    if (start == True):
        lda = gensim.models.ldamodel.LdaModel.load('data/model5.gensim')
        lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
        return lda_display

## written by Aravind Patnam
## edited by Timothy Le
## edit 1: Rounded the mean value of each category to remove power of 10 notation
def do_describe_analysis(describeMap, countries, describesKeys):
    views_count = []
    views_mean = []
    views_max = []
    views_min = []
    likes_count = []
    likes_mean = []
    likes_max = []
    likes_min = []
    dislikes_count = []
    dislikes_mean = []
    dislikes_max = []
    dislikes_min = []

    comments_mean = []
    comments_max = []
    comments_min = []

    tags_mean = []
    tags_max = []
    tags_min = []

    for a in countries:
        views_count.append(describeMap[a].loc['count', 'views'])
        views_mean.append(round(describeMap[a].loc['mean', 'views']))
        views_max.append(describeMap[a].loc['max', 'views'])
        views_min.append(describeMap[a].loc['min', 'views'])

        likes_count.append(describeMap[a].loc['count', 'likes'])
        likes_mean.append(round(describeMap[a].loc['mean', 'likes']))
        likes_max.append(describeMap[a].loc['max', 'likes'])
        likes_min.append(describeMap[a].loc['min', 'likes'])

        dislikes_count.append(describeMap[a].loc['count', 'dislikes'])
        dislikes_mean.append(round(describeMap[a].loc['mean', 'dislikes']))
        dislikes_max.append(describeMap[a].loc['max', 'dislikes'])
        dislikes_min.append(describeMap[a].loc['min', 'dislikes'])

        comments_mean.append(round(describeMap[a].loc['mean', 'comment_count']))
        comments_max.append(describeMap[a].loc['max', 'comment_count'])
        comments_min.append(describeMap[a].loc['min', 'comment_count'])

        tags_mean.append(round(describeMap[a].loc['mean', 'tags']))
        tags_max.append(describeMap[a].loc['max', 'tags'])
        tags_min.append(describeMap[a].loc['min', 'tags'])
    # visualize above numeric data
    bokeh.io.reset_output()
    bokeh.io.output_notebook()

    countries = describesKeys
    categories = ['views_mean', 'views_max', 'views_min']
    categoriesLikes = ['likes_mean', 'likes_max', 'likes_min']
    categoriesDislikes = ['dislikes_mean', 'dislikes_max', 'dislikes_min']
    categoriesComments = ['comments_mean', 'comments_max', 'comments_min']
    categoriesTags = ['tags_mean', 'tags_max', 'tags_min']

    data = {'countries': countries,
            'views_mean': views_mean,
            'views_max': views_max,
            'views_min': views_min}
    dataLikes = {'countries': countries, 'likes_mean': likes_mean, 'likes_max': likes_max, 'likes_min': likes_min}

    dataDislikes = {'countries': countries, 'dislikes_mean': dislikes_mean, 'dislikes_max': dislikes_max,
                    'dislikes_min': dislikes_min}

    dataComments = {'countries': countries, 'comments_mean': comments_mean, 'comments_max': comments_max,
                    'comments_min': comments_min}

    dataTags = {'countries': countries, 'tags_mean': tags_mean, 'tags_max': tags_max, 'tags_min': tags_min}

    # this creates [ ("Apples", "2015"), ("Apples", "2016"), ("Apples", "2017"), ("Pears", "2015), ... ]
    x = [(c, category) for c in countries for category in categories]
    # print (len(x))
    yLikes = [(c, l) for c in countries for l in categoriesLikes]
    yDislikes = [(c, h) for c in countries for h in categoriesDislikes]

    yComments = [(c, comms) for c in countries for comms in categoriesComments]

    yTags = [(c, ts) for c in countries for ts in categoriesTags]

    counts = sum(zip(data['views_mean'], data['views_max'], data['views_min']), ())  # like an hstack
    countsLikes = sum(zip(dataLikes['likes_mean'], dataLikes['likes_max'], dataLikes['likes_min']),
                      ())  # like an hstack
    countsDislikes = sum(zip(dataDislikes['dislikes_mean'], dataDislikes['dislikes_max'], dataDislikes['dislikes_min']),
                         ())
    countsComments = sum(zip(dataComments['comments_mean'], dataComments['comments_max'], dataComments['comments_min']),
                         ())

    countsTags = sum(zip(dataTags['tags_mean'], dataTags['tags_max'], dataTags['tags_min']), ())

    source = ColumnDataSource(data=dict(x=x, counts=counts))
    sourceLikes = ColumnDataSource(data=dict(x=yLikes, counts=countsLikes))
    sourceDislikes = ColumnDataSource(data=dict(x=yDislikes, counts=countsDislikes))

    sourceComments = ColumnDataSource(data=dict(x=yComments, counts=countsComments))

    sourceTags = ColumnDataSource(data=dict(x=yTags, counts=countsTags))

    hover = HoverTool()

    hover.tooltips = [
        ("(x,y)", "($x, $y)"),
        ("Country", "@x"),
        ("Stat", "@counts")
    ]

    p = figure(x_range=FactorRange(*x), plot_height=250, plot_width=3000,
               title="Views Counts Per Country Trending/NonTrending",
               toolbar_location=None,
               tools=[BoxSelectTool(), LassoSelectTool(), ResetTool(), hover, TapTool(), PolySelectTool()])

    p.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",
           fill_color=factor_cmap('x', palette=Spectral6, factors=categories, start=1, end=2))

    p1 = figure(x_range=FactorRange(*yLikes), plot_height=250, plot_width=3000,
                title="Likes Counts Per Country Trending/NonTrending",
                toolbar_location=None,
                tools=[BoxSelectTool(), LassoSelectTool(), ResetTool(), hover, TapTool(), PolySelectTool()])

    p1.vbar(x='x', top='counts', width=0.9, source=sourceLikes, line_color="white",
            fill_color=factor_cmap('x', palette=Spectral6, factors=categoriesLikes, start=1, end=2))

    p2 = figure(x_range=FactorRange(*yDislikes), plot_height=250, plot_width=3000,
                title="Dislikes Counts Per Country Trending/NonTrending",
                toolbar_location=None,
                tools=[BoxSelectTool(), LassoSelectTool(), ResetTool(), hover, TapTool(), PolySelectTool()])

    p2.vbar(x='x', top='counts', width=0.9, source=sourceDislikes, line_color="white",
            fill_color=factor_cmap('x', palette=Spectral6, factors=categoriesDislikes, start=1, end=2))

    p3 = figure(x_range=FactorRange(*yComments), plot_height=250, plot_width=3000,
                title="Comment Counts Per Country Trending/NonTrending",
                toolbar_location=None,
                tools=[BoxSelectTool(), LassoSelectTool(), ResetTool(), hover, TapTool(), PolySelectTool()])

    p3.vbar(x='x', top='counts', width=0.9, source=sourceComments, line_color="white",
            fill_color=factor_cmap('x', palette=Spectral6, factors=categoriesComments, start=1, end=2))

    p4 = figure(x_range=FactorRange(*yTags), plot_height=250, plot_width=3000,
                title="Tags Counts Per Country Trending/NonTrending",
                toolbar_location=None,
                tools=[BoxSelectTool(), LassoSelectTool(), ResetTool(), hover, TapTool(), PolySelectTool()])

    p4.vbar(x='x', top='counts', width=0.9, source=sourceTags, line_color="white",
            fill_color=factor_cmap('x', palette=Spectral6, factors=categoriesTags, start=1, end=2))

    p.y_range.start = 0
    p.y_range.end = 10000
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None

    p1.y_range.start = 0
    p1.y_range.end = 10000
    p1.x_range.range_padding = 0.1
    p1.xaxis.major_label_orientation = 1
    p1.xgrid.grid_line_color = None

    p2.y_range.start = 0
    p2.y_range.end = 10000
    p2.x_range.range_padding = 0.1
    p2.xaxis.major_label_orientation = 1
    p2.xgrid.grid_line_color = None

    p3.y_range.start = 0
    p3.y_range.end = 10000
    p3.x_range.range_padding = 0.1
    p3.xaxis.major_label_orientation = 1
    p3.xgrid.grid_line_color = None

    p4.y_range.start = 0
    p4.y_range.end = 100
    p4.x_range.range_padding = 0.1
    p4.xaxis.major_label_orientation = 1
    p4.xgrid.grid_line_color = None

    # Bokeh Library
    from bokeh.io import output_file
    from bokeh.models.widgets import Tabs, Panel

    # Create two panels, one for each conference
    pPanel = Panel(child=p, title='Views Analysis')
    p1Panel = Panel(child=p1, title='Likes Analysis')
    p2Panel = Panel(child=p2, title='Dislikes Analysis')
    p3Panel = Panel(child=p3, title='Comments Analysis')
    p4Panel = Panel(child=p4, title='Tags Analysis')

    # Assign the panels to Tabs
    tabs = Tabs(tabs=[pPanel, p1Panel, p2Panel, p3Panel, p4Panel])

    # Show the tabbed layout
    show(tabs)
    return likes_count, dislikes_count

## written by Aravind Patnam
def do_likes_to_dislikes_analysis(likes_count, dislikes_count, countries):
    ## written by Aravind Patnam

    ## visualizes likes to dislikes ratio from above numeric data

    likesToDislikesRatio = [i / j for i, j in zip(likes_count, dislikes_count)]
    # Bokeh libraries

    bokeh.io.reset_output()
    bokeh.io.output_notebook()
    colorlist = Category20c[13] + Plasma[10]

    hover = HoverTool()

    hover.tooltips = [
        ("Country", "@countries"),
        ("Stat", "@likesToDislikesRatio")
    ]

    source = ColumnDataSource(data=dict(countries=countries, likesToDislikesRatio=likesToDislikesRatio))
    pRatio = figure(x_range=countries, plot_height=250, plot_width=2500, toolbar_location=None,
                    title="Likes To Dislikes", tools=[hover])
    pRatio.vbar(x='countries', top='likesToDislikesRatio', width=0.9, source=source,
                line_color='white', fill_color=factor_cmap('countries', palette=colorlist, factors=countries))

    pRatio.xgrid.grid_line_color = None
    pRatio.y_range.start = 0
    pRatio.y_range.end = 50
    pRatio.legend.orientation = "horizontal"
    pRatio.legend.location = "top_center"

    show(pRatio)

## written by Aravind Patnam
def do_wordcloud(us_trending_most_common_tags, gb_trending_most_common_tags, in_trending_most_common_tags, us_nontrending_most_common_tags, gb_nontrending_most_common_tags, in_nontrending_most_common_tags):
    ## written by Aravind Patnam

    ## wordcloud visualization on countries with most top tags

    links = ['https://cdn.pixabay.com/photo/2017/03/14/21/00/american-flag-2144392_960_720.png',
             'https://i.pinimg.com/originals/93/85/dd/9385dde0f8cf96e0c60e5e659036b303.jpg',
             'https://cdn.britannica.com/97/1597-004-7C2918C6/Flag-India.jpg']

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))
    mask_USA = np.array(Image.open(requests.get(links[0], stream=True).raw))
    mask_GB = np.array(Image.open(requests.get(links[1], stream=True).raw))
    mask_IN = np.array(Image.open(requests.get(links[2], stream=True).raw))
    image_colors_usa = ImageColorGenerator(mask_USA)
    image_colors_gb = ImageColorGenerator(mask_GB)
    image_colors_in = ImageColorGenerator(mask_IN)
    wordcloud_usa_trending = WordCloud(width=1000, height=1000, mask=mask_USA, background_color="white", max_words=1000,
                                       max_font_size=1000).generate(str(list(us_trending_most_common_tags.keys())))
    wordcloud_gb_trending = WordCloud(width=1000, height=1000, mask=mask_GB, background_color="white", max_words=1000,
                                      max_font_size=1000).generate(str(list(gb_trending_most_common_tags.keys())))
    wordcloud_in_trending = WordCloud(width=1000, height=1000, mask=mask_IN, background_color="white", max_words=1000,
                                      max_font_size=1000).generate(str(list(in_trending_most_common_tags.keys())))

    wordcloud_usa_nontrending = WordCloud(width=1000, height=1000, mask=mask_USA, background_color="white",
                                          max_words=1000, max_font_size=1000).generate(
        str(list(us_nontrending_most_common_tags.keys())))
    wordcloud_gb_nontrending = WordCloud(width=1000, height=1000, mask=mask_GB, background_color="white",
                                         max_words=1000, max_font_size=1000).generate(
        str(list(gb_nontrending_most_common_tags.keys())))
    wordcloud_in_nontrending = WordCloud(width=1000, height=1000, mask=mask_IN, background_color="white",
                                         max_words=1000, max_font_size=1000).generate(
        str(list(in_nontrending_most_common_tags.keys())))

    plt.figure(figsize=(20, 20), facecolor='k')

    axes[0, 0].imshow(wordcloud_usa_trending.recolor(color_func=image_colors_usa), interpolation="bilinear")
    axes[0, 0].set_title("USA Trending Tags")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(wordcloud_gb_trending.recolor(color_func=image_colors_gb), interpolation="bilinear")
    axes[1, 0].set_title("Great Britain Trending Tags")
    axes[1, 0].axis("off")

    axes[2, 0].imshow(wordcloud_in_trending.recolor(color_func=image_colors_in), interpolation="bilinear")
    axes[2, 0].set_title("India Trending Tags")
    axes[2, 0].axis("off")

    axes[0, 1].imshow(wordcloud_usa_nontrending.recolor(color_func=image_colors_usa), interpolation="bilinear")
    axes[0, 1].set_title("USA Non-Trending Tags")
    axes[0, 1].axis("off")

    axes[1, 1].imshow(wordcloud_gb_nontrending.recolor(color_func=image_colors_gb), interpolation="bilinear")
    axes[1, 1].set_title("Great Britain Non-Trending Tags")
    axes[1, 1].axis("off")

    axes[2, 1].imshow(wordcloud_in_nontrending.recolor(color_func=image_colors_in), interpolation="bilinear")
    axes[2, 1].set_title("India Non-Trending Tags")
    axes[2, 1].axis("off")

    plt.tight_layout(pad=0)
    plt.show()

## written by Aravind Patnam
def do_sentiment_analysis_visualization(data):
    ## written by Aravind Patnam

    ## visualizes sentiment analysis ratios

    ## import pandas as pd

    bokeh.io.reset_output()
    bokeh.io.output_notebook()
    mySource = bp.ColumnDataSource(data)
    # Use the field name of the column source
    mapper = linear_cmap(field_name='Positives', palette="Spectral6",
                         low=max(list(data['Positives']) + list(data['Negatives'])),
                         high=min(list(data['Positives']) + list(data['Negatives'])))

    hover = HoverTool()
    hover.tooltips = [
        ("Country", "@Country"),
        ("Ratio", "@PositiveNegativeRatio"),
        ("Positives", "@Positives"),
        ("Negatives", "@Negatives")
    ]
    myPlot = bp.figure(
        title='Positives vs. Negatives',
        plot_height=500,
        plot_width=500,
        tools=[BoxSelectTool(), LassoSelectTool(), ResetTool(), hover, TapTool(), PolySelectTool()],
        background_fill_color="black",
        x_axis_label='Positives', y_axis_label='Negatives',
        x_range=(0, 1000), y_range=(0, 1000))
    myPlot.title.text_font_size = '20pt'
    myPlot.xaxis.axis_label_text_font_size = "20pt"
    myPlot.yaxis.axis_label_text_font_size = "20pt"
    myPlot.xaxis.ticker = SingleIntervalTicker(interval=50)
    myPlot.yaxis.ticker = SingleIntervalTicker(interval=50)
    myPlot.circle("Positives",
                  "Negatives",
                  line_color=mapper, color=mapper, fill_alpha=1,
                  source=mySource,
                  size=12,
                  selection_color='deepskyblue',
                  nonselection_color='lightgray',
                  hover_fill_color='yellow', hover_alpha=2.0)
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0, 0))

    myPlot.add_layout(color_bar, 'right')

    bokeh.io.reset_output()
    bokeh.io.output_notebook()

    countries = list(data['Country'])
    hover = HoverTool(
        tooltips=[
            ("Ratio", "@top")
        ]
    )

    colors = ['red', 'red', 'red', 'red',
              'red', 'red', 'red', 'red',
              'red', 'red', 'green', 'green', 'green', 'green',
              'green', 'green', 'green', 'green',
              'green', 'green']

    p = figure(x_range=countries, plot_height=500, title="Sentiment Analysis Ratios", width=500,
               background_fill_color="black")
    legends = ['Trending', 'Not Trending']
    v = p.vbar(x=countries, top=list(data['PositiveNegativeRatio']), width=0.5, color=colors)
    v1 = p.vbar(x=countries, top=list(data['PositiveNegativeRatio']), width=0.5, color='green')
    p.add_layout(Legend(items=[
        ("Trending", [v]),
        ("NonTrending", [v1]),
    ]))
    v1.visible = False

    p.add_tools(hover)
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.major_label_orientation = -1.5
    p.xaxis.axis_label = "Countries"
    p.yaxis.axis_label = "Ratio of Positive:Negative"
    p.xgrid.grid_line_color = None

    p.title.text_font_size = '20pt'
    p.xaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_text_font_size = "20pt"

    # Bokeh Library
    from bokeh.io import output_file
    from bokeh.models.widgets import Tabs, Panel

    # Increase the plot widths
    myPlot.plot_width = p.plot_width = 1000

    # Create two panels, one for each conference
    myPlotPanel = Panel(child=myPlot, title='Positives vs. Negatives')
    pPanel = Panel(child=p, title='Sentiment Analysis Ratios')

    # Assign the panels to Tabs
    tabs = Tabs(tabs=[pPanel, myPlotPanel])

    # Show the tabbed layout
    show(tabs)


# written by Jeremy Tan
# How long did a video stay trending?
def makeVideoTrending(full_trending_df, full_trending_df_fill):
    new = pd.DataFrame(full_trending_df_fill.groupby([full_trending_df_fill.index,'country']).count()['title'].sort_values(ascending=False)).reset_index()
    new.head(), new.tail()
    video_list,max_list = list(),list()
    country_list = full_trending_df.groupby(['country']).count().index

    bokeh.io.reset_output()
    bokeh.io.output_notebook()

    # gabs countries and grabs the times it appears in the dataframe
    # print(country_list)
    for c in country_list:
        video_list.append(new[new['country']==c]['title'].value_counts().sort_index())
        max_list.append(max(new[new['country']==c]['title'].value_counts().sort_index().index))
    # print(video_list)
    # print(max_list)

    # make line plots for the different countries (repeat another 9 times)
    first_fig = figure(x_axis_type='linear',
                 plot_height=300, plot_width=600,
                 title='How long did a video stay trending?',
                 x_axis_label='Conccurent Appearences (days)', y_axis_label=country_list[0],
                 toolbar_location=None)

    # plot for different countries (repeat another 9 times)
    first_fig.line(x=video_list[0].index, y=video_list[0],
             color='gray', line_width=1)

    second_fig = figure(x_axis_type='linear',
                 plot_height=300, plot_width=600,
                 title='How long did a video stay trending?',
                 x_axis_label='Conccurent Appearences (days)', y_axis_label=country_list[1],
                 toolbar_location=None)

    second_fig.line(x=video_list[1].index, y=video_list[1],
             color='gray', line_width=1)

    third_fig = figure(x_axis_type='linear',
                 plot_height=300, plot_width=600,
                 title='How long did a video stay trending?',
                 x_axis_label='Conccurent Appearences (days)', y_axis_label=country_list[2],
                 toolbar_location=None)

    third_fig.line(x=video_list[2].index, y=video_list[2],
             color='gray', line_width=1)


    fourth_fig = figure(x_axis_type='linear',
                 plot_height=300, plot_width=600,
                 title='How long did a video stay trending?',
                 x_axis_label='Conccurent Appearences (days)', y_axis_label=country_list[3],
                 toolbar_location=None)

    fourth_fig.line(x=video_list[3].index, y=video_list[3],
             color='gray', line_width=1)

    fifth_fig = figure(x_axis_type='linear',
                 plot_height=300, plot_width=600,
                 title='How long did a video stay trending?',
                 x_axis_label='Conccurent Appearences (days)', y_axis_label=country_list[4],
                 toolbar_location=None)

    fifth_fig.line(x=video_list[4].index, y=video_list[4],
             color='gray', line_width=1)

    sixth_fig = figure(x_axis_type='linear',
                 plot_height=300, plot_width=600,
                 title='How long did a video stay trending?',
                 x_axis_label='Conccurent Appearences (days)', y_axis_label=country_list[5],
                 toolbar_location=None)

    sixth_fig.line(x=video_list[5].index, y=video_list[5],
             color='gray', line_width=1)


    seventh_fig = figure(x_axis_type='linear',
                 plot_height=300, plot_width=600,
                 title='How long did a video stay trending?',
                 x_axis_label='Conccurent Appearences (days)', y_axis_label=country_list[6],
                 toolbar_location=None)

    seventh_fig.line(x=video_list[6].index, y=video_list[6],
             color='gray', line_width=1)

    eight_fig = figure(x_axis_type='linear',
                 plot_height=300, plot_width=600,
                 title='How long did a video stay trending?',
                 x_axis_label='Conccurent Appearences (days)', y_axis_label=country_list[7],
                 toolbar_location=None)

    eight_fig.line(x=video_list[7].index, y=video_list[7],
             color='gray', line_width=1)

    ninth_fig = figure(x_axis_type='linear',
                 plot_height=300, plot_width=600,
                 title='How long did a video stay trending?',
                 x_axis_label='Conccurent Appearences (days)', y_axis_label=country_list[8],
                 toolbar_location=None)

    ninth_fig.line(x=video_list[8].index, y=video_list[8],
             color='gray', line_width=1)

    tenth_fig = figure(x_axis_type='linear',
                 plot_height=300, plot_width=600,
                 title='How long did a video stay trending?',
                 x_axis_label='Conccurent Appearences (days)', y_axis_label=country_list[9],
                 toolbar_location=None)

    tenth_fig.line(x=video_list[9].index, y=video_list[9],
             color='gray', line_width=1)

    #Add panels for tabs
    first_panel = Panel(child=first_fig, title='Canada')
    second_panel = Panel(child=second_fig, title='Germany')
    third_panel = Panel(child=third_fig, title='France')
    fourth_panel = Panel(child=fourth_fig, title='Great Britan')
    fifth_panel = Panel(child=fifth_fig, title='India')
    sixth_panel = Panel(child=sixth_fig, title='Japan')
    seventh_panel = Panel(child=seventh_fig, title='Korea')
    eighth_panel = Panel(child=eight_fig, title='Mexico')
    ninth_panel = Panel(child=ninth_fig, title='Russia')
    tenth_panel = Panel(child=tenth_fig, title='United States')


    # Assign the panels to Tabs
    tabs = Tabs(tabs=[first_panel, second_panel, third_panel, fourth_panel, fifth_panel, sixth_panel, seventh_panel, eighth_panel, ninth_panel, tenth_panel])
    show(tabs)


# Written by Jeremy Tan
# What is the correlation between views, likes, disklies, and comment count
def makeScatter(full_trending_df):
    # make a scatter matrix to visualzie correlations
    fig = px.scatter_matrix(full_trending_df, dimensions=['views', 'likes', 'dislikes', 'comments', 'title_length', 'description_length'])
    fig.update_traces(opacity=0.3, showupperhalf=False)
    fig.update_layout(title = "Correlations", width = 900, height = 900)
    fig.show()

# Written by Jeremy Tan
def trendingCategories(country, name):
    cat = country['category'].value_counts().reset_index()
    pop = px.bar(x=cat['category'], y=cat['index'], orientation='h')
    pop.update_layout(title_text="Most successful categories of trend videos in " + name, height=400)
    pop.update_xaxes(title_text="Number of Videos")
    pop.update_yaxes(title_text="Categories")
    pop.show()

# Written by Jeremy Tan
def nontrendingCategories(country, name):
    cat = country['category'].value_counts().reset_index()
    pop = px.bar(x=cat['category'], y=cat['index'], orientation='h')
    pop.update_layout(title_text="Categories that fail to trend in " + name, height=400)
    pop.update_xaxes(title_text="Number of Videos")
    pop.update_yaxes(title_text="Categories")
    pop.show()

# Written by Jeremy Tan
# What is the correlation between views, likes, disklies, and comment count
def makeScatter(full_trending_df):
    # make a scatter matrix to visualzie correlations
    fig = px.scatter_matrix(full_trending_df, dimensions=['views', 'likes', 'dislikes', 'comments', 'title_length', 'description_length'])
    fig.update_traces(opacity=0.3, showupperhalf=False)
    fig.update_layout(title = "Correlations", width = 900, height = 900)
    fig.show()

# Written by Jeremy Tan
def makeHeatMap(correlations):
    fig = go.Figure(data=go.Heatmap(
        z=correlations.values,
        x=correlations.index,
        y=correlations.index,
        colorscale="Blues",
        zmin=-1,
        zmax=1
    ))
    fig.update_layout(title='Correlations between title, description, comments, dislikes, likes, and views')
    fig.show()

# Written by Jeremy Tan
# What is the correlation between views, likes, disklies, and comment count in categories?
def makeCategoryHeatMap(corr_list, categories):
    fig = go.Figure()
    for idx, corr in enumerate(corr_list):
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=corr.index,
                y=corr.index,
                colorscale="Blues",
                zmin=-1,
                zmax=1,
                visible=False))
    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                buttons=list([
                    dict(label=cat,
                         method="update",
                         args=[{"visible": [False if sub_idx != idx else True for sub_idx, sub_cat in enumerate(categories)]},
                               {"title": "Correlation heatmap for category: " + cat}])
                    for idx, cat in enumerate(categories)
                ] )
            )
        ])
    fig.show()

# Written by Jeremy Tan
def trendingCategories(country, name):
    cat = country['category'].value_counts().reset_index()
    pop = px.bar(x=cat['category'], y=cat['index'], orientation='h')
    pop.update_layout(title_text="Most successful categories of trend videos in " + name, height=400)
    pop.update_xaxes(title_text="Number of Videos")
    pop.update_yaxes(title_text="Categories")
    pop.show()

# Written by Jeremy Tan
def nontrendingCategories(country, name):
    cat = country['category'].value_counts().reset_index()
    pop = px.bar(x=cat['category'], y=cat['index'], orientation='h')
    pop.update_layout(title_text="Categories that fail to trend in " + name, height=400)
    pop.update_xaxes(title_text="Number of Videos")
    pop.update_yaxes(title_text="Categories")
    pop.show()

# Written by Jeremy Tan
def likes_to_categories(country, name, log):
    country['likes_log'] = np.log(country['likes'] + 1)
    country['views_log'] = np.log(country['views'] + 1)
    if "likes" in log:
        fig = px.box(country, y=log, x='category')
        fig.update_layout(title_text="Like Distribuition by Category in " + name, height=700)
        fig.update_yaxes(title_text="Likes(log)")
        fig.show()
    else:
        fig = px.box(country, y=log, x='category')
        fig.update_layout(title_text="View Distribuition by Category in " + name, height=700)
        fig.update_yaxes(title_text="Views(log)")
        fig.show()

# Written by Jeremy Tan
def videos_top(country, name):
    tmp = country.channel_title.value_counts()[:25]

    pop = px.bar(x=tmp, y=tmp.index, orientation='h')
    pop.update_layout(title_text="Top Channels", height=700)
    pop.update_xaxes(title_text="# times channel reached trending")
    pop.update_yaxes(title_text="channel name")

    pop.show()

# Written by Jeremy Tan
def showHours(full_trending_df):
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Hours Videos are Published", "Like Rate")
    )

    s = full_trending_df['hour'].value_counts()
    new = pd.DataFrame({'Hour':s.index, 'Count':s.values})

    fig.add_trace(go.Bar(x=new['Hour'], y=new['Count']), row=1, col=1)
    fig.add_trace(go.Box(x=full_trending_df["hour"], y=full_trending_df["like_rate"]), row=2, col=1)
    fig.update_xaxes(title_text="Hour", row=1, col=1)
    fig.update_xaxes(title_text="Hour", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Like Rate(log)", row=2, col=1)
    fig.show()


# Written by Jeremy Tan
def likes_to_categories(country, name, log):
    country['likes_log'] = np.log(country['likes'] + 1)
    country['views_log'] = np.log(country['views'] + 1)
    if "likes" in log:
        fig = px.box(country, y=log, x='category')
        fig.update_layout(title_text="Like Distribuition by Category in " + name, height=700)
        fig.update_yaxes(title_text="Likes(log)")
        fig.show()
    else:
        fig = px.box(country, y=log, x='category')
        fig.update_layout(title_text="View Distribuition by Category in " + name, height=700)
        fig.update_yaxes(title_text="Views(log)")
        fig.show()

# Written by Jeremy Tan
def videos_top(country, name):
    tmp = country.channel_title.value_counts()[:25]

    pop = px.bar(x=tmp, y=tmp.index, orientation='h')
    pop.update_layout(title_text="Top Channels", height=700)
    pop.update_xaxes(title_text="# times channel reached trending")
    pop.update_yaxes(title_text="channel name")

    pop.show()

# Written by Jeremy Tan
def visualize_most(my_df, column):
    # sort df by column
    sorted_df = my_df.sort_values(column, ascending=False).iloc[:10]
    # make a bar plot
    pop = px.bar(x=sorted_df['title'], y=sorted_df[column])
    pop.update_layout(title_text="Top Ten " + column, height=700)
    pop.update_yaxes(title_text="#")
    pop.update_xaxes(title_text="Video Title")
    pop.show()

# Written by Jeremy Tan
def engagement(full_trending_df):
    country_list = full_trending_df.groupby(['country']).count().index
    p=['views', 'likes', 'dislikes', 'comments']
    calcs = list()
    for i, typ in enumerate(p):
        calc = list()
        for c in country_list:
            calc.append(full_trending_df[full_trending_df['country'] == c][typ].agg('sum') / len(full_trending_df[full_trending_df['country'] == c].index.unique()))
        calcs.append(calc)

    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("Views", "Likes", "Dislikes", "Comments")
    )

    # Add traces
    fig.add_trace(go.Bar(x=list(country_list), y=calcs[0]), row=1, col=1)
    fig.add_trace(go.Bar(x=list(country_list), y=calcs[1]), row=1, col=2)
    fig.add_trace(go.Bar(x=list(country_list), y=calcs[2]), row=2, col=1)
    fig.add_trace(go.Bar(x=list(country_list), y=calcs[3]), row=2, col=2)
    fig.update_layout(title_text="Viewer Engagement", height=700)

    fig.update_xaxes(title_text="countries", row=1, col=1)
    fig.update_xaxes(title_text="countries", row=1, col=2)
    fig.update_xaxes(title_text="countries", row=2, col=1)
    fig.update_xaxes(title_text="countries", row=2, col=2)

    fig.update_yaxes(title_text="num", row=1, col=1)
    fig.update_yaxes(title_text="num", row=1, col=2)
    fig.update_yaxes(title_text="num", row=2, col=1)
    fig.update_yaxes(title_text="num", row=2, col=2)
    fig.show()
    
# written by Timothy Le
# bar graph of most common tags within a dataframe no dups
def df_to_bar(country_df, title):
    common_tags = no_dups_common_tags(country_df)
    tagdf = pd.DataFrame(list(common_tags.items()), columns=['tag', 'count']).head(50)
    tagcds = ColumnDataSource(tagdf.head(50))
    
    bokeh.io.reset_output()
    bokeh.io.output_notebook()
    colorlist = Category20[20] + Category20[20] + Category20[20]  
    hover = HoverTool()
    hover.tooltips = [
        ("Tag", "@tag"),
        ("Count" , "@count")
    ]

    pBar = figure(height=600, width=500, y_range=tagdf['tag'], title=title+" Most Common Tags", x_axis_label='Count', tools=[hover])
    pBar.hbar(source=tagcds, height=.5, y='tag', right='count', fill_color=factor_cmap('tag', palette=colorlist, factors=tagdf['tag']))
    show(pBar)

# written by Timothy Le
# get most common tags from a country df and removes duplicates within df based on title of video
def no_dups_common_tags(country_df):
    nodups_df = country_df.drop_duplicates(subset ="title", keep = "last")
    tags = nodups_df['tags'].to_string(index=False, header=False)
    split_tags = [i.replace('"', '') for i in tags.split("|")]
    stop_words = stopwords.words('english')
    filtered_tags = [word for word in split_tags if word not in stop_words]
    fdist = FreqDist(split_tags)
    most_popular_tags = fdist.most_common(1000)
    return dict(most_popular_tags)

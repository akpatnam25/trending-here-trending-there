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
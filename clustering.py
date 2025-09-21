import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import kmeans
from sklearn.decomposition import PCA
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.api_key = os.getenv("GEMINI API KEY")